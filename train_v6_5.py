import random
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error

from datapreprocessing_deap_ECG import DEAP_ECG_RegressionDataset, DEAP_GSR_RegressionDataset
from model_v6_5 import (ECGKANModel, seed_everything, SplineWeightLayer, FastKANLayer,
                        AatyFusionModel, setup_fusion_freeze_strategy)

# --- 1. 损失函数 (保持你要求的 fusion_layer 识别逻辑) ---
def kan_optimal_loss(outputs, targets, model, epoch,
                     lamb=1e-3, mu=5e-3, gamma_smth=2e-2, mu_att=3e-4, reg=1e-5):
    device = outputs.device
    mse_loss = F.mse_loss(outputs, targets)
    mean_bias_weight = 0.1 + (epoch / 50) * 0.4
    mean_bias_loss = torch.pow(outputs.mean(dim=0) - targets.mean(dim=0), 2).sum()

    reg_loss, entropy_loss, smooth_loss, attn_loss, stability_reg = [torch.tensor(0.0).to(device) for _ in range(5)]

    for name, m in model.named_modules():
        is_fusion_target = "fusion_layer" in name
        if isinstance(m, FastKANLayer):
            w = m.spline_weights
            if w.shape[-1] > 2:
                diff2 = w[:, :, 2:] - 2 * w[:, :, 1:-1] + w[:, :, :-2]
                smooth_loss += torch.mean(diff2 ** 2)
            if is_fusion_target:
                reg_loss += (torch.mean(torch.abs(w)) + 0.1 * torch.mean(w ** 2))
                edge_strengths = torch.sqrt(torch.mean(w ** 2, dim=-1) + m.omiga ** 2 + 1e-8)
                prob = F.softmax(edge_strengths / 0.3, dim=0)
                entropy_loss += (-torch.sum(prob * torch.log(prob + 1e-8)) / m.out_neurons)
            if hasattr(m, 'last_W1') and m.last_W1 is not None:
                attn_loss += (torch.sum(torch.abs(m.last_W1)) / (m.last_W1.numel() + 1e-6))
        if isinstance(m, SplineWeightLayer):
            stability_reg += (m.alpha ** 2 + m.beta ** 2 + m.theta ** 2 + m.gamma ** 2).sum()

    warmup = min(1.0, epoch / 5.0)
    total_loss = mse_loss + mean_bias_weight * mean_bias_loss + \
                 warmup * (lamb * reg_loss + mu * entropy_loss + gamma_smth * smooth_loss + mu_att * attn_loss + reg * stability_reg)
    return total_loss, {"mse": mse_loss.item(), "total": total_loss.item()}


def run_unified_trainer(model, train_loaders, val_loaders, epochs=50, lr=1e-3, nerve_cluster="old", fresh_start=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    save_dir = "/content/drive/MyDrive/deap/fusion_results" if nerve_cluster == "fusion" else "/content/drive/MyDrive/deap/expert_results"
    os.makedirs(save_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(save_dir, f"{nerve_cluster}_latest_checkpoint_v2.pth")
    best_pth_path = os.path.join(save_dir, f"{nerve_cluster}_best_v2.pth")

    if nerve_cluster == "old":
        if hasattr(model.layer5, 'weight_layers'):
            model.layer5.weight_layers[4].requires_grad_(False)

        slow_keys, mid_keys = ['alpha', 'omiga'], ['beta', 'tau', 'temperature', 'gamma']
        params_slow, params_mid, params_base, params_expert_unfrozen = [],  [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if any(key in name for key in slow_keys):
                params_slow.append(param)
            elif any(key in name for key in mid_keys):
                params_mid.append(param)
            else:
                params_base.append(param)

        optimizer = optim.AdamW([
            {'params': params_base, 'lr': lr},
            {'params': params_mid, 'lr': lr * 2.0},
            {'params': params_slow, 'lr': lr * 3.0}
        ], weight_decay=1e-5)

    else:
        setup_fusion_freeze_strategy(model)
        slow_keys, mid_keys = ['alpha', 'omiga'], ['beta', 'tau', 'temperature', 'gamma']
        params_slow, params_mid, params_base, params_expert_unfrozen = [], [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if "fusion_layer" in name:
                if any(key in name for key in slow_keys):
                    params_slow.append(param)
                elif any(key in name for key in mid_keys):
                    params_mid.append(param)
                else:
                    params_base.append(param)
            else:
                params_expert_unfrozen.append(param)

        optimizer = optim.AdamW([
            {'params': params_expert_unfrozen, 'lr': lr * 0.1},
            {'params': params_base, 'lr': lr},
            {'params': params_mid, 'lr': lr * 2.0},
            {'params': params_slow, 'lr': lr * 3.0}
        ], weight_decay=1e-5)


    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    early_stop_patience = 15
    early_stop_counter = 0

    start_epoch = 0
    best_val_rmse = float('inf')
    if os.path.exists(latest_ckpt_path):
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        early_stop_counter = checkpoint.get('early_stop_counter', 0)
        start_epoch = checkpoint['epoch'] + 1
        best_val_rmse = checkpoint['best_val_rmse']

        # 恢复所有随机状态
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng_state']])
        np.random.set_state(checkpoint['np_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

        print(f"\n>>> 成功恢复断点：从 Epoch {start_epoch} 继续训练，历史最佳 RMSE: {best_val_rmse:.4f}")

    # 2. 初始状态检测 (适配双路)
    if fresh_start:
        model.eval()
        init_mse_sum, init_total = 0, 0
        with torch.no_grad():
            # 这里判断 val_loaders 是否为元组
            target_loader = zip(*val_loaders) if isinstance(val_loaders, tuple) else val_loaders
            for data in target_loader:
                if nerve_cluster == "fusion":
                    (ex, _), (gx, vy) = data
                    v_out = model(ex.to(device), gx.to(device))
                else:
                    vx, vy = data[0].to(device), data[1].to(device)
                    v_out = model(vx, mode='independent')
                init_mse_sum += F.mse_loss(v_out, vy.to(device), reduction='sum').item()
                init_total += vy.size(0)
        print(f"\n[Initial State] Val RMSE: {np.sqrt(init_mse_sum / init_total):.4f}")

    # 3. 主训练循环
    for epoch in range(start_epoch, epochs):
        model.train()
        # 构建数据迭代器
        loader_iter = zip(*train_loaders) if nerve_cluster == "fusion" else train_loaders
        pbar = tqdm(loader_iter, desc=f"{nerve_cluster.upper()} Epoch {epoch + 1}",
                    total=len(train_loaders[0] if nerve_cluster == "fusion" else train_loaders))

        for data in pbar:
            optimizer.zero_grad()
            if nerve_cluster == "fusion":
                (ex, _), (gx, vy) = data  # 取 ECG-X, GSR-X 和 共享的 Label
                outputs = model(ex.to(device), gx.to(device))
                batch_y = vy.to(device)
            else:
                batch_x, batch_y = data[0].to(device), data[1].to(device)
                outputs = model(batch_x, mode='independent')

            loss, info = kan_optimal_loss(outputs, batch_y, model, epoch)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(mse=f"{info['mse']:.4f}")

        # 4. 全量指标评估 (适配双路)
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            eval_iter = zip(*val_loaders) if nerve_cluster == "fusion" else val_loaders
            for data in eval_iter:
                if nerve_cluster == "fusion":
                    (ex, _), (gx, vy) = data
                    v_out = model(ex.to(device), gx.to(device))
                else:
                    vx, vy = data[0].to(device), data[1].to(device)
                    v_out = model(vx, mode='independent')
                all_preds.append(v_out)
                all_targets.append(vy.to(device))

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_regression_metrics(all_preds, all_targets)
        avg_val_rmse = np.mean([m['RMSE'] for m in metrics.values()])

        print(f"\n[Epoch {epoch + 1} Evaluation Summary]")
        print(f"{'Dimension':<12} | {'RMSE(1-9)':<10} | {'MAE':<8} | {'PCC':<8} | {'CCC':<8} | {'R2':<8}")
        print("-" * 75)
        for dim, m in metrics.items():
            print(f"{dim:<12} | {m['RMSE'] * 4.5:<10.4f} | {m['MAE']:<8.4f} | {m['PCC']:<8.4f} | {m['CCC']:<8.4f} | {m['R2']:<8.4f}")

        # --- 4. 深度动力学与拓扑审计监控 ---
        alphas, betas, thetas, gammas = [], [], [], []
        taus, omigas, temps = [], [], []

        for m in model.modules():
            # 提取 SplineWeightLayer 中的组件权重
            if isinstance(m, SplineWeightLayer):
                alphas.append(m.alpha.abs().mean().item())
                betas.append(m.beta.abs().mean().item())
                thetas.append(m.theta.abs().mean().item())
                gammas.append(m.gamma.abs().mean().item())

            # 提取 FastKANLayer 中的路由与残差参数
            if isinstance(m, FastKANLayer):
                taus.append(m.tau.abs().mean().item())
                omigas.append(m.omiga.abs().mean().item())
                temps.append(m.temperature.abs().mean().item())

        # 打印组件权重 (Alpha, Beta, Theta, Gamma)
        print(f"\n  [Components] Alpha(I): {np.mean(alphas):.3f} | Beta(W1): {np.mean(betas):.3f} | "
              f"Theta(W3): {np.mean(thetas):.3f} | Gamma(Res): {np.mean(gammas):.3f}")

        # 打印路由参数 (Tau, Omega, Temp)
        print(f"  [Routing]    Tau(Gate): {np.mean(taus):.4f} | Omega(Lin): {np.mean(omigas):.4f} | Temp: {np.mean(temps):.3f}")

        # 打印每一层的导通率 (Active Ratio)
        conn = model.get_active_conn_info()
        print(f"  [Topology]   审计：")
        for layer_name in sorted(conn.keys()):
            d = conn[layer_name]
            # ratio 反映了有多少边突破了 Tau 的过滤
            print(f"    {layer_name:<10}: {d['ratio'] * 100:>5.1f}% active | tau_avg: {d['tau_mean']:.4f}")

        current_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_rmse': best_val_rmse,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'np_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'early_stop_counter': early_stop_counter
        }
        torch.save(current_checkpoint, latest_ckpt_path)

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            torch.save(model.state_dict(), best_pth_path)
            early_stop_counter = 0
            print(f">>> 发现更佳模型，已更新 best.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience: break

        scheduler.step()
        print("-" * 75)


def compute_regression_metrics(preds, targets):
    """
    计算并返回 RMSE, MAE, PCC, CCC, R2
    preds/targets: [N, 4]
    """
    metrics = {}
    dim_names = ['Valence', 'Arousal', 'Dominance', 'Liking']
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

    for i, name in enumerate(dim_names):
        p = preds_np[:, i]
        t = targets_np[:, i]

        rmse = np.sqrt(np.mean((p - t) ** 2))
        mae = mean_absolute_error(t, p)
        pcc, _ = pearsonr(p, t)
        r2 = r2_score(t, p)

        # 4. 一致性相关系数 (CCC)
        # 公式: (2 * corr * std_p * std_t) / (var_p + var_t + (mu_p - mu_t)^2)
        mu_p, mu_t = np.mean(p), np.mean(t)
        var_p, var_t = np.var(p), np.var(t)
        std_p, std_t = np.std(p), np.std(t)
        ccc = (2 * pcc * std_p * std_t) / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8)

        metrics[name] = {'RMSE': rmse, 'MAE': mae, 'PCC': pcc, 'CCC': ccc, 'R2': r2}
    return metrics


def calculate_class_weights(dataloader, boost_dict={1:2.0, 3:1.5}):
    all_labels = []
    for _, y in dataloader: all_labels.extend(y.cpu().numpy())
    counts = Counter(all_labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = torch.tensor([total / (num_classes * counts[i]) for i in range(num_classes)]).sqrt()
    for idx, factor in boost_dict.items():
        if idx < num_classes:
            weights[idx] = weights[idx] * factor
            print(f">>> 类别 {idx} (S-Premature) 权重已强化: 原权重 x {factor}")
    final_weights = weights / weights.mean()
    return final_weights


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    seed_everything(42)
    DATA_DIR = "/content/drive/MyDrive/deap/dataset/data_preprocessed_python"

    # 设置当前训练模式
    current_mode = "fusion"  # "old" 为专家训练, "fusion" 为互注意力训练

    if current_mode == "fusion":
        # 1. 加载双模态数据 (同步划分)
        ecg_full = DEAP_ECG_RegressionDataset(DATA_DIR, subject_list=range(1, 33), window_size=200, stride=100,
                                              is_train=True)
        gsr_full = DEAP_GSR_RegressionDataset(DATA_DIR, subject_list=range(1, 33), window_size=512, stride=96,
                                              is_train=True)
        ecg_val = DEAP_ECG_RegressionDataset(DATA_DIR, subject_list=range(1, 33), window_size=200, is_train=False)
        gsr_val = DEAP_GSR_RegressionDataset(DATA_DIR, subject_list=range(1, 33), window_size=512, is_train=False)

        assert len(ecg_full) == len(gsr_full), f"样本数不匹配: ECG({len(ecg_full)}) vs GSR({len(gsr_full)})"

        # --- 【新增：40% 随机采样逻辑】 ---
        # 1. 生成所有样本的索引
        num_train = len(ecg_full)
        indices = np.arange(num_train)
        # 2. 随机打乱索引
        np.random.shuffle(indices)
        # 3. 截取前 40%
        subset_size = int(num_train * 0.5)
        train_indices = indices[:subset_size]

        # 4. 使用 Subset 包装，确保 ECG 和 GSR 使用完全相同的样本索引（保持时序对齐）
        ecg_train = Subset(ecg_full, train_indices)
        gsr_train = Subset(gsr_full, train_indices)
        print(f">>> 已启用子集训练：从 {num_train} 个样本中随机抽取 {len(ecg_train)} 个 (50%)")

        # 2. 初始化专家并加载权重
        expert_ecg = ECGKANModel(kernel_size=[31, 25, 17, 9, 3], ks=[7, 17, 31], grid_size=40, seq_len=200)
        expert_gsr = ECGKANModel(kernel_size=[63, 47, 31, 15, 7], ks=[15, 31, 63], grid_size=64, seq_len=512)

        expert_ecg.load_state_dict(torch.load("ecg_best.pth", map_location='cpu', weights_only=False), strict=False)
        expert_gsr.load_state_dict(torch.load("gsr_best.pth", map_location='cpu', weights_only=False), strict=False)

        # 3. 组建融合模型
        model = AatyFusionModel(expert_ecg, expert_gsr, seq_len=512)

        # --- 【优化：提升 Batch Size 和多线程以提速】 ---
        bs = 24  # 从 20 提升到 64，显存允许的话可以设为 128
        shuffle = False  # 必须为 False 以保证 zip 同步
        g = torch.Generator()
        g.manual_seed(42)

        train_loaders = (
            DataLoader(ecg_train, batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g),
            DataLoader(gsr_train, batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        )
        val_loaders = (
            DataLoader(ecg_val, batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g),
            DataLoader(gsr_val, batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        )

    else:
        # 专家模式逻辑...
        train_ds = DEAP_ECG_RegressionDataset(DATA_DIR, subject_list=range(1, 33), is_train=True)

        # 专家模式也可以加 40% 采样
        indices = np.arange(len(train_ds))
        np.random.shuffle(indices)
        train_ds = Subset(train_ds, indices[:int(len(train_ds) * 0.5)])

        model = ECGKANModel(kernel_size=[31, 25, 17, 9, 3], seq_len=200)
        train_loaders = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
        val_loaders = DataLoader(DEAP_ECG_RegressionDataset(DATA_DIR, subject_list=range(1, 33), is_train=False),
                                 batch_size=64)

    # 4. 执行统一训练
    run_unified_trainer(model, train_loaders, val_loaders, nerve_cluster=current_mode, fresh_start=False)