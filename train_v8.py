import torch

import torch.nn.functional as F
import os
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
import numpy as np

from database_mit import prepare_ecg_data
from model_v6_5 import ECGKANModel, seed_everything, SplineWeightLayer, FastKANLayer


def kan_optimal_loss(outputs, targets, model, epoch,
                     lamb=2e-4,  # 样条权重正则系数 (L1+L2)
                     mu=1e-2,  # 路径熵系数 (引导特征解耦)
                     gamma_smth=4e-2,  # 样条平滑度系数 (抑制高频噪声)
                     mu_att=1e-3,  # 注意力矩阵 W1 稀疏系数
                     reg=1e-5,  # SplineWeightLayer 标量稳定性系数
                     label_smooth=0):  # 逻辑标签平滑系数 (心电建议 0.05-0.1)
    """
    针对多分类心电任务优化的 KAN 损失函数。
    修复了 AttributeError: 'FastKANLayer' object has no attribute 'base_scales'。
    """
    device = outputs.device

    # --- 1. 核心分类损失：针对心电 5 分类使用 CrossEntropy ---
    # outputs: [Batch, 5], targets: [Batch] (类别索引 0-4)
    ce_loss = F.cross_entropy(
        outputs,
        targets,
        weight=getattr(model, 'class_weights', None),
        label_smoothing=label_smooth
    )

    # 初始化各类正则项
    reg_loss = torch.tensor(0.0).to(device)
    entropy_loss = torch.tensor(0.0).to(device)
    smooth_loss = torch.tensor(0.0).to(device)
    attn_loss = torch.tensor(0.0).to(device)
    stability_reg = torch.tensor(0.0).to(device)

    for m in model.modules():
        # --- 2. FastKANLayer 结构正则 ---
        if isinstance(m, FastKANLayer):
            # A. 权重正则：L1 (促进稀疏) + 0.1 * L2 (抑制爆炸)
            w = m.spline_weights
            reg_loss = reg_loss + (torch.mean(torch.abs(w)) + 0.1 * torch.mean(w ** 2))

            # B. 样条平滑：二阶导数惩罚，确保激活函数曲线平滑
            diff2 = w[:, :, 2:] - 2 * w[:, :, 1:-1] + w[:, :, :-2]
            smooth_loss = smooth_loss + torch.mean(diff2 ** 2)

            # C. 路径熵正则：引导模型进行有效的通路选择
            # 修正点：将 m.base_scales 替换为 m.omiga，以匹配你模型中的定义
            edge_strengths = torch.sqrt(torch.mean(w ** 2, dim=-1) + m.omiga ** 2 + 1e-8)

            # 使用 Softmax 归一化路径强度，温度设为 0.2 以平滑梯度
            prob = F.softmax(edge_strengths / 0.2, dim=0)
            entropy_loss = entropy_loss + (-torch.sum(prob * torch.log(prob + 1e-8)) / m.out_neurons)

            # D. W1 自注意力稀疏化
            if hasattr(m, 'last_W1') and m.last_W1 is not None:
                attn_loss = attn_loss + (torch.sum(torch.abs(m.last_W1)) / (m.last_W1.numel() + 1e-6))

        # --- 3. SplineWeightLayer 稳定性正则 ---
        if isinstance(m, SplineWeightLayer):
            # 约束 alpha, beta, theta, gamma 标量参数不要过大
            stability_reg = stability_reg + (m.alpha ** 2 + m.beta ** 2 + m.theta ** 2 + m.gamma ** 2).sum()

    # --- 4. 动态 Warmup ---
    # 前 4 轮优先拟合信号，随后逐渐加强结构化约束
    warmup = min(1.0, epoch / 5.0)

    total_loss = ce_loss + warmup * (
            lamb * reg_loss +
            mu * entropy_loss +
            gamma_smth * smooth_loss +
            mu_att * attn_loss +
            reg * stability_reg
    )

    return total_loss, {
        "ce": ce_loss.item(),
        "smth": smooth_loss.item(),
        "ent": entropy_loss.item(),
        "att": attn_loss.item(),
        "total": total_loss.item()
    }


def train_ecg_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    save_dir = "/content/drive/MyDrive/heart_wave"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 参数分组：强化结构演化动力学
    # 第一组：结构核心参数，给予 10x 学习率打破初始值锁定 (5e-3)
    slow_keys = ['alpha', 'omiga']
    # 第二组：路由与门控参数，给予 2x 学习率 (1e-3)
    mid_keys = ['beta', 'tau', 'temperature', 'gamma']

    params_slow, params_mid, params_base = [], [], []

    for name, param in model.named_parameters():
        if any(key in name for key in slow_keys):
            params_slow.append(param)
        elif any(key in name for key in mid_keys):
            params_mid.append(param)
        else:
            params_base.append(param)

    # 2. 构造差异化优化器
    optimizer = optim.AdamW([
        {'params': params_base, 'lr': lr},
        {'params': params_mid, 'lr': lr * 2.0},
        {'params': params_slow, 'lr': lr * 3.0}  # 提升至 10 倍速以对标 SOTA 演化速度
    ], weight_decay=1e-5)

    # 自动计算类别权重以应对不平衡数据集
    model.class_weights = calculate_class_weights(train_loader).to(device)

    # --- 关键修改：替换为余弦退火调度器 ---
    # T_max 为总轮数，使得学习率在训练结束前平滑降低到最低点
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.eval()
    init_correct, init_total, init_v_ce = 0, 0, 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            v_out = model(vx)
            _, v_info = kan_optimal_loss(v_out, vy, model, epoch=0)
            init_v_ce += v_info['ce']
            init_total += vy.size(0)
            init_correct += (v_out.argmax(1) == vy).sum().item()

    init_acc = 100 * init_correct / init_total
    print(f"\n[Initial State] Loss: {init_v_ce / len(val_loader):.6f}, Val Acc: {init_acc:.2f}%")
    print("-" * 75)

    best_val_acc = 0.0
    early_stop_counter = 0
    early_stop_patience = 10  # 配合余弦退火，适当增加耐心

    for epoch in range(epochs):
        model.train()
        train_correct, train_total = 0, 0
        epoch_ce_loss, epoch_ent_loss, epoch_total_loss = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss, info = kan_optimal_loss(outputs, batch_y, model, epoch)
            loss.backward()

            # 梯度裁剪：放宽至 2.0 以允许结构参数有更强的爆发力
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_ce_loss += info['ce']
            epoch_ent_loss += info['ent']
            train_correct += (outputs.argmax(1) == batch_y).sum().item()
            train_total += batch_y.size(0)

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(ce=f"{info['ce']:.3f}", lr=f"{current_lr:.1e}",
                             acc=f"{100 * train_correct / train_total:.2f}%")

        # 验证逻辑
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)
                val_correct += (v_out.argmax(1) == vy).sum().item()
                val_total += vy.size(0)

        val_acc = 100 * val_correct / val_total
        avg_total = epoch_total_loss / len(train_loader)
        avg_ce = epoch_ce_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        print(f"\n[Epoch {epoch + 1} Result] Val Acc: {val_acc:.2f}% | Train Acc: {train_acc:.2f}%")
        print(f"  Avg Loss: {avg_total:.6f} (CE: {avg_ce:.4f})")

        # 核心动力学参数监控
        alphas, betas, thetas, gammas, taus, temps, omigas = [], [], [], [], [], [], []
        for m in model.modules():
            if isinstance(m, SplineWeightLayer):
                alphas.append(m.alpha.abs().mean().item())
                betas.append(m.beta.abs().mean().item())
                thetas.append(m.theta.abs().mean().item())
                gammas.append(m.gamma.abs().mean().item())
            if isinstance(m, FastKANLayer):
                taus.append(m.tau.abs().mean().item())
                temps.append(m.temperature.abs().mean().item())
                omigas.append(m.omiga.abs().mean().item())

        print(f"  Weights: α={np.mean(alphas):.3f}, β={np.mean(betas):.3f}, θ={np.mean(thetas):.3f}, γ={np.mean(gammas):.3f}, τ={np.mean(taus):.3f}, ω={np.mean(omigas):.4f}")
        # 拓扑审计
        conn = model.get_active_conn_info()
        for layer_name in sorted(conn.keys()):
            d = conn[layer_name]
            print(f"    {layer_name}: {d['ratio'] * 100:.1f}% active | tau_avg: {d['tau_mean']:.4f}")

        # --- 修改调度器步进逻辑：余弦退火按 epoch 自动更新 ---
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, "heart_wave_2.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> 参数已更新，保存最佳模型 (Acc: {best_val_acc:.2f}%)")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"  >>> 验证集未提升 ({early_stop_counter}/{early_stop_patience})")

        if early_stop_counter >= early_stop_patience:
            print(f"\n[Early Stopping] 连续 {early_stop_patience} 轮未提升，提前结束。")
            break
        print("-" * 75)


def calculate_class_weights(dataloader, boost_dict={1 : 2.0, 3 : 1.5}):
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


if __name__ == "__main__":
    seed_everything(42)
    train_loader = prepare_ecg_data('/content/drive/MyDrive/mitbih/mitbih_train.csv', batch_size=72)
    val_loader = prepare_ecg_data('/content/drive/MyDrive/mitbih/mitbih_test.csv', batch_size=72)
    model = ECGKANModel()
    train_ecg_model(model, train_loader, val_loader)