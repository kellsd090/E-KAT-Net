import torch
import os
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
import numpy as np
import glob
from torch.utils.data import TensorDataset, DataLoader
from dataset_gal import WAYEEGDataset, calculate_pos_weights
from model import BCIKANModel, seed_everything, SplineWeightLayer, FastKANLayer


# For detailed annotations of this section, please refer to train.py of MIT-BIH ECG
def kan_optimal_loss(outputs, targets, model, epoch,
                     lamb=5e-4,  
                     mu=5e-4,  
                     gamma_smth=8e-2,  
                     mu_att=1e-3,  
                     reg=1e-5,  
                     label_smooth=0):  
    device = outputs.device
    pos_weights = getattr(model, 'pos_weights', None)                   
    with torch.no_grad():
        smoothed_targets = targets * (1.0 - label_smooth) + 0.5 * label_smooth
    bce_loss = F.binary_cross_entropy_with_logits(
        outputs,
        smoothed_targets,
        pos_weight=pos_weights)

    reg_loss = torch.tensor(0.0).to(device)
    entropy_loss = torch.tensor(0.0).to(device)
    smooth_loss = torch.tensor(0.0).to(device)
    attn_loss = torch.tensor(0.0).to(device)
    stability_reg = torch.tensor(0.0).to(device)

    for m in model.modules():
        if isinstance(m, FastKANLayer):
            w = m.spline_weights
            reg_loss = reg_loss + (torch.mean(torch.abs(w)) + 0.1 * torch.mean(w ** 2))
            diff2 = w[:, :, 2:] - 2 * w[:, :, 1:-1] + w[:, :, :-2]
            smooth_loss = smooth_loss + torch.mean(diff2 ** 2)
            edge_strengths = torch.sqrt(torch.mean(w ** 2, dim=-1) + m.omiga ** 2 + 1e-8)
            prob = F.softmax(edge_strengths / 0.2, dim=0)
            entropy_loss = entropy_loss + (-torch.sum(prob * torch.log(prob + 1e-8)) / m.out_neurons)
            if hasattr(m, 'last_W1') and m.last_W1 is not None:
                attn_loss = attn_loss + (torch.sum(torch.abs(m.last_W1)) / (m.last_W1.numel() + 1e-6))
        if isinstance(m, SplineWeightLayer):
            stability_reg = stability_reg + (m.alpha ** 2 + m.beta ** 2 + m.theta ** 2 + m.gamma ** 2).sum()

    warmup = min(1.0, epoch / 4.0)
    total_loss = bce_loss + warmup * (
            lamb * reg_loss +
            mu * entropy_loss +
            gamma_smth * smooth_loss +
            mu_att * attn_loss +
            reg * stability_reg)
    return total_loss, {
        "bce": bce_loss.item(),
        "smth": smooth_loss.item(),
        "ent": entropy_loss.item(),
        "att": attn_loss.item(),
        "total": total_loss.item()}


def train_gal_model(model, train_loader, val_loader, epochs=100, lr=1e-3, resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    save_dir = "/content/drive/MyDrive/GAL_KAN_Project"
    os.makedirs(save_dir, exist_ok=True)
  
    slow_keys = []
    mid_keys = ['alpha', 'temperature', 'gamma', 'beta', 'theta']
    params_slow, params_mid, params_base = [], [], []
    for name, param in model.named_parameters():
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

    start_epoch = 0
    best_val_acc = 0.0

    # Load breakpoint
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'best_val_acc' in checkpoint:
                best_val_acc = checkpoint['best_val_acc']
            print(f"recover from breakpoint successfully，starting Epoch: {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            start_epoch = 4
            best_val_acc = 95.36
            print("Warning: Optimizer state not found. Only restoring model weights.")
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
    else:
        model.eval()
        init_loss, init_correct = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device, dtype=torch.float)
                vy = vy.to(device, dtype=torch.float)
                v_out = model(vx)
                loss_v, _ = kan_optimal_loss(v_out, vy, model, 0)
                init_loss += loss_v.item()
                init_correct += ((torch.sigmoid(v_out) > 0.5) == (vy > 0.5)).float().mean().item()
            print(
                f"Initial -> Loss: {init_loss / len(val_loader):.4f} | Accuracy: {100 * init_correct / len(val_loader):.2f}%")
        print("=" * 74 + "\n")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, last_epoch=start_epoch - 1)
  
    early_stop_counter = 0
    early_stop_patience = 10

    for epoch in range(start_epoch, epochs):
        model.train()
        t_bce_sum = 0
        t_total_sum = 0
        t_correct_sum = 0
        processed_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for bx, by in pbar:
            bx, by = bx.to(device, dtype=torch.float), by.to(device, dtype=torch.float)
            optimizer.zero_grad()
            out = model(bx)
            loss, info = kan_optimal_loss(out, by, model, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            processed_batches += 1
            t_bce_sum += info['bce']
            t_total_sum += info['total']
            current_batch_acc = ((torch.sigmoid(out) > 0.5) == (by > 0.5)).float().mean().item()
            t_correct_sum += current_batch_acc
            pbar.set_postfix(
                loss=f"{t_total_sum / processed_batches:.4f}",
                bce=f"{t_bce_sum / processed_batches:.4f}",
                acc=f"{100 * t_correct_sum / processed_batches:.2f}%")
        avg_t_loss = t_total_sum / len(train_loader)
        avg_t_bce = t_bce_sum / len(train_loader)
        avg_t_acc = 100 * t_correct_sum / len(train_loader)
        model.eval()
        v_correct_sum, v_bce_sum, v_total_sum = 0, 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device, dtype=torch.float)
                vy = vy.to(device, dtype=torch.float)
                vo = model(vx)
                _, iv = kan_optimal_loss(vo, vy, model, epoch)
                v_bce_sum += iv['bce']
                v_correct_sum += ((torch.sigmoid(vo) > 0.5) == (vy > 0.5)).float().mean().item()
                v_total_sum += iv['total']
        avg_v_acc = 100 * v_correct_sum / len(val_loader)
        avg_v_bce = v_bce_sum / len(val_loader)
        avg_v_loss = v_total_sum / len(val_loader)

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

        print(f"\n[Epoch {epoch + 1}] Train Acc: {avg_t_acc:.2f}% | Val Acc: {avg_v_acc:.2f}%")
        print(f"            Train BCE: {avg_t_bce:.3f} | Val BCE: {avg_v_bce:.3f} | Train loss: {avg_t_loss:3f} | Total: {avg_v_loss:.3f}")
        print(f"  Weights: α={np.mean(alphas):.3f}, β={np.mean(betas):.3f}, θ={np.mean(thetas):.3f}, γ={np.mean(gammas):.3f}, τ={np.mean(taus):.3f}, ω={np.mean(omigas):.4f}")
        conn = model.get_active_conn_info()
        for layer_name in sorted(conn.keys()):
            d = conn[layer_name]
            print(f"    {layer_name}: {d['ratio'] * 100:.1f}% active | tau_avg: {d['tau_mean']:.4f}")

        scheduler.step()

        state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'best_val_acc': best_val_acc}
        torch.save(state, os.path.join(save_dir, "gal_last_checkpoint.pth"))
        if avg_v_acc > best_val_acc:
            best_val_acc = avg_v_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_gal_model_05.pth"))
            print(f" Break through the best accuracy")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print(f"\n[Early Stopping] 触发。最佳 Acc: {best_val_acc:.2f}%")
            break
        print("-" * 85)


def calculate_class_weights(dataloader, boost_dict={}):
    all_labels = []
    for _, y in dataloader:
        all_labels.extend(y.cpu().numpy())
    counts = Counter(all_labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = torch.tensor([total / (num_classes * counts[i]) for i in range(num_classes)], dtype=torch.float)
    weights = torch.sqrt(weights)
    for idx, factor in boost_dict.items():
        if idx < num_classes:
            weights[idx] = weights[idx] * factor
    final_weights = weights / weights.mean()
    return final_weights


if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    PATH_train = "/content/drive/MyDrive/grasp-and-lift-eeg-detection/train/train"
    PATH_past = "/content/drive/MyDrive/parameter.pth"

    all_subjects = list(range(1, 13))
    train_series = [1, 2, 3, 5, 6, 7]   # training set
    val_series = [4, 8]   # testing set
    p_weights = calculate_pos_weights(PATH_train, all_subjects, range(1, 9)).to(device)
    print(f" pos_weights: {p_weights}")
    train_ds = WAYEEGDataset(
        PATH_train,
        all_subjects,
        window_size=250,
        series_ids=train_series,
        is_train=True)
    current_stats = {
        'mean': train_ds.mean,
        'std': train_ds.std}
    print(f">>>  Mean Dtype: {train_ds.mean.dtype} |  Std Dtype: {train_ds.std.dtype}")

    val_ds = WAYEEGDataset(
        PATH_train,
        all_subjects,
        window_size=250,
        series_ids=val_series,
        is_train=False,
        stats=current_stats)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = BCIKANModel().to(device)
    model.pos_weights = p_weights

    print(f"\n Data preparation is complete. Number of training samples : {len(train_ds)} | Verification sample size : {len(val_ds)}")

    train_gal_model(model, train_loader, val_loader, epochs=100, lr=1e-3, resume_path=PATH_past)
