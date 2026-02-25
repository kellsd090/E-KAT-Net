import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
import numpy as np
from dataset_cifar10 import get_dataloader, add_coordinate_encoding
from model import ECGKANModel, seed_everything, SplineWeightLayer, FastKANLayer


def kan_optimal_loss(outputs, targets, model, epoch,
                     lamb=5e-5,  # Basic L1 Strength
                     mu=4e-3,  # Path competition intensity
                     gamma_smth=2e-2,  # Spline smoothness
                     mu_att=1e-3,
                     reg=1e-5):  # W1 sparsity constraint
    if epoch >= 60: reg = 7.5e-7
    if targets.ndim > 1:
        ce_loss = torch.mean(torch.sum(-targets * F.log_softmax(outputs, dim=-1), dim=-1))
    else:
        ce_loss = F.cross_entropy(outputs, targets, weight=getattr(model, 'class_weights', None))

    reg_loss = 0  
    entropy_loss = 0  
    smooth_loss = 0  
    attn_loss = 0  
    stability_reg = 0  

    for m in model.modules():
        if hasattr(m, 'spline_weights'):
            reg_loss += torch.sum(torch.abs(m.spline_weights)) / m.num_coeffs
            reg_loss += torch.sum(torch.abs(m.base_scales))

            w = m.spline_weights
            diff2 = w[:, :, 2:] - 2 * w[:, :, 1:-1] + w[:, :, :-2]
            smooth_loss += torch.mean(diff2 ** 2)

            edge_strengths = torch.mean(torch.abs(w), dim=-1) + torch.abs(m.base_scales)
            prob = F.softmax(edge_strengths / 0.1, dim=0)
            entropy_loss += -torch.sum(prob * torch.log(prob + 1e-8)) / m.out_neurons

            if hasattr(m, 'last_W1') and m.last_W1 is not None:
                attn_loss += torch.mean(m.last_W1 ** 2)

        if hasattr(m, 'W2') and hasattr(m, 'W3_conv'):
            reg_loss += torch.mean(torch.abs(m.W2)) + torch.mean(torch.abs(m.b))
            reg_loss += torch.mean(torch.abs(m.W3_conv.weight))
            stability_reg += (m.alpha ** 2 + m.beta ** 2 + m.theta ** 2 + m.gamma ** 2).sum()

    warmup = min(1.0, epoch / 5.0)
    total_loss = ce_loss + \
                 warmup * (lamb * reg_loss +
                           mu * entropy_loss +
                           gamma_smth * smooth_loss +
                           mu_att * attn_loss +
                           reg * stability_reg)
    return total_loss, { "ce": ce_loss.item(), "ent": entropy_loss.item(), "reg": reg_loss.item()}


def train_ecg_model(model, train_loader, val_loader, epochs=135, lr=1e-3, mixup_fn=None, resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    save_dir = "/content/drive/MyDrive/CIFAR10"
    os.makedirs(save_dir, exist_ok=True)

    slow_keys = ['alpha', 'theta', 'omiga']
    mid_keys = ['beta', 'tau', 'temperature', 'gamma']
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

    start_epoch = 111
    best_val_acc = 0.9172  

    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'best_val_acc' in checkpoint:
                best_val_acc = checkpoint['best_val_acc']
        else:
            model.load_state_dict(checkpoint)
            start_epoch = 111  

        for group in optimizer.param_groups:
            group['lr'] = 5e-5
            group.setdefault('initial_lr', 1e-3)  
    else:
        print("Weight file not found")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, last_epoch=start_epoch - 1)

    model.eval()
    init_correct, init_total = 0, 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            vx = add_coordinate_encoding(vx)
            v_out = model(vx)
            init_total += vy.size(0)
            init_correct += (v_out.argmax(1) == vy).sum().item()
    print(f"\n[Initial State] Val Acc: {100 * init_correct / init_total:.2f}%")
    print("-" * 85)

    # best_val_acc = 0.0
    early_stop_counter = 0
    early_stop_patience = 10

    log_file = open("evolution_logs.txt", "w")
    log_file.write("epoch,alpha,beta,theta,gamma,omiga\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_correct, train_total = 0, 0
        epoch_ce_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if mixup_fn is not None:
                batch_x, batch_y = mixup_fn(batch_x, batch_y)
            batch_x = add_coordinate_encoding(batch_x)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss, info = kan_optimal_loss(outputs, batch_y, model, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            epoch_ce_loss += info['ce']
            if batch_y.ndim > 1:
                train_correct += (outputs.argmax(1) == batch_y.argmax(1)).sum().item()
            else:
                train_correct += (outputs.argmax(1) == batch_y).sum().item()
            train_total += batch_y.size(0)
            pbar.set_postfix(ce=f"{info['ce']:.3f}", acc=f"{100 * train_correct / train_total:.2f}%")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                vx = add_coordinate_encoding(vx)
                v_out = model(vx)
                val_correct += (v_out.argmax(1) == vy).sum().item()
                val_total += vy.size(0)
        val_acc = 100 * val_correct / val_total
        train_acc = 100 * train_correct / train_total
        betas, gammas, alphas, thetas = [], [], [], []
        omiga_means = []
        for m in model.modules():
            if isinstance(m, SplineWeightLayer):
                alphas.append(torch.abs(m.alpha).mean().item())
                betas.append(torch.abs(m.beta).mean().item())
                thetas.append(torch.abs(m.theta).mean().item())
                gammas.append(torch.abs(m.gamma).mean().item())
            if isinstance(m, FastKANLayer):
                omiga_means.append(torch.abs(m.omiga).mean().item())
        avg_alpha, avg_beta, avg_theta, avg_gamma = np.mean(alphas), np.mean(betas), np.mean(thetas), np.mean(gammas)
        avg_omiga = np.mean(omiga_means)

        print(f"\n[Epoch {epoch + 1} Result] Val Acc: {val_acc:.2f}% | Train Acc: {train_acc:.2f}%")
        print(
            f"  Evolution: α(I)={avg_alpha:.3f} | β(W1)={avg_beta:.3f} | θ(W3)={avg_theta:.3f} | γ(Res)={avg_gamma:.3f}")
        print(f"  Residuals: Omiga Mean={avg_omiga:.4f} | Avg CE: {epoch_ce_loss / len(train_loader):.4f}")

        log_file.write(f"{epoch + 1},{avg_alpha},{avg_beta},{avg_theta},{avg_gamma},{avg_omiga}\n")
        log_file.flush()

        conn = model.get_active_conn_info()
        print("  Topology Audit (Connectivity Ratio):")
        for layer_name in sorted(conn.keys()):
            d = conn[layer_name]
            print(f"    {layer_name}: {d['ratio'] * 100:.1f}% active | tau_avg: {d['tau_mean']:.4f}")
          
        scheduler.step()

        last_path = os.path.join(save_dir, "last_checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
        }, last_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = "/content/drive/MyDrive/KAN_CIFAR10/best_model_8th.pth"
            torch.save(model.state_dict(), save_path)
            # torch.save(model.state_dict(), "best_kan_cifar10_v3.pth")
            print(f"  >>> 突破最高准确率: {best_val_acc:.2f}%, 模型已保存")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print(f"\n[Early Stopping], best Acc: {best_val_acc:.2f}%")
            break
        print("-" * 85)
    log_file.close()


def calculate_class_weights(dataloader, boost_dict={}):
    all_labels = []
    for _, y in dataloader: all_labels.extend(y.cpu().numpy())
    counts = Counter(all_labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = torch.tensor([total / (num_classes * counts[i]) for i in range(num_classes)]).sqrt()
    for idx, factor in boost_dict.items():
        if idx < num_classes:
            weights[idx] = weights[idx] * factor
    final_weights = weights / weights.mean()
    return final_weights


if __name__ == "__main__":
    seed_everything(42)
    train_loader, val_loader, mixup_fn = get_dataloader(
        data_dir=r'dataset/KAN_CIFAR10',
        batch_size=32)
    model = ECGKANModel()

    resume_checkpoint = "/content/drive/MyDrive/KAN_CIFAR10/parameter.pth"
    train_ecg_model(
        model,
        train_loader,
        val_loader,
        mixup_fn=None,
        resume_path=resume_checkpoint)
  
