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
                     lamb=2e-4,  # Spline weight regularization coefficient (L1 + L2)
                     mu=1e-2,  # Path entropy coefficient (guiding feature decoupling)
                     gamma_smth=4e-2,  # Spline smoothness coefficient (suppressing high-frequency noise)
                     mu_att=1e-3,  # Sparse coefficient of the attention matrix W1
                     reg=1e-5,  # SplineWeightLayer Scalar stability coefficient
                     label_smooth=0):  # Logical label smoothing coefficient
    device = outputs.device
    ce_loss = F.cross_entropy(
        outputs,
        targets,
        weight=getattr(model, 'class_weights', None),
        label_smoothing=label_smooth)
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
          
    warmup = min(1.0, epoch / 5.0)
    total_loss = ce_loss + warmup * (
            lamb * reg_loss +
            mu * entropy_loss +
            gamma_smth * smooth_loss +
            mu_att * attn_loss +
            reg * stability_reg)
    return total_loss, {
        "ce": ce_loss.item(),
        "smth": smooth_loss.item(),
        "ent": entropy_loss.item(),
        "att": attn_loss.item(),
        "total": total_loss.item()}


def train_ecg_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    save_dir = "/content/drive/MyDrive/heart_wave"
    os.makedirs(save_dir, exist_ok=True)
  
    slow_keys = ['alpha', 'omiga']
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

    # Automatically calculate category weights to deal with imbalanced datasets
    model.class_weights = calculate_class_weights(train_loader).to(device)
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
    early_stop_patience = 10  

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
        conn = model.get_active_conn_info()
        for layer_name in sorted(conn.keys()):
            d = conn[layer_name]
            print(f"    {layer_name}: {d['ratio'] * 100:.1f}% active | tau_avg: {d['tau_mean']:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, "heart_wave_2.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  Parameters have been updated. Save the best model. (Acc: {best_val_acc:.2f}%)")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f" The validation set did not show any improvement: ({early_stop_counter}/{early_stop_patience})")

        if early_stop_counter >= early_stop_patience:
            print(f"\n[Early Stopping] Failed to make progress for {early_stop_patience} rounds, the process is prematurely terminated")
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
    final_weights = weights / weights.mean()
    return final_weights


if __name__ == "__main__":
    seed_everything(42)
    train_loader = prepare_ecg_data('/content/drive/MyDrive/mitbih/mitbih_train.csv', batch_size=72)
    val_loader = prepare_ecg_data('/content/drive/MyDrive/mitbih/mitbih_test.csv', batch_size=72)
    model = ECGKANModel()

    train_ecg_model(model, train_loader, val_loader)
