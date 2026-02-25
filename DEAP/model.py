import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from datapreprocessing_deap_ECG import DEAP_ECG_RegressionDataset, DEAP_GSR_RegressionDataset


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SplineWeightLayer(nn.Module):
    def __init__(self, in_hidden, out_hidden, kernel_size, in_features=200):
        super(SplineWeightLayer, self).__init__()
        self.in_features = in_features  #  750
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        self.k = kernel_size
        self.W2 = nn.Parameter(torch.Tensor(in_hidden, out_hidden))
        self.b = nn.Parameter(torch.zeros(in_features, out_hidden))
        self.ln = nn.LayerNorm(in_hidden)
        self.alpha = nn.Parameter(torch.ones(1))  
        self.beta = nn.Parameter(torch.ones(1))  
        self.theta = nn.Parameter(torch.ones(1))  
        self.gamma = nn.Parameter(torch.ones(1))  
        self.W3_conv = nn.Conv1d(
            out_hidden, out_hidden, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=out_hidden, bias=False)
        self.register_buffer('identity', torch.eye(in_features).unsqueeze(0))
        if in_hidden != out_hidden:
            self.residual_proj = nn.Linear(in_hidden, out_hidden)
        else:
            self.residual_proj = nn.Identity()
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x, W1=None):
        b, seq_len, h_dim = x.shape
        x_norm = self.ln(x)
        x_prime = torch.matmul(x_norm, self.W2) + self.b
        x_temporal = x_prime.transpose(1, 2)
        w3_feat = self.W3_conv(x_temporal).transpose(1, 2)
        if W1 is not None:
            attn_kernel = torch.abs(self.beta) * W1 + torch.abs(self.alpha) * self.identity
            global_part = torch.bmm(attn_kernel, x_prime)
            spatial_combined = global_part + torch.abs(self.theta) * w3_feat
        else:
            spatial_combined = x_prime + torch.abs(self.theta) * w3_feat
        res_x = self.residual_proj(x)
        return spatial_combined + self.gamma * res_x, x_prime


class FastKANLayer(nn.Module):
    def __init__(self, in_neurons, out_neurons, neuron_out_dim, grid_size=96, spline_order=3, hidden_dim=64,
                 kernel_size=15, seq_len=200):
        super(FastKANLayer, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.hidden_dim = hidden_dim
        self.num_coeffs = grid_size + spline_order
        self.spline_weights = nn.Parameter(torch.randn(in_neurons, out_neurons, self.num_coeffs) * 0.1)
        if isinstance(neuron_out_dim, int):
            self.out_dims = [neuron_out_dim] * out_neurons
        else:
            self.out_dims = neuron_out_dim
        self.weight_layers = nn.ModuleList([
            SplineWeightLayer(in_features=seq_len, in_hidden=hidden_dim, out_hidden=self.out_dims[j], kernel_size=kernel_size)
            for j in range(out_neurons)])
        self.in_norm = nn.InstanceNorm1d(hidden_dim)
        self.register_buffer("grid", torch.linspace(-1, 1, self.num_coeffs))
        self.tau = nn.Parameter(torch.randn(in_neurons, out_neurons) * 0.1 + 0.1)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.omiga = nn.Parameter(torch.full((in_neurons, out_neurons), 0.2))
        self.last_W1 = None
        with torch.no_grad():
            lin_init = torch.linspace(-1.0, 1.0, self.num_coeffs).to(self.spline_weights.device)
            initial_weights = lin_init.view(1, 1, -1).repeat(self.in_neurons, self.out_neurons, 1)
            self.spline_weights.data = initial_weights + torch.randn_like(initial_weights) * 0.01

    def b_spline_cubic(self, x):
        h = 2.0 / (self.num_coeffs - 1)
        dist = torch.abs(x.unsqueeze(-1) - self.grid) / h
        res_outer = torch.relu(2.0 - dist) ** 3 / 6.0
        res_inner = torch.relu(1.0 - dist) ** 3 * (4.0 / 6.0)
        res = res_outer - res_inner
        return res

    def forward(self, x_in, proj_x_prev=None):
        b, in_n, seq_len, feat_dim = x_in.shape # x_in: [B, in_n, 200, 128]
        x_reshaped = x_in.reshape(b * in_n, feat_dim, seq_len)
        x_normed = self.in_norm(x_reshaped)
        x_norm = x_normed.reshape(b, in_n, seq_len, feat_dim)
        x_norm = torch.clamp(x_norm * 0.5, -0.99, 0.99)
        basis = self.b_spline_cubic(x_norm)
        spline_mapping = torch.einsum('binfc,ioc->bio nf', basis, self.spline_weights)
        omega_expanded = torch.abs(self.omiga).view(1, in_n, self.out_neurons, 1, 1)
        activated_signals = spline_mapping + omega_expanded * x_in.unsqueeze(2)
        next_outputs, next_projections = [], []
        t_val = torch.abs(self.temperature) * np.sqrt(seq_len / 256.0) + 1e-4

        for j in range(self.out_neurons):
            current_edges = activated_signals[:, :, j, :, :]
            edge_energies = torch.mean(current_edges ** 2, dim=(-1, -2))
            tau_j = torch.abs(self.tau[:, j]).unsqueeze(0)
            mask = torch.sigmoid((torch.sqrt(edge_energies + 1e-8) - tau_j) / t_val).unsqueeze(-1).unsqueeze(-1)
            W1_j = None
            
            if proj_x_prev is not None:
                multiplier = (torch.sqrt(edge_energies + 1e-8) / (tau_j + 1e-8)).unsqueeze(-1).unsqueeze(-1)
                weighted_prev = proj_x_prev * multiplier * mask
                mid = self.in_neurons // 2
                K = torch.cat([weighted_prev[:, 2*i, :, :] for i in range(mid)], dim=-1)
                Q = torch.cat([weighted_prev[:, 2*i+1, :, :] for i in range(mid)], dim=-1)
                K, Q = F.layer_norm(K, [K.size(-1)]), F.layer_norm(Q, [Q.size(-1)])
                raw_attn = torch.bmm(K, Q.transpose(-1, -2))
                W1_j = F.softmax(raw_attn / (np.sqrt(K.shape[-1]) * t_val), dim=-1)

            if hasattr(self, 'visualize_idx') and j == self.visualize_idx:
                self.last_W1 = W1_j.detach() if W1_j is not None else None

            combined_input = torch.sum(current_edges * mask, dim=1)
            out_j, proj_j = self.weight_layers[j](combined_input, W1=W1_j)
            next_outputs.append(out_j)
            next_projections.append(proj_j)
            
        if all(o.shape == next_outputs[0].shape for o in next_outputs):
            return torch.stack(next_outputs, dim=1), torch.stack(next_projections, dim=1)
        else:
            return next_outputs, next_projections


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiScalePrepConv(nn.Module):
    def __init__(self, kernel_size):   # 7,17,31; 15,31,63;
        super(MultiScalePrepConv, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=kernel_size[0], padding=kernel_size[0]//2), # 抓高频细节
            nn.BatchNorm1d(8),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=kernel_size[1], padding=kernel_size[1]//2), # 抓标准形态
            nn.BatchNorm1d(20),
            nn.SiLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=kernel_size[2], padding=kernel_size[2]//2), # 抓长程节律(R-R间期)
            nn.BatchNorm1d(4),
            nn.SiLU()
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return out # self.se(out)


class ECGKANModel(nn.Module):
    def __init__(self, kernel_size, ks=[7, 17, 31], grid_size=64, spline_order=3, seq_len=200): # 31,25,17,9,3; 63,47,31,15,7;
        super(ECGKANModel, self).__init__()
        self.prep_conv = MultiScalePrepConv(kernel_size=ks)
        self.layer1 = FastKANLayer(in_neurons=1, out_neurons=4, neuron_out_dim=64,
                                   grid_size=grid_size, spline_order=spline_order,
                                   hidden_dim=32, kernel_size=kernel_size[0], seq_len=seq_len)

        self.layer2 = FastKANLayer(in_neurons=4, out_neurons=8, neuron_out_dim=96,
                                   grid_size=grid_size, spline_order=spline_order,
                                   hidden_dim=64, kernel_size=kernel_size[1], seq_len=seq_len)

        self.layer3 = FastKANLayer(in_neurons=8, out_neurons=10, neuron_out_dim=64,
                                   grid_size=grid_size, spline_order=spline_order,
                                   hidden_dim=96, kernel_size=kernel_size[2], seq_len=seq_len)

        self.layer4 = FastKANLayer(in_neurons=10, out_neurons=8, neuron_out_dim=32,
                                   grid_size=grid_size, spline_order=spline_order,
                                   hidden_dim=64, kernel_size=kernel_size[3], seq_len=seq_len)
        
        self.layer5 = FastKANLayer(in_neurons=8, out_neurons=5, neuron_out_dim=[1, 1, 1, 1, 32],
                                   grid_size=grid_size, spline_order=spline_order,
                                   hidden_dim=32, kernel_size=kernel_size[4], seq_len=seq_len)

    def forward(self, x, mode='independent', target_seq_len=None):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.prep_conv(x)  # [B, 32, 187]
        x = x.transpose(1, 2).unsqueeze(1)
        
        x, proj1 = self.layer1(x, proj_x_prev=None)
        x, proj2 = self.layer2(x, proj_x_prev=proj1)
        x, proj3 = self.layer3(x, proj_x_prev=proj2)
        x, proj4 = self.layer4(x, proj_x_prev=proj3)  # x: [B, 5, 187, 1]
        x_list, proj5 = self.layer5(x, proj_x_prev=proj4)
        
        if mode == 'independent':
            x_emotion = torch.stack(x_list[:4], dim=1)
            return torch.mean(x_emotion, dim=(-1, -2))
        elif mode == 'cascade':
            bridge_out = x_list[4].unsqueeze(1)
            if target_seq_len is not None:
                bridge_out = align_sequence_length(bridge_out, target_seq_len)
            return bridge_out

    def get_active_conn_info(self):
        info = {}
        for name, m in self.named_modules():
            if isinstance(m, FastKANLayer):
                with torch.no_grad():
                    w_strength = torch.mean(torch.abs(m.spline_weights), dim=-1) + torch.abs(m.omiga)
                    active_count = (w_strength > torch.abs(m.tau)).sum().item()
                    info[name] = {
                        "active": active_count,
                        "total": m.in_neurons * m.out_neurons,
                        "tau_mean": torch.abs(m.tau).mean().item(),
                        "ratio": active_count / (m.in_neurons * m.out_neurons + 1e-8)}
        return info

class AatyFusionModel(nn.Module):
    def __init__(self, expert_ecg, expert_gsr, fusion_hidden_dim=32, grid_size=32, seq_len=512):
        super(AatyFusionModel, self).__init__()
        self.expert_ecg = expert_ecg  # Pre-trained ECG experts
        self.expert_gsr = expert_gsr  # Pre-trained GSR experts
        self.experts = nn.ModuleList([expert_ecg, expert_gsr])

        self.fusion_layer1 = FastKANLayer(in_neurons=2, out_neurons=4,
                                          neuron_out_dim=64, grid_size=grid_size,
                                          hidden_dim=fusion_hidden_dim, seq_len=seq_len)
        self.fusion_layer2 = FastKANLayer(in_neurons=4, out_neurons=8,
                                          neuron_out_dim=96, grid_size=grid_size,
                                          hidden_dim=64, seq_len=seq_len)
        self.fusion_layer3 = FastKANLayer(in_neurons=8, out_neurons=10,
                                          neuron_out_dim=64, grid_size=grid_size,
                                          hidden_dim=96, seq_len=seq_len)
        self.fusion_layer4 = FastKANLayer(in_neurons=10, out_neurons=8,
                                          neuron_out_dim=32, grid_size=grid_size,
                                          hidden_dim=64, seq_len=seq_len)
        self.fusion_layer5 = FastKANLayer(in_neurons=8, out_neurons=4,
                                          neuron_out_dim=1, grid_size=grid_size,
                                          hidden_dim=32, seq_len=seq_len)

    def forward(self, ecg_raw, gsr_raw):
        feat_ecg = self.expert_ecg(ecg_raw, mode='cascade', target_seq_len=512)
        feat_gsr = self.expert_gsr(gsr_raw, mode='cascade', target_seq_len=None)

        x_fusion = torch.cat([feat_ecg, feat_gsr], dim=1)

        x, proj1 = self.fusion_layer1(x_fusion, proj_x_prev=x_fusion)
        x, proj2 = self.fusion_layer2(x, proj_x_prev=proj1)
        x, proj3 = self.fusion_layer3(x, proj_x_prev=proj2)
        x, proj4 = self.fusion_layer4(x, proj_x_prev=proj3)
        x_final, _ = self.fusion_layer5(x, proj_x_prev=proj4)
        return torch.mean(x_final, dim=(-1, -2))

    def get_active_conn_info(self):
        info = {}
        for name, m in self.named_modules():
            if isinstance(m, FastKANLayer):
                with torch.no_grad():
                    w_strength = torch.mean(torch.abs(m.spline_weights), dim=-1) + torch.abs(m.omiga)
                    active_count = (w_strength > torch.abs(m.tau)).sum().item()
                    info[name] = {
                        "active": active_count,
                        "total": m.in_neurons * m.out_neurons,
                        "tau_mean": torch.abs(m.tau).mean().item(),
                        "ratio": active_count / (m.in_neurons * m.out_neurons + 1e-8)}
        return info


def align_sequence_length(tensor, target_len):
    b, n, s, h = tensor.shape
    if s == target_len:
        return tensor
    x = tensor.view(b * n, s, h).transpose(1, 2)
    x = F.interpolate(x, size=target_len, mode='linear', align_corners=True)
    return x.transpose(1, 2).view(b, n, target_len, h)


def setup_fusion_freeze_strategy(fusion_model):
    for expert in fusion_model.experts:
        expert.requires_grad_(False)
        expert.layer4.requires_grad_(True)
        expert.layer5.weight_layers[4].requires_grad_(True)
        expert.layer5.spline_weights.requires_grad = True
        expert.layer5.omiga.requires_grad = True
        expert.layer5.tau.requires_grad = True

    fusion_model.fusion_layer1.requires_grad_(True)
    fusion_model.fusion_layer2.requires_grad_(True)
    fusion_model.fusion_layer3.requires_grad_(True)
    fusion_model.fusion_layer4.requires_grad_(True)
    fusion_model.fusion_layer5.requires_grad_(True)


# visualization
def visualize_model_internals(model, layer_idx=1, neuron_idx=0, edge_idx=(0, 0)):
    model.eval()
    layer = getattr(model, f'layer{layer_idx}')
    if not hasattr(layer, 'b_spl_cubic'):
        layer.b_spl_cubic = layer.b_spline_cubic
    weight_layer = layer.weight_layers[neuron_idx]
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    sns.heatmap(weight_layer.W2.detach().cpu().numpy(), ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title(f"L{layer_idx}-N{neuron_idx}: W2 Matrix (Feature Projection)")

    if layer.last_W1 is not None:
        sns.heatmap(layer.last_W1[0].detach().cpu().numpy(), ax=axes[0, 1], cmap='magma')
        axes[0, 1].set_title(f"L{layer_idx}-N{neuron_idx}: W1 Attention (Global Space)")
    else:
        axes[0, 1].text(0.5, 0.5, "W1 is None", ha='center')

    with torch.no_grad():
        conv_weight = weight_layer.W3_conv.weight.detach().cpu()
        d_out, _, k = conv_weight.shape
        sns.heatmap(conv_weight.squeeze(1).numpy(), ax=axes[1, 0], cmap='coolwarm', center=0)
        axes[1, 0].set_title(f"L{layer_idx}-N{neuron_idx}: W3 1D Convolutional Kernels")
        axes[1, 0].set_xlabel("Time Window (k)")
        axes[1, 0].set_ylabel("Hidden Channels")

    params = {
        'alpha (I)': torch.abs(weight_layer.alpha).item(),
        'beta (W1)': torch.abs(weight_layer.beta).item(),
        'theta (W3)': torch.abs(weight_layer.theta).item(),
        'gamma (Res)': torch.abs(weight_layer.gamma).item()
    }
    axes[1, 1].bar(params.keys(), params.values(), color=['gray', 'purple', 'blue', 'green'])
    axes[1, 1].set_title(f"L{layer_idx}-N{neuron_idx}: Component Weights")

    with torch.no_grad():
        x_range = torch.linspace(-1, 1, 200).to(layer.spline_weights.device)
        current_omiga = torch.abs(layer.omiga[edge_idx[0], edge_idx[1]])
        basis = layer.b_spline_cubic(x_range.unsqueeze(0))
        coeffs = layer.spline_weights[edge_idx[0], edge_idx[1]]
        y_spline = torch.matmul(basis, coeffs).squeeze()
        y_linear = current_omiga * x_range
        y_final = y_spline + y_linear
        axes[2, 0].plot(x_range.cpu(), y_spline.cpu(), '--', label='Spline Mapping $f(x)$', alpha=0.7)
        axes[2, 0].plot(x_range.cpu(), y_linear.cpu(), ':', label=r'Linear Shortcut $\omega \cdot x$', alpha=0.7)
        axes[2, 0].plot(x_range.cpu(), y_final.cpu(), 'r', lw=2.5, label='Total Activation (Learned)')
        axes[2, 0].plot(x_range.cpu(), x_range.cpu(), 'k', alpha=0.2, lw=1, label='Identity (y=x)')
        axes[2, 0].set_title(f"Edge ({edge_idx[0]}->{edge_idx[1]}): Value Mapping Function")
        axes[2, 0].set_xlabel("Input Normalized Voltage")
        axes[2, 0].set_ylabel("Output Mapped Voltage")
        axes[2, 0].legend(loc='upper left', fontsize='small')
        axes[2, 0].grid(True, alpha=0.3)

    with torch.no_grad():
        w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
        active_mask = (w_strength > torch.abs(layer.tau)).float()
        sns.heatmap(active_mask.cpu().numpy(), ax=axes[2, 1], cmap='Greens', cbar=False)
    plt.show()


if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGKANModel(grid_size=64, spline_order=3)
    test_csv_path = 'E:/database/archive/mitbih_test.csv'
    val_loader = DEAP_ECG_RegressionDataset(test_csv_path)

    state_dict = torch.load("D:/treasure/true/thesis/pth/heart_wave/v7/heart_wave_1_9851.pth", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    if "model" in locals():
        sample_data, sample_label = next(iter(val_loader))
        sample_data = sample_data.to(device)

        target_layer_idx = 3  # which layer
        target_neuron_idx = 2  # which neuron
        edge = (1, 1)  # which edge

        target_layer = getattr(model, f'layer{target_layer_idx}')
        target_layer.visualize_idx = target_neuron_idx

        model.eval()
        with torch.no_grad():
            _ = model(sample_data)

        visualize_model_internals(model,
                                  layer_idx=target_layer_idx,
                                  neuron_idx=target_neuron_idx,
                                  edge_idx=edge)

    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"total parameters: {total_params}")

