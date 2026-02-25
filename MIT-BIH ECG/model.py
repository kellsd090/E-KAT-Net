import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import umap
import hdbscan
from thop import profile, clever_format
from sklearn.preprocessing import StandardScaler
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter, zoom
from matplotlib.collections import LineCollection
from database_mit import prepare_ecg_data


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


class SplineWeightLayer(nn.Module): # neurone cell body
    def __init__(self, in_features=187, in_hidden=64, out_hidden=64, kernel_size=15):
        super(SplineWeightLayer, self).__init__()
        self.in_features = in_features  
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        self.k = kernel_size
        self.W2 = nn.Parameter(torch.Tensor(in_hidden, out_hidden))
        self.b = nn.Parameter(torch.zeros(in_features, out_hidden))
        self.ln = nn.LayerNorm(in_hidden)
        self.alpha = nn.Parameter(torch.ones(1))  # I
        self.beta = nn.Parameter(torch.ones(1))  # W1 (Attention)
        self.theta = nn.Parameter(torch.ones(1))  # W3 (Convolution)
        self.gamma = nn.Parameter(torch.ones(1))  # residual
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


class FastKANLayer(nn.Module):   # edge
    def __init__(self, in_neurons, out_neurons, grid_size=64, spline_order=3, hidden_dim=64, neuron_out_dim=64,
                 kernel_size=15, seq_len=187):
        super(FastKANLayer, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.hidden_dim = hidden_dim
        self.num_coeffs = grid_size + spline_order
        self.spline_weights = nn.Parameter(torch.randn(in_neurons, out_neurons, self.num_coeffs) * 0.1)
        self.weight_layers = nn.ModuleList([
            SplineWeightLayer(in_features=seq_len, in_hidden=hidden_dim, out_hidden=neuron_out_dim, kernel_size=kernel_size)
            for _ in range(out_neurons)])
        self.register_buffer("grid", torch.linspace(-1, 1, self.num_coeffs))
        self.tau = nn.Parameter(torch.randn(in_neurons, out_neurons) * 0.1 + 0.1)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.omiga = nn.Parameter(torch.full((in_neurons, out_neurons), 0.2))
        self.last_W1 = None
        with torch.no_grad():
            lin_init = torch.linspace(-1.0, 1.0, self.num_coeffs).to(self.spline_weights.device)
            initial_weights = lin_init.view(1, 1, -1).repeat(self.in_neurons, self.out_neurons, 1)
            self.spline_weights.data = initial_weights + torch.randn_like(initial_weights) * 0.01

    def b_spline_cubic(self, x):   # Kolmogorov-Arnold b-spline
        h = 2.0 / (self.num_coeffs - 1)
        dist = torch.abs(x.unsqueeze(-1) - self.grid) / h
        res_outer = torch.relu(2.0 - dist) ** 3 / 6.0
        res_inner = torch.relu(1.0 - dist) ** 3 * (4.0 / 6.0)
        res = res_outer - res_inner
        return res

    def forward(self, x_in, proj_x_prev=None):
        b, in_n, seq_len, feat_dim = x_in.shape # x_in: [B, in_n, 200, 128]
        max_val = torch.max(torch.abs(x_in)) + 1e-8
        x_norm = (x_in / max_val) * 0.95
        x_norm = torch.clamp(x_norm, -0.99, 0.99)
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
        return torch.stack(next_outputs, dim=1), torch.stack(next_projections, dim=1)


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


class MultiScalePrepConv(nn.Module):   # Conv layer before entering main model
    def __init__(self):
        super(MultiScalePrepConv, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), # 1 layer 64 channels
            nn.BatchNorm1d(16),
            nn.SiLU())
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 40, kernel_size=15, padding=7), 
            nn.BatchNorm1d(40),
            nn.SiLU())
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=25, padding=12), 
            nn.BatchNorm1d(8),
            nn.SiLU())

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return out # self.se(out)


class ECGKANModel(nn.Module):   # main E-KAT model
    def __init__(self, grid_size=64, spline_order=3, seq_len=187):
        super(ECGKANModel, self).__init__()
        self.prep_conv = MultiScalePrepConv()
        self.layer1 = FastKANLayer(1, 4, grid_size, spline_order, hidden_dim=64, neuron_out_dim=64, kernel_size=25, seq_len=seq_len)
        self.layer2 = FastKANLayer(4, 8, grid_size, spline_order, hidden_dim=64, neuron_out_dim=128, kernel_size=15, seq_len=seq_len)
        self.layer3 = FastKANLayer(8, 8, grid_size, spline_order, hidden_dim=128, neuron_out_dim=32, kernel_size=9, seq_len=seq_len)
        self.layer4 = FastKANLayer(8, 5, grid_size, spline_order, hidden_dim=32, neuron_out_dim=1, kernel_size=3, seq_len=seq_len)

    def forward(self, x, return_latent=False):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.prep_conv(x)  # [B, 32, 187]
        x = x.transpose(1, 2).unsqueeze(1)
        x, proj1 = self.layer1(x, proj_x_prev=None)
        x, proj2 = self.layer2(x, proj_x_prev=proj1)
        x, proj3 = self.layer3(x, proj_x_prev=proj2)
        x, proj4 = self.layer4(x, proj_x_prev=proj3)  # x: [B, 5, 187, 1]
        if return_latent: return x.squeeze(-1)
        return torch.mean(x, dim=(-1, -2))

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
                        "ratio": active_count / (m.in_neurons * m.out_neurons + 1e-8)
                    }
        return info


# visulization functions
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
        'gamma (Res)': torch.abs(weight_layer.gamma).item()}
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


def visualize_w1_analysis(layer, layer_idx, neuron_idx):
    if layer.last_W1 is None: return None
    W1_data = layer.last_W1[0].detach().cpu().numpy()
    col_ratio = np.sum(W1_data, axis=0) / (np.sum(W1_data) + 1e-8)
    fig = plt.figure(figsize=(8, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[25, 1], hspace=0.05, wspace=0.02)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.bar(range(len(col_ratio)), col_ratio, color='skyblue', width=0.8)
    ax_bar.set_xlim(-0.5, len(col_ratio) - 0.5)
    ax_bar.set_title(f"L{layer_idx}-N{neuron_idx}: W1 Attention Analysis (Sparse Grid)", fontsize=14)
    ax_bar.grid(True, linestyle='--', alpha=0.5)
    ax_heat = fig.add_subplot(gs[1, 0])
    sns.heatmap(W1_data, ax=ax_heat, cmap='magma', cbar=False)
    ax_heat.set_xlabel("Time/Feature Index")
    ax_cbar = fig.add_subplot(gs[1, 1])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(W1_data.min(), W1_data.max()), cmap='magma'), cax=ax_cbar)
    plt.show()
    return col_ratio


def visualize_waveforms(raw_segment, processed_segment, sample_rate=125):
    t = np.linspace(0, len(raw_segment) / sample_rate, len(raw_segment))
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].plot(t, raw_segment, color='gray', lw=1.2)
    axes[0].set_title("ECG Raw Waveform (Normalized [-1, 1])")
    axes[1].plot(t, processed_segment[0, :], color='black', lw=1.2)
    axes[1].set_title("ECG Processed Feature (First MS-Conv Branch)")
    for a in axes: a.grid(True, alpha=0.3, linestyle=':')
    plt.show()


def visualize_spline_mapping_window(layer, layer_idx, neuron_idx, edge_idx):
    device = next(layer.parameters()).device
    fig, ax = plt.subplots(figsize=(8, 6))
    with torch.no_grad():
        x_range = torch.linspace(-1, 1, 200).to(device)
        current_omiga = torch.abs(layer.omiga[edge_idx[0], edge_idx[1]])
        basis = layer.b_spline_cubic(x_range.unsqueeze(0))
        coeffs = layer.spline_weights[edge_idx[0], edge_idx[1]]
        y_spline = torch.matmul(basis, coeffs).squeeze()
        y_linear = current_omiga * x_range
        y_final = y_spline + y_linear
        ax.plot(x_range.cpu(), y_final.cpu(), 'r', lw=2, label='Total Activation')
        ax.plot(x_range.cpu(), x_range.cpu(), 'k', alpha=0.1, label='Identity')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.show()


def visualize_params_and_active_edges(layer, layer_idx, neuron_idx):
    weight_layer = layer.weight_layers[neuron_idx]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    params = {'alpha': torch.abs(weight_layer.alpha).item(), 'beta': torch.abs(weight_layer.beta).item(),
              'theta': torch.abs(weight_layer.theta).item(), 'gamma': torch.abs(weight_layer.gamma).item()}
    ax1.bar(params.keys(), params.values(), color='royalblue', alpha=0.7)
    w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
    active_energy = w_strength.clone()
    active_energy[w_strength <= torch.abs(layer.tau)] = np.nan
    sns.heatmap(active_energy.detach().cpu().numpy(), ax=ax2, cmap='YlGn', annot=True, fmt=".2f", linewidths=1)
    plt.show()


def visualize_attention_waveforms_v2(raw_segment, processed_segment, col_ratio):
    from matplotlib.collections import LineCollection
    t = np.linspace(0, 1.0, len(raw_segment))
    weights_expanded = np.interp(np.linspace(0, 1, len(raw_segment)),
                                 np.linspace(0, 1, len(col_ratio)),
                                 col_ratio)
    norm = plt.Normalize(vmin=-weights_expanded.max() * 0.1, vmax=weights_expanded.max())
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3) 
    titles = ["Raw ECG Waveform: Attention Shading", "Processed Feature: Attention Shading"]
    data_list = [raw_segment, processed_segment]
    colors = ['gray', 'black']
    for i, ax in enumerate(axes):
        wave_data = data_list[i]
        points = np.array([t, wave_data]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='Blues', norm=norm, lw=2.5, zorder=3)
        lc.set_array(weights_expanded[:-1])
        ax.plot(t, wave_data, color=colors[i], alpha=0.1, lw=1, zorder=1)
        ax.add_collection(lc)
        ax.set_xlim(t.min(), t.max())
        ax.set_ylim(wave_data.min() * 1.5, wave_data.max() * 1.5)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(titles[i], fontsize=14)
        plt.colorbar(lc, ax=ax, label='W1 Weight')
    axes[1].set_xlabel("Time (Normalized)")
    plt.show()


def visualize_fusion_matrices(layer, layer_idx, neuron_idx):
    weight_layer = layer.weight_layers[neuron_idx]
    seq_len = weight_layer.in_features
    kernel = weight_layer.W3_conv.weight.detach().cpu()[0, 0, :].numpy()
    k_size = len(kernel)
    first_col = np.zeros(seq_len)
    first_row = np.zeros(seq_len)
    first_col[:k_size // 2 + 1] = kernel[k_size // 2:]
    first_row[:k_size // 2 + 1] = kernel[k_size // 2::-1]
    toe_mat = toeplitz(first_col, first_row)
    if layer.last_W1 is not None:
        W1 = layer.last_W1[0].cpu().numpy()
        alpha, beta, theta = [torch.abs(getattr(weight_layer, p)).item() for p in ['alpha', 'beta', 'theta']]
        W_combined = beta * W1 + theta * toe_mat + alpha * np.eye(seq_len)
        fig3d = plt.figure(figsize=(12, 10))
        ax3d = fig3d.add_subplot(111, projection='3d')
        W_smooth = gaussian_filter(zoom(W_combined, 2, order=3), sigma=0.5)
        X, Y = np.meshgrid(np.linspace(0, seq_len - 1, W_smooth.shape[0]),
                           np.linspace(0, seq_len - 1, W_smooth.shape[0]))
        surf = ax3d.plot_surface(X, Y, W_smooth, cmap=plt.cm.Blues, alpha=0.9, rcount=150, ccount=150)
        fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)
        ax3d.view_init(elev=30, azim=250)
        plt.show()
        plt.figure(figsize=(10, 8))
        sns.heatmap(W_combined, cmap=plt.cm.Blues)
        plt.title("2D Fusion Matrix (Magma)")
        plt.show()


def visualize_network_topology(model):
    model.eval()
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, FastKANLayer):
            layers.append(m)
    if not layers:
        print("未在模型中找到 FastKANLayer 层。")
        return
    layer_sizes = [layers[0].in_neurons] + [l.out_neurons for l in layers]
    num_layers = len(layer_sizes)
    fig, ax = plt.subplots(figsize=(16, 10))
    pos = {}
    x_space = 2.5 
    for l_idx, size in enumerate(layer_sizes):
        x = l_idx * x_space
        y_positions = np.linspace(-size / 2, size / 2, size)
        for n_idx, y in enumerate(y_positions):
            pos[(l_idx, n_idx)] = (x, y)
    all_lines = []
    colors = []

    with torch.no_grad():
        for l_idx, layer in enumerate(layers):
            w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
            tau = torch.abs(layer.tau)
            in_dim, out_dim = w_strength.shape
            for i in range(in_dim):
                for j in range(out_dim):
                    energy = w_strength[i, j].item()
                    threshold = tau[i, j].item()
                    if energy > threshold:
                        start_pos = pos[(l_idx, i)]
                        end_pos = pos[(l_idx + 1, j)]
                        all_lines.append([start_pos, end_pos])
                        colors.append(energy)
                        
    if all_lines:
        widths = 1.0 + 3.0 * (np.array(colors) / (max(colors) + 1e-8))
        lc = LineCollection(all_lines, cmap=plt.cm.Blues, array=np.array(colors),
                            linewidths=widths, alpha=0.8)
        line_obj = ax.add_collection(lc)
        cbar = fig.colorbar(line_obj, ax=ax)
        cbar.set_label(r'Edge Energy ($E = mean|W| + |\omega|$)')

    for (l_idx, n_idx), (x, y) in pos.items():
        ax.scatter(x, y, s=200, c='white', edgecolors='black', zorder=3)

    ax.set_title("ECG-KAN Global Topology: Active Paths Audit", fontsize=16)
    ax.set_xlabel("Processing Stage (Layers)", fontsize=12)
    plt.xticks(np.arange(num_layers) * x_space,
               ['Input'] + [f'Layer {i + 1}' for i in range(num_layers - 1)])
    ax.get_yaxis().set_visible(False)
    plt.grid(False)
    plt.show()


def analyze_ecg_representation(model, dataloader, device, n_samples=2000):
    model.eval()
    all_features = []
    all_labels = []
    collected = 0
    with torch.no_grad():
        for batch_data, batch_label in dataloader:
            latent = model(batch_data.to(device), return_latent=True)
            flat_feat = latent.view(latent.size(0), -1)
            all_features.append(flat_feat.cpu().numpy())
            all_labels.append(batch_label.numpy())
            collected += batch_data.size(0)
            if collected >= n_samples: break
    X = np.concatenate(all_features, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    X_scaled = StandardScaler().fit_transform(X)
    print("The UMAP and HDBSCAN are currently in progress...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embedding)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=y_true,
                           cmap='Spectral', s=10, alpha=0.6)
    ax1.set_title("Latent Space: Ground Truth Labels", fontsize=14)
    plt.colorbar(scatter1, ax=ax1, label='ECG Category (0-4)')
    scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels,
                           cmap='tab20', s=10, alpha=0.6)
    ax2.set_title("Latent Space: HDBSCAN Clusters", fontsize=14)
    plt.colorbar(scatter2, ax=ax2, label='Cluster ID (-1: Noise)')
    plt.suptitle("Output Layer Representation Analysis (N=5, L=187)", fontsize=16)
    plt.show()
    return embedding, cluster_labels


def print_layer_parameter_summary(model):
    header = f"{'Layer Name':<12} | {'beta (W1)':>10} | {'theta (W3)':>10} | {'alpha (I)':>10} | {'gamma (Res)':>10} | {'omiga (Edge)':>12} | {'Temp (T)':>10}"
    print(header)
    print("-" * len(header))
    found_any = False
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight_layers') and hasattr(layer, 'omiga') and hasattr(layer, 'temperature'):
            found_any = True
            temp_val = torch.abs(layer.temperature).item()
            omiga_mean = torch.mean(torch.abs(layer.omiga)).item()
            betas, thetas, alphas, gammas = [], [], [], []
            for swl in layer.weight_layers:
                betas.append(torch.abs(swl.beta).item())   # Attention
                thetas.append(torch.abs(swl.theta).item()) # Convolution
                alphas.append(torch.abs(swl.alpha).item()) # Identity
                gammas.append(torch.abs(swl.gamma).item()) # Residual
            avg_beta = np.mean(betas)
            avg_theta = np.mean(thetas)
            avg_alpha = np.mean(alphas)
            avg_gamma = np.mean(gammas)
            print(f"{name:<12} | {avg_beta:10.4f} | {avg_theta:10.4f} | {avg_alpha:10.4f} | {avg_gamma:10.4f} | {omiga_mean:12.4f} | {temp_val:10.4f}")
    if not found_any:
        print("!!! Warning: No FastKANLayer found. Check model structure.")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"==================================================")
    print(f"Model Summary:")
    print(f"   Total Parameters:     {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"==================================================")
    return total_params, trainable_params


def calculate_model_flops(model, device):
    model.eval()
    dummy_input = torch.randn(1, 187).to(device)
    try:
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        readable_flops, _ = clever_format([flops, 0], "%.3f")
        print(f"\n" + "=" * 50)
        print(f"   amount of FLOPs:  {readable_flops}")
        print(f"   GFLOPs 单位: {flops / 1e9:.4f} G")
        print(f"=" * 50 + "\n")
        return flops
    except Exception as e:
        print(f"error: {e}")
        return 0


if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGKANModel(grid_size=64, spline_order=3).to(device)
    test_csv_path = 'mitbih_test.csv'
    
    val_loader = prepare_ecg_data(test_csv_path, batch_size=1) 
    weight_path = "parameter.pth"
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        
    # tensors[0] represents the data, and tensors[1] represents the label
    sample_data = val_loader.dataset.tensors[0][target_sample_idx].unsqueeze(0)
    sample_label = val_loader.dataset.tensors[1][target_sample_idx]

    print(f"\n managing {target_sample_idx} th sample")
    print(f" label : {sample_label.item()}")
    
    target_sample_idx = 4  # Change this number to switch the sample.
    target_layer_idx, target_neuron_idx = 3, 0  # Which layer and which neuron
    edge = (3, 6)  # which edge
    target_layer = getattr(model, f'layer{target_layer_idx}')
    target_layer.visualize_idx = target_neuron_idx

    # calculate_model_flops(model, device)
    # count_parameters(model)

    model.eval()
    with torch.no_grad():
        prep_feat = model.prep_conv(sample_data.unsqueeze(1).to(device))
        _ = model(sample_data.to(device))  
    audit_loader = prepare_ecg_data(test_csv_path, batch_size=64)

    # analyze_ecg_representation(model, audit_loader, device, n_samples=3000)

    print("\n visualizing")
    # (1) W1 
    col_ratio = visualize_w1_analysis(target_layer, target_layer_idx, target_neuron_idx)
    # (2) Comparison of raw/processed waveforms
    visualize_waveforms(sample_data[0].numpy(), prep_feat[0].cpu().numpy())
    # (5) attention waveforms
    visualize_attention_waveforms_v2(sample_data[0].cpu().numpy(), prep_feat[0, 0].cpu().numpy(), col_ratio)
    # (3) activation function
    visualize_spline_mapping_window(target_layer, target_layer_idx, target_neuron_idx, edge)
    # (4) weights
    # visualize_params_and_active_edges(target_layer, target_layer_idx, target_neuron_idx)
    # (6 & 7) 3D and 2D W matrics
    visualize_fusion_matrices(target_layer, target_layer_idx, target_neuron_idx)
    # (8) topology
    # visualize_network_topology(model)
