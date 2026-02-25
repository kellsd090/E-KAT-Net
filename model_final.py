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
from sklearn.preprocessing import StandardScaler
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter, zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataset_cifar10 import get_dataloader, add_coordinate_encoding


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
    def __init__(self, in_features=256, in_hidden=64, out_hidden=64, kernel_size=3):
        super(SplineWeightLayer, self).__init__()
        self.in_features = in_features  # 256 (16x16)
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        self.k = kernel_size

        # 特征投影矩阵 W2
        self.W2 = nn.Parameter(torch.Tensor(in_hidden, out_hidden))
        # 偏置 b 改为初始为 0
        self.b = nn.Parameter(torch.zeros(in_features, out_hidden))
        self.ln = nn.LayerNorm(in_hidden)

        # 空间权重控制参数
        self.alpha = nn.Parameter(torch.tensor([0.5]))  # I 的幅度
        self.beta = nn.Parameter(torch.ones(1))  # W1 的幅度
        self.gamma = nn.Parameter(torch.tensor([0.5]))  # W3 的幅度 (初始较小)
        self.theta = nn.Parameter(torch.ones(1))  # 残差幅度

        # --- [核心修改] W3 局部卷积先验 ---
        # groups=out_hidden 实现了你要求的“全部维度同时进行独立二维卷积”
        # padding=k//2 自动处理了你说的“16x16 -> 18x18 补0”逻辑
        self.W3_conv = nn.Conv2d(
            out_hidden, out_hidden, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=out_hidden, bias=False
        )

        if in_hidden != out_hidden:
            self.residual_proj = nn.Linear(in_hidden, out_hidden)
        else:
            self.residual_proj = nn.Identity()

        nn.init.xavier_uniform_(self.W2)

    def forward(self, x, W1=None):
        # x: [B, 256, Hidden]
        b, seq_len, h_dim = x.shape
        x_norm = self.ln(x)

        # 1. 计算 x' = LN(x)*W2 + b
        # 这里对应你设想的 (LN(x)*W2 + b)
        x_prime = torch.matmul(x_norm, self.W2) + self.b  # [B, 256, out_hidden]

        # 2. 计算 W3(x')：局部空间先验
        # 恢复 16x16 空间进行卷积
        x_spatial = x_prime.transpose(1, 2).view(b, self.out_hidden, 16, 16)
        w3_feat = self.W3_conv(x_spatial)  # 卷积自动处理了 padding 和滑窗
        w3_feat = w3_feat.view(b, self.out_hidden, seq_len).transpose(1, 2)

        # 3. 空间组合：(β*W1 + α*I) * x' + θ*W3(x')
        identity = torch.eye(seq_len, device=x.device).unsqueeze(0)

        # 注意：W3 在你的公式里是作为独立加项还是在算子内部？
        # 按照你的公式：((β*W1 + α*I + θ*W3) * x')
        # 我们这里通过元素加法实现等效逻辑
        if W1 is not None:
            attn_kernel = torch.abs(self.beta) * W1 + torch.abs(self.alpha) * identity
            global_part = torch.bmm(attn_kernel, x_prime)
            spatial_combined = global_part + torch.abs(self.theta) * w3_feat
        else:
            spatial_combined = x_prime + torch.abs(self.theta) * w3_feat

        # 4. 加上残差 γ*res(x)
        res_x = self.residual_proj(x)
        return spatial_combined + self.gamma * res_x, x_prime


class FastKANLayer(nn.Module):
    def __init__(self, in_neurons, out_neurons, grid_size=64, spline_order=3, hidden_dim=64, neuron_out_dim=64, kernel_size=3):
        super(FastKANLayer, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.hidden_dim = hidden_dim
        self.num_coeffs = grid_size + spline_order

        self.spline_weights = nn.Parameter(torch.randn(in_neurons, out_neurons, self.num_coeffs) * 0.1)
        self.base_scales = nn.Parameter(torch.empty(in_neurons, out_neurons))
        nn.init.normal_(self.base_scales, mean=0.6, std=0.1)

        # 每一个输出神经元对应一个增强版的 SplineWeightLayer
        self.weight_layers = nn.ModuleList([
            SplineWeightLayer(in_hidden=hidden_dim, out_hidden=neuron_out_dim, kernel_size=kernel_size) for _ in range(out_neurons)
        ])

        self.register_buffer("grid", torch.linspace(-1, 1, self.num_coeffs))
        self.tau = nn.Parameter(torch.randn(in_neurons, out_neurons) * 0.1 + 0.1)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.last_W1 = None
        self.omiga = nn.Parameter(torch.full((in_neurons, out_neurons), 0.2))

    # b_spline_cubic 等方法保持不变...
    def b_spline_cubic(self, x):
        h = 2.0 / (self.num_coeffs - 1)
        dist = torch.abs(x.unsqueeze(-1) - self.grid) / h
        res = torch.zeros_like(dist)
        mask1 = dist < 1.0
        res[mask1] = (2 / 3) - (dist[mask1] ** 2) + (dist[mask1] ** 3 / 2)
        mask2 = (dist >= 1.0) & (dist < 2.0)
        res[mask2] = ((2 - dist[mask2]) ** 3) / 6
        return res

    def forward(self, x_in, proj_x_prev=None):
        b, in_n, seq_len, feat_dim = x_in.shape
        # 计算样条输入
        x_energy_in = torch.mean(x_in, dim=-1)
        x_norm = torch.clamp(x_energy_in / (torch.abs(x_energy_in).max(dim=-1, keepdim=True)[0] + 1e-8), -0.99, 0.99)

        base_feat = F.silu(x_norm).unsqueeze(2) * self.base_scales.unsqueeze(-1).unsqueeze(0)
        basis = self.b_spline_cubic(x_norm)
        spline_feat = torch.einsum('bifc,ijc->bijf', basis, self.spline_weights)

        edge_response = (base_feat + spline_feat).unsqueeze(-1)
        omiga_expanded = torch.abs(self.omiga).view(1, self.in_neurons, self.out_neurons, 1, 1)
        activated_signals = (edge_response + omiga_expanded) * x_in.unsqueeze(2)

        next_outputs, next_projections = [], []
        t_val = torch.abs(self.temperature) + 1e-4

        for j in range(self.out_neurons):
            current_edges = activated_signals[:, :, j, :, :]
            edge_energies = torch.mean(torch.abs(current_edges), dim=(-1, -2))
            tau_j = torch.abs(self.tau[:, j]).unsqueeze(0)
            mask = torch.sigmoid((edge_energies - tau_j) / t_val).unsqueeze(-1).unsqueeze(-1)

            W1_j = None
            if proj_x_prev is not None:
                multiplier = (edge_energies / (tau_j + 1e-8)).unsqueeze(-1).unsqueeze(-1)
                weighted_prev = proj_x_prev * multiplier * mask
                mid = self.in_neurons // 2
                K = torch.cat([weighted_prev[:, i, :, :] for i in range(mid)], dim=-1)
                Q = torch.cat([weighted_prev[:, i, :, :] for i in range(mid, mid * 2)], dim=-1)
                K, Q = F.layer_norm(K, [K.size(-1)]), F.layer_norm(Q, [Q.size(-1)])
                raw_attn = torch.bmm(K, Q.transpose(-1, -2))
                W1_j = F.softmax(raw_attn / (np.sqrt(K.shape[-1]) * t_val), dim=-1)

            if hasattr(self, 'visualize_idx') and j == self.visualize_idx:
                self.last_W1 = W1_j.detach() if W1_j is not None else None

            combined_input = torch.mean(current_edges * mask, dim=1)
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
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ImagePrepConv(nn.Module):
    def __init__(self, in_channels=5, out_channels=32):
        super(ImagePrepConv, self).__init__()

        # --- 1. 维度投影分支 (Shortcut) ---
        # 用于将 5 通道输入直接映射到 out_channels 维度，以便执行加法
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        # --- 2. 残差学习分支 (Residual Branches) ---
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels // 2),
            nn.SiLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU()
        )

        # --- 3. 融合后的进一步提取层 ---
        self.post_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # 激活函数
        self.relu = nn.SiLU()
        # 降采样：将 32x32 缩小为 16x16
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        #self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: [B, 5, 32, 32]
        # 计算残差路径
        identity = self.shortcut(x)  # [B, 64, 32, 32]
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # 拼接多尺度特征
        out = torch.cat([x1, x2, x3], dim=1)  # [B, 64, 32, 32]
        # 进一步卷积增强
        out = self.post_conv(out)
        # --- 核心：执行残差加法 ---
        out += identity
        out = self.relu(out)
        # 降采样并转为 KAN 序列格式
        #out = self.pool(out)  # [B, 64, 16, 16]
        out = self.pixel_unshuffle(out)
        b, c, h, w = out.shape
        out = out.flatten(2)  # [B, 64, 256]
        out = out.transpose(1, 2)  # [B, 256, 64]
        out = out.unsqueeze(1)  # [B, 1, 256, 64]
        return out


class ECGKANModel(nn.Module):
    def __init__(self, grid_size=64, spline_order=3):
        super(ECGKANModel, self).__init__()
        # 1. 初始多尺度卷积，将 5x32x32 映射为序列 [B, 1, 256, 64]
        # 注意这里我显式设置了 out_channels=64
        self.prep_conv = ImagePrepConv(in_channels=5, out_channels=32)

        self.layer1 = FastKANLayer(1, 4, grid_size, spline_order, hidden_dim=128, neuron_out_dim=64, kernel_size=7)
        self.layer2 = FastKANLayer(4, 8, grid_size, spline_order, hidden_dim=64, neuron_out_dim=64, kernel_size=5)
        self.layer3 = FastKANLayer(8, 16, grid_size, spline_order, hidden_dim=64, neuron_out_dim=128, kernel_size=5)
        self.layer4 = FastKANLayer(16, 16, grid_size, spline_order, hidden_dim=128, neuron_out_dim=128, kernel_size=3)
        self.layer5 = FastKANLayer(16, 20, grid_size, spline_order, hidden_dim=128, neuron_out_dim=128, kernel_size=3)
        self.layer6 = FastKANLayer(20, 16, grid_size, spline_order, hidden_dim=128, neuron_out_dim=16, kernel_size=3)
        self.layer7 = FastKANLayer(16, 10, grid_size, spline_order, hidden_dim=16, neuron_out_dim=1, kernel_size=1)

    def forward(self, x, return_latent=False, audit_layer=None):
        # x: [B, 5, 32, 32]
        x = self.prep_conv(x)  # [B, 1, 256, 64]

        # 逐层演化并检查是否需要审计该层输入
        layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]
        projections = [None]

        for i, layer in enumerate(layers):
            layer_idx = i + 1
            # 如果这是我们要审计的层，记录其输入 x
            if audit_layer == layer_idx:
                return x

            x, proj = layer(x, proj_x_prev=projections[-1])
            projections.append(proj)

        if return_latent:
            return x.squeeze(-1)

        return torch.mean(x, dim=(-1, -2))

    def get_active_conn_info(self):
        # 保持原有审计逻辑不变
        info = {}
        for name, m in self.named_modules():
            if isinstance(m, FastKANLayer):
                with torch.no_grad():
                    w_strength = torch.mean(torch.abs(m.spline_weights), dim=-1) + torch.abs(m.base_scales)
                    active_count = (w_strength > torch.abs(m.tau)).sum().item()
                    info[name] = {"active": active_count, "total": m.in_neurons * m.out_neurons,
                                  "tau_mean": torch.abs(m.tau).mean().item(), 'ratio': active_count / (m.in_neurons * m.out_neurons)}
        return info


def visualize_model_internals(model, layer_idx=1, neuron_idx=0, edge_idx=(0, 0)):
    model.eval()
    layer = getattr(model, f'layer{layer_idx}')
    weight_layer = layer.weight_layers[neuron_idx]

    # 扩展画布为 3行2列，以容纳 W3
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # --- (1) W2 矩阵热力图 ---
    sns.heatmap(weight_layer.W2.detach().cpu().numpy(), ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title(f"L{layer_idx}-N{neuron_idx}: W2 Matrix (Feature Projection)")

    # --- (2) W1 矩阵热力图 (动态生成) ---
    if layer.last_W1 is not None:
        sns.heatmap(layer.last_W1[0].detach().cpu().numpy(), ax=axes[0, 1], cmap='magma')
        axes[0, 1].set_title(f"L{layer_idx}-N{neuron_idx}: W1 Attention (Global Space)")
    else:
        axes[0, 1].text(0.5, 0.5, "W1 is None", ha='center')

    # --- (3) [新增] W3 矩阵可视化 (局部卷积先验) ---
    # 逻辑：将 k*k 卷积核展开为 1D 稀疏向量 [k^2 + 2*(16-k)个0]
    with torch.no_grad():
        # 提取 Depthwise 卷积核: [Hidden, 1, k, k]
        conv_weight = weight_layer.W3_conv.weight.detach().cpu()
        d_out, _, k, _ = conv_weight.shape

        # 构造稀疏展示矩阵: d * [k + (16-k) + k + (16-k) + k]
        # 总长度 = 3k + 2(16-k) = k + 32
        sparse_w3 = []
        for d in range(d_out):
            kernel_2d = conv_weight[d, 0]  # [k, k]
            row_list = []
            for r in range(k):
                row_list.append(kernel_2d[r])  # 加入 k 个权重
                if r < k - 1:
                    row_list.append(torch.zeros(16 - k))  # 插入 16-k 个 0
            sparse_w3.append(torch.cat(row_list))

        w3_display = torch.stack(sparse_w3).numpy()
        sns.heatmap(w3_display, ax=axes[1, 0], cmap='coolwarm', center=0)
        axes[1, 0].set_title(f"L{layer_idx}-N{neuron_idx}: W3 Sparse Matrix (Local Prior)")
        axes[1, 0].set_xlabel("Sparse Kernel Index (3k + 2*(16-k))")
        axes[1, 0].set_ylabel("Channel Dimension (d)")

    # --- (4) θ, α, β, γ 参数柱状图 (权重分配比例) ---
    params = {
        'alpha (I)': torch.abs(weight_layer.alpha).item(),
        'beta (W1)': torch.abs(weight_layer.beta).item(),
        'theta (W3)': torch.abs(weight_layer.theta).item(),
        'gamma (Res)': torch.abs(weight_layer.gamma).item()
    }
    axes[1, 1].bar(params.keys(), params.values(), color=['gray', 'purple', 'blue', 'green'])
    axes[1, 1].set_title(f"L{layer_idx}-N{neuron_idx}: Component Weights")
    axes[1, 1].set_ylim(0, max(params.values()) * 1.2)

    # --- (5) SiLU + B样条激活形状 ---
    with torch.no_grad():
        x_range = torch.linspace(-1, 1, 200).to(layer.spline_weights.device)
        base_scale = layer.base_scales[edge_idx[0], edge_idx[1]]
        y_base = base_scale * F.silu(x_range)

        # 兼容方法名：支持 b_spl_cubic 或 b_spline_cubic
        spl_func = layer.b_spline_cubic if hasattr(layer, 'b_spline_cubic') else layer.b_spl_cubic
        basis = spl_func(x_range.unsqueeze(0))
        coeffs = layer.spline_weights[edge_idx[0], edge_idx[1]]
        y_spline = torch.matmul(basis, coeffs).squeeze()

        axes[2, 0].plot(x_range.cpu(), y_base.cpu(), '--', label='Base (SiLU)', alpha=0.6)
        axes[2, 0].plot(x_range.cpu(), y_spline.cpu(), '--', label='Spline', alpha=0.6)
        axes[2, 0].plot(x_range.cpu(), (y_base + y_spline).cpu(), 'r', lw=2, label='Combined')
        axes[2, 0].set_title(f"Activation: Edge ({edge_idx[0]} -> {edge_idx[1]})")
        axes[2, 0].legend()
        axes[2, 0].grid(True)

    # --- (6) 层间导通热力图 ---
    with torch.no_grad():
        w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.base_scales)
        active_mask = (w_strength > torch.abs(layer.tau)).float()
        sns.heatmap(active_mask.cpu().numpy(), ax=axes[2, 1], annot=False, cmap='Blues', cbar=False)
        axes[2, 1].set_title(f"Layer {layer_idx}: Edge Conductivity")
        axes[2, 1].set_xlabel("Output Neurons")
        axes[2, 1].set_ylabel("Input Neurons")

    plt.show()


# --- 1. W1 深度分析：对齐柱状图与热力图 ---
def visualize_w1_analysis(layer, layer_idx, neuron_idx):
    if layer.last_W1 is None: return None
    W1_data = layer.last_W1[0].detach().cpu().numpy()
    col_ratio = np.sum(W1_data, axis=0) / (np.sum(W1_data) + 1e-8)

    fig = plt.figure(figsize=(8, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[25, 1], hspace=0.05, wspace=0.02)

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.bar(range(len(col_ratio)), col_ratio, color='skyblue', width=0.8)
    ax_bar.set_xlim(-0.5, len(col_ratio) - 0.5)
    ax_bar.set_title(f"L{layer_idx}-N{neuron_idx}: W1 Attention Analysis (Image Pixels)", fontsize=14)
    ax_bar.grid(True, linestyle='--', alpha=0.5)

    ax_heat = fig.add_subplot(gs[1, 0])
    sns.heatmap(W1_data, ax=ax_heat, cmap='magma', cbar=False)
    ax_heat.set_xlabel("Pixel Sequence Index (0-255)")

    ax_cbar = fig.add_subplot(gs[1, 1])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(W1_data.min(), W1_data.max()), cmap='magma'), cax=ax_cbar)
    plt.show()
    return col_ratio


# --- 2. 原始与预处理图像对比 ---
def visualize_images_comparison(raw_img, label_name):
    """
    仅显示原始图像，并标注类别名称
    raw_img: [3, 32, 32] 原始图像数据
    label_name: 字符串，当前样本的类别
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # 原始图反归一化 (CIFAR-10 标准参数)
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
    img_show = np.clip(raw_img * std + mean, 0, 1).transpose(1, 2, 0)

    ax.imshow(img_show)
    ax.set_title(f"Initial Image: {label_name} (32x32 RGB)", fontsize=14)
    ax.axis('off')  # 隐藏坐标轴
    plt.show()


# --- 3. B-样条映射图 (蓝黄虚线版) ---
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

        ax.plot(x_range.cpu(), y_spline.cpu(), color='#1f77b4', linestyle='--', label='Spline $f(x)$')
        ax.plot(x_range.cpu(), y_linear.cpu(), color='#ff7f0e', linestyle=':', label=r'Linear $\omega \cdot x$')
        ax.plot(x_range.cpu(), y_final.cpu(), 'r', lw=2.5, label='Total Activation')
        ax.plot(x_range.cpu(), x_range.cpu(), color='lightgray', alpha=0.5, label='Identity')

        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.6', color='lightgray', alpha=0.7)
        ax.grid(which='minor', linestyle=':', linewidth='0.4', color='gray', alpha=0.3)
        ax.set_title(f"L{layer_idx}-N{neuron_idx} Edge{edge_idx}: Activation Function")
        ax.legend()
    plt.show()


# --- 4. 参数审计与能量图 ---
def visualize_params_and_active_edges(layer, layer_idx, neuron_idx):
    weight_layer = layer.weight_layers[neuron_idx]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    plt.subplots_adjust(hspace=0.3)

    params = {'alpha': torch.abs(weight_layer.alpha).detach().item(),
              'beta': torch.abs(weight_layer.beta).detach().item(),
              'theta': torch.abs(weight_layer.theta).detach().item(),
              'gamma': torch.abs(weight_layer.gamma).detach().item()}
    ax1.bar(params.keys(), params.values(), color='royalblue', alpha=0.7)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
    active_energy = w_strength.clone()
    active_energy[w_strength <= torch.abs(layer.tau)] = np.nan
    sns.heatmap(active_energy.detach().cpu().numpy(), ax=ax2, cmap='YlGn', annot=True, fmt=".2f", linewidths=1)
    plt.show()


# --- 5. 注意力引导图像 (黑白背景 + 注意力上色) ---
def visualize_attention_images(raw_img, col_ratio, label_name):
    """
    调整两个子图大小为完全相同，并合并显示
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                             gridspec_kw={'width_ratios': [1, 1.07]})

    # 1. 原始图像反归一化
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
    rgb_img = np.clip(raw_img * std + mean, 0, 1).transpose(1, 2, 0)

    # 2. 准备注意力掩码 (16x16 -> 32x32 对齐像素)
    attn_mask = col_ratio.reshape(16, 16)
    attn_mask_norm = (attn_mask - attn_mask.min()) / (attn_mask.max() - attn_mask.min() + 1e-8)
    attn_upsampled = zoom(attn_mask_norm, 2, order=1)

    # --- 左图：原始 RGB 图像 ---
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Initial RGB Image: {label_name}", fontsize=12)
    axes[0].axis('off')

    # --- 右图：黑白背景 + 注意力红色着色 ---
    bw_img = np.dot(rgb_img[..., :3], [0.2989, 0.5870, 0.1140])
    axes[1].imshow(bw_img, cmap='gray')
    # 使用 Reds 颜色映射，alpha=0.6 确保轮廓可见
    im = axes[1].imshow(attn_upsampled, cmap='Reds', alpha=0.7)
    axes[1].set_title("Attention Heatmap (Overlaid)", fontsize=12)
    axes[1].axis('off')

    # --- 核心修改：确保 Colorbar 不会挤压右图，使两个图像框等大 ---
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)  # 颜色条放置在侧边且不占主图位
    plt.colorbar(im, cax=cax, label='Attention Intensity')

    plt.suptitle(f"Audit Analysis: Category [{label_name}]", fontsize=16, y=0.95)
    plt.show()

# --- 6. 3D 与 7. 2D 融合矩阵 ---
def visualize_fusion_matrices(layer, layer_idx, neuron_idx):
    """
    修复：构造正确的 2D 卷积 Block Toeplitz 展开矩阵
    """
    weight_layer = layer.weight_layers[neuron_idx]
    seq_len = 256  # 16x16
    H, W = 16, 16

    # 提取 3x3 卷积核
    kernel = weight_layer.W3_conv.weight.detach().cpu()[0, 0].numpy()
    k_size = kernel.shape[0]
    pad = k_size // 2

    # --- 核心修复：模拟滑窗构造稀疏卷积矩阵 ---
    toe_mat = np.zeros((seq_len, seq_len))
    for i in range(H):
        for j in range(W):
            row_idx = i * W + j
            for m in range(-pad, pad + 1):
                for n in range(-pad, pad + 1):
                    ni, nj = i + m, j + n
                    # 边界检查
                    if 0 <= ni < H and 0 <= nj < W:
                        col_idx = ni * W + nj
                        toe_mat[row_idx, col_idx] = kernel[m + pad, n + pad]

    if layer.last_W1 is not None:
        W1 = layer.last_W1[0].cpu().numpy()
        alpha, beta, theta = [torch.abs(getattr(weight_layer, p)).item() for p in ['alpha', 'beta', 'theta']]
        # 融合矩阵：β*W1 (全局) + θ*W3 (局部带状) + α*I (恒等)
        W_combined = beta * W1 + theta * toe_mat + alpha * np.eye(seq_len)

        # 3D 视图：此时可清晰观察到平行的条带结构
        fig3d = plt.figure(figsize=(12, 10))
        ax3d = fig3d.add_subplot(111, projection='3d')
        W_smooth = gaussian_filter(W_combined, sigma=0.4)
        X, Y = np.meshgrid(np.arange(seq_len), np.arange(seq_len))
        ax3d.plot_surface(X, Y, W_smooth, cmap=plt.cm.Blues, alpha=0.8, rcount=128, ccount=128)
        ax3d.view_init(elev=30, azim=250)
        plt.show()

        # 2D 热力图：能直观看到主对角线及偏移为 ±16 的旁路带
        plt.figure(figsize=(10, 8))
        sns.heatmap(W_combined, cmap=plt.cm.Blues)
        plt.title(f"L{layer_idx}-N{neuron_idx}: 2D Fusion Matrix (Block Toeplitz)")
        plt.show()


# --- 8. 全局拓扑图 ---
def visualize_network_topology(model):
    model.eval()
    layers = [m for m in model.modules() if isinstance(m, FastKANLayer)]
    layer_sizes = [layers[0].in_neurons] + [l.out_neurons for l in layers]
    fig, ax = plt.subplots(figsize=(16, 10))
    pos = {}
    x_space = 2.0
    for l_idx, size in enumerate(layer_sizes):
        x = l_idx * x_space
        y_positions = np.linspace(-size / 2, size / 2, size)
        for n_idx, y in enumerate(y_positions):
            pos[(l_idx, n_idx)] = (x, y)

    all_lines, colors = [], []
    with torch.no_grad():
        for l_idx, layer in enumerate(layers):
            w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
            tau = torch.abs(layer.tau)
            for i in range(w_strength.shape[0]):
                for j in range(w_strength.shape[1]):
                    if w_strength[i, j] > tau[i, j]:
                        all_lines.append([pos[(l_idx, i)], pos[(l_idx + 1, j)]])
                        colors.append(w_strength[i, j].item())

    from matplotlib.collections import LineCollection
    lc = LineCollection(all_lines, cmap=plt.cm.Blues, array=np.array(colors),
                        linewidths=1.0 + 3.0 * (np.array(colors) / (max(colors) + 1e-8)), alpha=0.8)
    ax.add_collection(lc)
    for p in pos.values(): ax.scatter(p[0], p[1], s=120, c='white', edgecolors='black', zorder=3)
    ax.set_title("CIFAR-KAN Global Topology")
    plt.show()


def analyze_image_representation(model, dataloader, device, n_samples=3000):
    """
    针对 CIFAR-10 的非监督审计流程
    展平 (10x256=2560) -> 归一化 -> UMAP 降维 -> HDBSCAN 聚类
    """
    model.eval()
    all_features = []
    all_labels = []
    collected = 0
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print(f">>> 正在提取 {n_samples} 个 CIFAR-10 样本的输出层表征...")
    with torch.no_grad():
        for batch_data, batch_label in dataloader:
            # 1. 提取 [B, 10, 256] 特征并拉平为 2560 维
            input_5ch = add_coordinate_encoding(batch_data.to(device))
            latent = model(input_5ch, return_latent=True)
            flat_feat = latent.reshape(latent.size(0), -1)  # [B, 2560]

            # 先对空间维度取平均，只保留 10 个类别的得分指纹
            #flat_feat = torch.mean(latent, dim=-1)  # [B, 10]

            all_features.append(flat_feat.cpu().numpy())
            all_labels.append(batch_label.numpy())

            collected += batch_data.size(0)
            if collected >= n_samples: break

    X = np.concatenate(all_features, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # 2. 标准化与降维
    X_scaled = StandardScaler().fit_transform(X)
    print(">>> 正在进行 UMAP 流形投影 (2560D -> 2D)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    # 3. HDBSCAN 聚类
    print(">>> 正在进行非监督聚类审计...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=40)
    cluster_labels = clusterer.fit_predict(embedding)

    # 4. 绘图对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    # 左图：真实类别 (10色)
    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=y_true,
                           cmap='tab10', s=15, alpha=0.6)
    ax1.set_title("CIFAR-10 Latent Space: Ground Truth (10 Classes)", fontsize=14)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_ticks(range(10))
    cbar1.set_ticklabels(cifar10_classes)

    # 右图：非监督簇 (HDBSCAN 自动发现)
    scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels,
                           cmap='Spectral', s=15, alpha=0.6)
    ax2.set_title("CIFAR-10 Latent Space: HDBSCAN Clusters", fontsize=14)
    plt.colorbar(scatter2, ax=ax2, label='Cluster ID (-1: Noise)')

    plt.suptitle("CIFAR-10 Output Layer Feature Audit (Flattened 2560D Representation)", fontsize=16)
    plt.show()


def visualize_convolution_residual_final(model, layer_idx, neuron_idx, x_prime_input, sample_img, device):
    """
    审计逻辑：可视化 Delta_x = theta * W3(x')
    修复：通过指定 neuron_idx 精确提取神经元特征，解决 4D 解包错误
    """
    layer = getattr(model, f'layer{layer_idx}')
    weight_layer = layer.weight_layers[neuron_idx]

    model.eval()
    with torch.no_grad():
        # --- 核心修复：从 4D 中提取特定神经元数据 ---
        # x_prime_input 形状为 [Batch, in_neurons, seq_len, feat_dim]
        if x_prime_input.dim() == 4:
            # 仅提取当前正在审计的神经元对应输入路径
            # 逻辑：对于该层第 j 个输出神经元，它接收上一层所有神经元的均值/聚合
            # 根据 forward 逻辑，输入已经过聚合，此处 squeeze 掉 Batch 后的 dim=1
            x_target = x_prime_input.squeeze(1) if x_prime_input.size(1) == 1 else x_prime_input[:, 0, :, :]
        else:
            x_target = x_prime_input

        # 1. 执行层内投影与归一化
        x_norm = weight_layer.ln(x_target)
        x_projected = torch.matmul(x_norm, weight_layer.W2) + weight_layer.b

        # 2. 动态获取形状并解包
        # 此时 x_projected 必定为 [Batch, Seq_len, Hidden]
        b, seq_len, h_dim = x_projected.shape
        side = int(np.sqrt(seq_len))

        # 3. 转换至空间维度进行卷积
        x_spatial = x_projected.transpose(1, 2).view(b, h_dim, side, side)

        # 4. 计算残差变化量 Delta_x = theta * W3
        w3_output = weight_layer.W3_conv(x_spatial)
        delta_x = torch.abs(weight_layer.theta) * w3_output

        # 5. 聚合能量：计算该神经元下所有通道的平均修正强度
        delta_energy = torch.mean(torch.abs(delta_x[0]), dim=0).cpu().numpy()

    # --- 绘图部分 (修复 LaTeX 转义警告) ---
    plt.figure(figsize=(7, 6))
    im = plt.imshow(delta_energy, cmap='magma')
    plt.title(r"Layer {} Conv Residual: $\Delta x = \theta \cdot W3$".format(layer_idx), fontsize=12)
    plt.colorbar(im, label=r"Residual Intensity ($\theta \cdot W3$)")
    plt.show()


def print_layer_parameter_summary(model):
    """
    输出每一层的核心参数均值，并新增温度系数 T 的输出
    """
    # 更新表头，增加 Temperature (T) 列
    header = f"{'Layer Name':<12} | {'beta (W1)':>10} | {'theta (W3)':>10} | {'alpha (I)':>10} | {'gamma (Res)':>10} | {'omiga (Edge)':>12} | {'Temp (T)':>10}"
    print(header)
    print("-" * len(header))

    found_any = False
    for name, layer in model.named_modules():
        # 检查是否为包含审计参数的 FastKANLayer 层
        if hasattr(layer, 'weight_layers') and hasattr(layer, 'omiga') and hasattr(layer, 'temperature'):
            found_any = True

            # 1. 提取温度系数 T (在 forward 中通常取绝对值以保证数值意义)
            temp_val = torch.abs(layer.temperature).item()

            # 2. 计算 omiga 的均值
            omiga_mean = torch.mean(torch.abs(layer.omiga)).item()

            # 3. 遍历该层所有神经元提取 beta, theta, alpha, gamma
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

            # 打印结果，包含温度系数 T
            print(f"{name:<12} | {avg_beta:10.4f} | {avg_theta:10.4f} | {avg_alpha:10.4f} | {avg_gamma:10.4f} | {omiga_mean:12.4f} | {temp_val:10.4f}")

    if not found_any:
        print("!!! Warning: No FastKANLayer found. Check model structure.")


def count_parameters(model):
    """
    计算并打印模型的总参数量和可训练参数量
    """
    # 计算所有参数的总和
    total_params = sum(p.numel() for p in model.parameters())
    # 仅计算 requires_grad=True 的参数（用于迁移学习/冻结场景）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"==================================================")
    print(f"Model Summary:")
    print(f"   Total Parameters:     {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"==================================================")

    return total_params, trainable_params


# --- 在文件顶部的 import 部分添加 ---
try:
    from thop import profile
    from thop import clever_format
except ImportError:
    print("提示: 未安装 thop 库，无法计算 FLOPs。请运行 'pip install thop'")


# --- 在 ECGKANModel 类下方添加计算函数 ---
def calculate_model_complexity(model, device):
    """
    计算模型的 FLOPs 和参数量
    input_size: [Batch, Channels, Height, Width] -> [1, 5, 32, 32]
    """
    model.eval()
    # 构造一个符合要求的随机输入张量
    dummy_input = torch.randn(1, 5, 32, 32).to(device)

    try:
        # 使用 thop 进行分析
        # 注意: inputs 必须以元组形式传递
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

        # 格式化输出 (自动转为 G, M, K 等单位)
        readable_flops, readable_params = clever_format([flops, params], "%.3f")

        print(f"\n" + "=" * 50)
        print(f"模型计算复杂度审计 (Model Complexity Audit):")
        print(f"   总计 FLOPs (计算量):  {readable_flops}")
        print(f"   总计 Parameters (参数): {readable_params}")
        print(f"   单次推理 (GFLOPs):    {flops / 1e9:.4f} G")
        print(f"=" * 50 + "\n")

        return flops, params
    except Exception as e:
        print(f"FLOPs 计算失败: {e}")
        return 0, 0


# --- 修改 if __name__ == "__main__": 部分 ---
if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 模型初始化
    model = ECGKANModel(grid_size=64, spline_order=3).to(device)

    # 2. [新增] 计算 FLOPs 和参数量审计
    calculate_model_complexity(model, device)

    # 3. 加载权重并执行后续审计
    checkpoint_path = 'D:\\treasure\\true\\thesis\\pth\\cifar10\\layer7\\best_model_7th_9172.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        print(f">>> 成功加载权重: {checkpoint_path}")

    # ... (后续原有的审计代码) ...

# # --- 测试运行 ---
# # --- 修改后的主函数：加载本地权重并可视化 ---
# if __name__ == "__main__":
#     seed_everything(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 1. 模型与数据加载
#     model = ECGKANModel(grid_size=64, spline_order=3).to(device)
#     train_loader, test_loader, _ = get_dataloader(batch_size=1)
#
#     # 加载权重
#     checkpoint_path = 'D:\\treasure\\true\\thesis\\pth\\cifar10\\layer7\\best_model_7th_9172.pth'
#     cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#     if os.path.exists(checkpoint_path):
#         model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
#
#     # count_parameters(model)
#     # 2. --- 执行非监督聚类审计 (新增) ---
#     # 使用较大的 batch 以提高特征提取速度
#     #audit_loader, _, _ = get_dataloader(batch_size=64)
#     print("\n--- 正在启动 CIFAR-10 隐空间表征审计 ---")
#     #analyze_image_representation(model, audit_loader, device, n_samples=3000)
#
#     # 2. 提取特定样本 (例如第 5 个测试样本)
#     target_sample_idx = 1
#     target_layer_idx, target_neuron_idx = 5, 10  # (0:6, 0:15)
#     edge = (5, 5)
#
#     iterator = iter(test_loader)
#     for _ in range(target_sample_idx + 1):
#         sample_img, sample_label = next(iterator)
#
#     # 3. 准备审计目标
#     target_layer = getattr(model, f'layer{target_layer_idx}')
#     target_layer.visualize_idx = target_neuron_idx
#     label_str = cifar10_classes[sample_label.item()]
#     print(f"\n>>> 正在审计样本索引: {target_sample_idx} | 真实类别: {label_str}")
#
#     model.eval()
#     with torch.no_grad():
#         input_5ch = add_coordinate_encoding(sample_img.to(device))
#         #x_prime_input = model(input_5ch, audit_layer=target_layer_idx)
#         logits = model(input_5ch)
#         probs = F.softmax(logits, dim=1)  # 转化为概率
#         conf, pred_idx = torch.max(probs, dim=1)
#
#     # 提取数值用于打印
#     true_idx = sample_label.item()
#     predicted_idx = pred_idx.item()
#
#     true_name = cifar10_classes[true_idx]
#     pred_name = cifar10_classes[predicted_idx]
#
#     # 核心修复：使用 .item() 后的整数进行比较
#     is_correct = true_idx == predicted_idx
#     status_str = "[CORRECT] ✓" if is_correct else "[WRONG] ✗"
#
#     print(f"\n" + "=" * 50)
#     print(f">>> 审计报告: 真实类别 [{true_name}] | 预测结果 [{pred_name}]")
#     print(f">>> 判断状态: {status_str}")
#     print(f">>> 预测置信度: {conf.item() * 100:.2f}%")  # 输出正确概率
#
#     #visualize_convolution_residual_final(model, target_layer_idx, target_neuron_idx, x_prime_input, sample_img, device)
#
#     # 4. 依次触发 8 窗口
#     # print("\n>>> 正在生成 CIFAR-10 全量审计可视化...")
#     # # (1) W1 对齐分析
#     # col_ratio = visualize_w1_analysis(target_layer, target_layer_idx, target_neuron_idx)
#     # # (5) 注意力引导图 (黑白背景+蓝色着色)
#     # visualize_attention_images(sample_img[0].numpy(), col_ratio, label_str)
#     # # (2) 图像对比 (Raw vs 16x16 Feature)
#     # #visualize_images_comparison(sample_img[0].numpy(), label_str)
#     # # (3) 激活映射
#     # visualize_spline_mapping_window(target_layer, target_layer_idx, target_neuron_idx, edge)
#     # # (4) 参数审计
#     # #visualize_params_and_active_edges(target_layer, target_layer_idx, target_neuron_idx)
#     # # (6 & 7) 融合矩阵
#     # visualize_fusion_matrices(target_layer, target_layer_idx, target_neuron_idx)
#     # # (8) 全局拓扑
#     # #visualize_network_topology(model)
