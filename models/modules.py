import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# class FourierGraphConv(nn.Module):
#     def __init__(self, embed_size, weight_generator, sparsity_threshold=0.01):
#         """
#         单层 Fourier-Gauss GNN (简化版)
#         :param embed_size: 输入特征维度 D
#         :param weight_generator: 权重生成模块，用于生成两个权重矩阵 (B, D, D)
#         :param sparsity_threshold: 稀疏化阈值
#         """
#         super().__init__()
#         self.embed_size = embed_size
#         self.sparsity_threshold = sparsity_threshold
#
#         # 权重生成模块
#         self.weight_generator = weight_generator  # 生成 weight_real 和 weight_imag
#
#         # 隐藏层权重
#         self.w_hidden = nn.Parameter(torch.empty(embed_size, embed_size))
#         self.b_hidden = nn.Parameter(torch.empty(embed_size))
#
#         # 权重初始化
#         nn.init.xavier_uniform_(self.w_hidden)
#         nn.init.zeros_(self.b_hidden)
#
#     def forward(self, x):
#         B, N, D = x.shape
#
#         # Step 1: Graph Fourier Transform (GFT)
#         x_freq = torch.fft.rfft(x, dim=1, norm="ortho")  # (B, N//2+1, D)
#
#         # Step 2: 生成频域权重矩阵
#         weight_real, weight_imag = self.weight_generator(x)  # 两个权重矩阵 (B, D, D), (B, D, D)
#
#         # Step 3: 频域权重操作
#         x_freq_real = F.relu(torch.einsum('bnd,bdd->bnd', x_freq.real, weight_real))  # 实部与权重卷积
#         x_freq_imag = F.relu(torch.einsum('bnd,bdd->bnd', x_freq.imag, weight_imag))  # 虚部与权重卷积
#
#         # 稀疏化
#         x_freq_real = F.softshrink(x_freq_real, lambd=self.sparsity_threshold)
#         x_freq_imag = F.softshrink(x_freq_imag, lambd=self.sparsity_threshold)
#
#         # Step 4: 逆傅里叶变换 (iGFT)
#         x_out_freq = torch.complex(x_freq_real, x_freq_imag)
#         x_out = torch.fft.irfft(x_out_freq, n=N, dim=1, norm="ortho")  # (B, N, D)
#
#         # Step 5: 消息传递 (隐藏层操作)
#         x_hidden = F.relu(torch.einsum('bnd,dd->bnd', x, self.w_hidden) + self.b_hidden)  # (B, N, D)
#
#         # Step 6: 合并频域和隐藏层特征
#         x_out = x_out + x_hidden  # (B, N, D)
#
#         return x_out

class FourierGraphConv(nn.Module):
    def __init__(self, embed_size, weight_generator, sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size  # 假设 embed_size = 1（单变量时间序列）
        self.sparsity_threshold = sparsity_threshold

        self.proj = nn.Linear(1, embed_size)
        self.proj1 = nn.Linear(embed_size, 1)
        # 权重生成模块需适配超图结构
        self.weight_generator = weight_generator  # 输出 (B, M, M)

        # 隐藏层参数
        self.w_hidden = nn.Parameter(torch.empty(embed_size, embed_size))
        self.b_hidden = nn.Parameter(torch.empty(embed_size))

        nn.init.xavier_uniform_(self.w_hidden)
        nn.init.zeros_(self.b_hidden)

    def forward(self, x):
        B, N, D = x.shape
        M = N * D

        # Step 1 展平为超图节点信号 (B, N, D) → (B, M, 1)
        x_flat = x.reshape(B, M, -1)  # (B, M, 1)
        x_flat = self.proj(x_flat)

        # Step 2 超图傅里叶变换（在节点维度 M 上）
        x_freq = torch.fft.rfft(x_flat, dim=1, norm="ortho")  # (B, M//2+1, 1)

        # Step 3 生成频域权重（需重新设计权重生成器）
        weight_real, weight_imag = self.weight_generator(x_flat)  # (B, M//2+1, M//2+1)

        # Step 4 频域卷积（需调整为复数操作）
        x_freq_real = torch.einsum('bmk,bkl->bml', x_freq.real, weight_real)
        x_freq_imag = torch.einsum('bmk,bkl->bml', x_freq.imag, weight_imag)
        x_out_freq = torch.complex(x_freq_real, x_freq_imag)

        # Step 5 逆变换重建超图信号
        x_out = torch.fft.irfft(x_out_freq, n=M, dim=1, norm="ortho")  # (B, M, 1)

        # Step 6 残差连接与隐藏层
        x_hidden = F.relu(torch.einsum('bmi,ij->bmj', x_flat, self.w_hidden) + self.b_hidden)
        x_out = x_out + x_hidden  # (B, M, 1)

        # Step 7 恢复原始形状 (B, M, 1) → (B, N, D)
        x_out = self.proj1(x_out)
        x_out = x_out.view(B, N, D)

        return x_out

# class TimeGCN(nn.Module):
#     def __init__(self, embed_size):
#         """
#         时域 GCN 模块
#         :param embed_size: 输入/输出特征维度
#         """
#         super().__init__()
#         self.w_hidden = nn.Parameter(torch.empty(embed_size, embed_size))
#         self.b_hidden = nn.Parameter(torch.empty(embed_size))
#
#         self.proj = nn.Linear(1, embed_size)
#         self.proj1 = nn.Linear(embed_size, 1)
#
#         nn.init.xavier_uniform_(self.w_hidden)
#         nn.init.zeros_(self.b_hidden)
#
#     def forward(self, x):
#         """
#         前向传播
#         :param x: 输入特征 (B, N, D)
#         :return: 输出特征 (B, N, D)
#         """
#         B, N, D = x.shape
#         M = N * D
#         x = x.reshape(B, M, -1)  # (B, M, 1)
#         x = self.proj(x)
#         x = F.relu(torch.einsum('bnd,dd->bnd', x, self.w_hidden) + self.b_hidden)  # 线性 + 激活
#         x = self.proj1(x)
#         x = x.view(B, N, D)
#         return x

class TimeGCN(nn.Module):
    def __init__(self, embed_size, num_layers=2):
        """
        多线性层时域 GCN
        :param embed_size: 特征维度
        :param num_layers: 线性变换层数
        """
        super().__init__()
        self.num_layers = num_layers

        # 投影层
        self.proj_in = nn.Linear(1, embed_size)
        self.proj_out = nn.Linear(embed_size, 1)

        # 多组权重矩阵
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(embed_size, embed_size))
            for _ in range(num_layers)
        ])

        # 对应偏置项
        self.biases = nn.ParameterList([
            nn.Parameter(torch.empty(embed_size))
            for _ in range(num_layers)
        ])

        # 初始化
        for w in self.weights:
            nn.init.xavier_uniform_(w)
        for b in self.biases:
            nn.init.zeros_(b)

    def forward(self, x):
        B, N, D = x.shape
        M = N * D

        # 输入投影 (B, M, 1) → (B, M, d)
        h = self.proj_in(x.reshape(B, M, 1))  # 原始输入展平

        # 多线性层串联
        for i in range(self.num_layers):
            h = torch.einsum('bnd,de->bne', h, self.weights[i]) + self.biases[i]  # 层间无激活
            if i < self.num_layers - 1:  # 最后一层不加激活
                h = F.relu(h)

        # 输出投影 (B, M, d) → (B, M, 1)
        return self.proj_out(h).reshape(B, N, D)

# class TimeCNN(nn.Module):
#     def __init__(self, in_channel, embed_size, kernel_size=(3, 3)):
#         """
#         时域 2D CNN 模块
#         :param in_channel: 输入通道数
#         :param embed_size: 输出通道数
#         :param kernel_size: 卷积核大小 (H, W)
#         """
#         super().__init__()
#         self.proj = nn.Linear(1, embed_size)
#         self.proj1 = nn.Linear(embed_size, 1)
#
#         self.conv = nn.Conv2d(in_channel, embed_size, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))
#
#     def forward(self, x):
#         """
#         前向传播
#         :param x: 输入特征 (B, N, D)
#         :return: 输出特征 (B, N, embed_size)
#         """
#         # 假设时域特征可以被扩展为 2D：将序列长度视为 H，特征维度视为 W
#         B, N, D = x.shape
#         M = N * D
#         x_flat = x.reshape(B, M, -1)  # (B, M, 1)
#         x_flat = self.proj(x_flat)
#         x_flat = x_flat.unsqueeze(1)  # (B, 1, N, D) 添加通道维度
#         x_flat = self.conv(x_flat)    # 2D 卷积操作
#         x_flat = F.relu(x_flat)       # 激活函数
#         x_flat = x_flat.squeeze(1)    # (B, N, embed_size) 移除通道维度
#         x_out = self.proj1(x_flat)
#         x_out = x_out.view(B, N, D)
#         return x_out

class TimeCNN1D(nn.Module):
    def __init__(self, input_dim, embed_size=64, kernel_size=3, num_layers=2):
        """
        修复版时域 1D CNN
        :param input_dim: 输入特征维度 (即原始特征维度 D)
        :param embed_size: 输出特征维度
        :param kernel_size: 卷积核长度
        :param num_layers: 卷积层数
        """
        super().__init__()
        self.num_layers = num_layers
        padding = kernel_size // 2

        # 输入通道适配 (输入通道数 = 原始特征维度 D)
        self.in_conv = nn.Conv1d(input_dim, embed_size, kernel_size=1)

        # 多层1D卷积
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_size, embed_size, kernel_size, padding=padding),
                nn.BatchNorm1d(embed_size),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: 输入特征 (B, N, D)
        :return: 输出特征 (B, N, embed_size)
        """
        B, N, D = x.shape

        # 输入整形: (B, N, D) → (B, D, N) [D作为通道维度]
        x = x.permute(0, 2, 1)  # (B, D, N)

        # 初始投影 (B, D, N) → (B, embed_size, N)
        x = self.in_conv(x)

        # 多层卷积
        for conv in self.conv_layers:
            x = conv(x)

        # 输出整形: (B, embed_size, N) → (B, N, embed_size)
        return x.permute(0, 2, 1)

class FreqCNN(nn.Module):
    def __init__(self, embed_size, kernel_size=(3, 3)):
        """
        频域 2D CNN 模块
        :param embed_size: 输出特征维度
        :param kernel_size: 卷积核大小 (H, W)
        """
        super().__init__()
        self.proj = nn.Linear(1, embed_size)
        self.proj1 = nn.Linear(embed_size, 1)
        self.conv = nn.Conv2d(2, embed_size, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))

    def forward(self, x):
        """
        前向传播
        :param x: 输入信号 (B, N, D)
        :return: 处理后的信号 (B, N, embed_size)
        """
        B, N, D = x.shape
        M = N * D
        x = x.reshape(B, M, -1)  # (B, M, 1)
        x = self.proj(x)

        # Step 1: Fourier Transform
        x_freq = torch.fft.rfft(x, dim=1, norm="ortho")  # 频域变换 (B, N//2+1, D)

        # Step 2: Stack real and imaginary parts
        x_freq_combined = torch.stack([x_freq.real, x_freq.imag], dim=1)  # (B, 2, N//2+1, D)

        # Step 3: Apply 2D CNN
        x_freq_out = self.conv(x_freq_combined)  # (B, embed_size, N//2+1, D)

        # Step 4: Inverse Fourier Transform
        x_freq_out = x_freq_out.permute(0, 2, 3, 1)  # 调整维度为 (B, N//2+1, D, embed_size)
        x_freq_out_complex = torch.complex(x_freq_out[..., 0], x_freq_out[..., 1])  # 还原复数 (B, N//2+1, D)

        x_out = torch.fft.irfft(x_freq_out_complex, n=N, dim=1, norm="ortho")  # 逆变换 (B, N, embed_size)
        x_out = self.proj1(x_out)
        x_out = x_out.view(B, N, D)
        return x_out

class TimeAttention(nn.Module):
    def __init__(self, embed_size, n_heads=4):
        """
        时域 Attention 模块 (使用官方 MultiheadAttention)
        :param embed_size: 输入/输出特征维度
        :param n_heads: 注意力头数
        """
        super().__init__()
        self.proj = nn.Linear(1, embed_size)
        self.proj1 = nn.Linear(embed_size, 1)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 (B, N, embed_size)
        :return: 输出特征 (B, N, embed_size)
        """
        B, N, D = x.shape
        M = N * D
        x = x.reshape(B, M, -1)  # (B, M, 1)
        x = self.proj(x)
        attn_output, _ = self.attention(x, x, x)  # 使用自身作为 Q, K, V
        attn_output = self.proj1(attn_output)
        attn_output = attn_output.view(B, N, D)
        return attn_output

class FreqAttention(nn.Module):
    def __init__(self, embed_size, n_heads=4):
        """
        频域 Attention 模块 (使用官方 MultiheadAttention)
        :param embed_size: 输入/输出特征维度
        :param n_heads: 注意力头数
        """
        super().__init__()
        self.proj = nn.Linear(1, embed_size)
        self.proj1 = nn.Linear(embed_size, 1)
        self.attention_real = nn.MultiheadAttention(embed_dim=embed_size, num_heads=n_heads, batch_first=True)
        self.attention_imag = nn.MultiheadAttention(embed_dim=embed_size, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        """
        前向传播
        :param x: 输入信号 (B, N, D)
        :return: 处理后的信号 (B, N, D)
        """
        B, N, D = x.shape
        M = N * D
        x = x.reshape(B, M, -1)  # (B, M, 1)
        x = self.proj(x)

        # Step 1: Fourier Transform
        x_freq = torch.fft.rfft(x, dim=1, norm="ortho")  # 频域变换 (B, N//2+1, D)

        # Step 2: 分别对实部和虚部应用 Multihead Attention
        x_freq_real, _ = self.attention_real(x_freq.real, x_freq.real, x_freq.real)  # (B, N//2+1, D)
        x_freq_imag, _ = self.attention_imag(x_freq.imag, x_freq.imag, x_freq.imag)  # (B, N//2+1, D)

        # Step 3: 将实部和虚部结合为复数
        x_out_freq = torch.complex(x_freq_real, x_freq_imag)  # (B, N//2+1, D)

        # Step 4: Inverse Fourier Transform
        x_out = torch.fft.irfft(x_out_freq, n=N, dim=1, norm="ortho")  # (B, N, D)
        x_out = self.proj1(x_out)
        x_out = x_out.view(B, N, D)
        return x_out


if __name__ == '__main__':
    # 测试代码
    B, C, T = 8, 32, 16  # 批量大小、通道数、时间步数
    x = torch.randn(B, C, T)  # 输入信号


    # 权重生成器
    class DummyWeightGenerator(nn.Module):
        def forward(self, x):
            B, D, _ = x.shape
            return torch.randn(B, D, D), torch.randn(B, D, D)

    from base_fun import CosineSineBasis
    # 初始化模型
    weight_generator = CosineSineBasis(embed_size=16)
    model = FourierHypergraphConv(embed_size=16, weight_generator=weight_generator)

    # 前向传播
    output = model(x)
    print("Output shape:", output.shape)  # 应为 (B, C, T)