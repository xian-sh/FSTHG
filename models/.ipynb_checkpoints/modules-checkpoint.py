import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.nn import init
import torch.nn as nn
from typing import Tuple, List, Dict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


class FourierGraphConv(nn.Module):
    def __init__(self, embed_size, weight_generator, sparsity_threshold=0.01):
        """
        单层 Fourier-Gauss GNN (简化版)
        :param embed_size: 输入特征维度 D
        :param weight_generator: 权重生成模块，用于生成两个权重矩阵 (B, D, D)
        :param sparsity_threshold: 稀疏化阈值
        """
        super(FourierGraphConv, self).__init__()
        self.embed_size = embed_size
        self.sparsity_threshold = sparsity_threshold

        # 权重生成模块
        self.weight_generator = weight_generator  # 生成 weight_real 和 weight_imag

        # 隐藏层权重
        self.w_hidden = nn.Parameter(torch.empty(embed_size, embed_size))
        self.b_hidden = nn.Parameter(torch.empty(embed_size))

        # 权重初始化
        nn.init.xavier_uniform_(self.w_hidden)
        nn.init.zeros_(self.b_hidden)

    def forward(self, x):
        B, N, D = x.shape

        # Step 1: Graph Fourier Transform (GFT)
        x_freq = torch.fft.rfft(x, dim=1, norm="ortho")  # (B, N//2+1, D)

        # Step 2: 生成频域权重矩阵
        weight_real, weight_imag = self.weight_generator(x)  # 两个权重矩阵 (B, D, D), (B, D, D)

        # Step 3: 频域权重操作
        x_freq_real = F.relu(torch.einsum('bnd,bdd->bnd', x_freq.real, weight_real))  # 实部与权重卷积
        x_freq_imag = F.relu(torch.einsum('bnd,bdd->bnd', x_freq.imag, weight_imag))  # 虚部与权重卷积

        # 稀疏化
        x_freq_real = F.softshrink(x_freq_real, lambd=self.sparsity_threshold)
        x_freq_imag = F.softshrink(x_freq_imag, lambd=self.sparsity_threshold)

        # Step 4: 逆傅里叶变换 (iGFT)
        x_out_freq = torch.complex(x_freq_real, x_freq_imag)
        x_out = torch.fft.irfft(x_out_freq, n=N, dim=1, norm="ortho")  # (B, N, D)

        # Step 5: 消息传递 (隐藏层操作)
        x_hidden = F.relu(torch.einsum('bnd,dd->bnd', x, self.w_hidden) + self.b_hidden)  # (B, N, D)

        # Step 6: 合并频域和隐藏层特征
        x_out = x_out + x_hidden  # (B, N, D)

        return x_out


class TimeGCN(nn.Module):
    def __init__(self, embed_size):
        """
        时域 GCN 模块
        :param embed_size: 输入/输出特征维度
        """
        super(TimeGCN, self).__init__()
        self.w_hidden = nn.Parameter(torch.empty(embed_size, embed_size))
        self.b_hidden = nn.Parameter(torch.empty(embed_size))
        nn.init.xavier_uniform_(self.w_hidden)
        nn.init.zeros_(self.b_hidden)

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 (B, N, D)
        :return: 输出特征 (B, N, D)
        """
        x = F.relu(torch.einsum('bnd,dd->bnd', x, self.w_hidden) + self.b_hidden)  # 线性 + 激活
        return x

class TimeCNN(nn.Module):
    def __init__(self, in_channel, embed_size, kernel_size=(3, 3)):
        """
        时域 2D CNN 模块
        :param in_channel: 输入通道数
        :param embed_size: 输出通道数
        :param kernel_size: 卷积核大小 (H, W)
        """
        super(TimeCNN, self).__init__()
        self.conv = nn.Conv2d(in_channel, embed_size, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 (B, N, D)
        :return: 输出特征 (B, N, embed_size)
        """
        # 假设时域特征可以被扩展为 2D：将序列长度视为 H，特征维度视为 W
        B, N, D = x.shape
        x = x.unsqueeze(1)  # (B, 1, N, D) 添加通道维度
        x = self.conv(x)    # 2D 卷积操作
        x = F.relu(x)       # 激活函数
        x = x.squeeze(1)    # (B, N, embed_size) 移除通道维度
        return x

class FreqCNN(nn.Module):
    def __init__(self, embed_size, weight_generator, kernel_size=(3, 3)):
        """
        频域 2D CNN 模块
        :param embed_size: 输入/输出特征维度
        :param weight_generator: 权重生成模块
        :param kernel_size: 卷积核大小 (H, W)
        """
        super(FreqCNN, self).__init__()
        self.weight_generator = weight_generator
        self.conv = nn.Conv2d(1, embed_size, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))

    def forward(self, x):
        B, N, D = x.shape

        # Step 1: Fourier Transform
        x_freq = torch.fft.rfft(x, dim=1, norm="ortho")  # (B, N//2+1, D)
        weight_real, weight_imag = self.weight_generator(x)  # (B, D, D), (B, D, D)

        # Step 2: Apply weights
        x_freq_real = F.relu(torch.einsum('bnd,bdd->bnd', x_freq.real, weight_real))
        x_freq_imag = F.relu(torch.einsum('bnd,bdd->bnd', x_freq.imag, weight_imag))

        # Step 3: Combine real and imaginary parts
        x_freq_combined = torch.stack([x_freq_real, x_freq_imag], dim=1)  # (B, 2, N//2+1, D)

        # Step 4: Apply 2D CNN
        x_freq_out = self.conv(x_freq_combined).squeeze(1)  # (B, N//2+1, embed_size)

        # Step 5: Inverse Fourier Transform
        x_out = torch.fft.irfft(x_freq_out, n=N, dim=1, norm="ortho")  # (B, N, embed_size)
        return x_out

class TimeAttention(nn.Module):
    def __init__(self, embed_size, n_heads=4):
        """
        时域 Attention 模块 (使用官方 MultiheadAttention)
        :param embed_size: 输入/输出特征维度
        :param n_heads: 注意力头数
        """
        super(TimeAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 (B, N, embed_size)
        :return: 输出特征 (B, N, embed_size)
        """
        attn_output, _ = self.attention(x, x, x)  # 使用自身作为 Q, K, V
        return attn_output

class FreqAttention(nn.Module):
    def __init__(self, embed_size, weight_generator, n_heads=4):
        """
        频域 Attention 模块 (使用官方 MultiheadAttention)
        :param embed_size: 输入/输出特征维度
        :param weight_generator: 权重生成模块
        :param n_heads: 注意力头数
        """
        super(FreqAttention, self).__init__()
        self.weight_generator = weight_generator
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        B, N, D = x.shape

        # Step 1: Fourier Transform
        x_freq = torch.fft.rfft(x, dim=1, norm="ortho")  # (B, N//2+1, D)
        weight_real, weight_imag = self.weight_generator(x)  # (B, D, D), (B, D, D)

        # Step 2: Apply weights
        x_freq_real = F.relu(torch.einsum('bnd,bdd->bnd', x_freq.real, weight_real))
        x_freq_imag = F.relu(torch.einsum('bnd,bdd->bnd', x_freq.imag, weight_imag))

        # Step 3: Apply Attention (on real part)
        attn_output, _ = self.attention(x_freq_real, x_freq_real, x_freq_real)  # (B, N//2+1, D)

        # Step 4: Combine real and imaginary parts
        x_out_freq = torch.complex(attn_output, x_freq_imag)

        # Step 5: Inverse Fourier Transform
        x_out = torch.fft.irfft(x_out_freq, n=N, dim=1, norm="ortho")  # (B, N, D)
        return x_out