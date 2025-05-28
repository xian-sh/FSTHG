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


class LinearAttention(nn.Module):
    def __init__(self, embed_size):
        """
        简单线性注意力模块，用于生成与序列长度 N 无关的全局权重
        :param embed_size: 输入特征维度 D
        """
        super(LinearAttention, self).__init__()
        self.q_proj = nn.Linear(embed_size, embed_size)  # Q 的投影
        self.k_proj = nn.Linear(embed_size, embed_size)  # K 的投影
        self.v_proj = nn.Linear(embed_size, embed_size)  # V 的投影
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局池化，用于聚合注意力输出

    def forward(self, x):
        """
        输入数据 x (B, N, D)
        :return: 初始化权重 (B, D)，注意力输出的全局表示
        """
        B, N, D = x.shape

        # Step 1: 生成 Q、K 和 V
        Q = self.q_proj(self.global_pool(x.transpose(1, 2)).squeeze(-1))  # 全局 Q (B, D)
        K = self.k_proj(x)  # 局部 K (B, N, D)
        V = self.v_proj(x)  # 局部 V (B, N, D)

        # Step 2: 计算注意力权重
        attention_scores = torch.einsum('bd,bnd->bn', Q, K) / np.sqrt(D)  # (B, N)
        attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)  # (B, N, 1)

        # Step 3: 加权生成注意力输出
        attention_output = attention_weights * V  # (B, N, D)

        # Step 4: 全局聚合注意力输出
        initial_weights = attention_output.mean(dim=1)  # 将 (B, N, D) 聚合为 (B, D)

        return initial_weights


class GaussianBasis(nn.Module):
    def __init__(self, embed_size):
        """
        高斯基模块，用于生成两个独立的高斯权重矩阵
        :param embed_size: 输入特征维度 D
        """
        super(GaussianBasis, self).__init__()
        self.att = LinearAttention(embed_size)
        
        # 高斯参数 1
        self.mean1 = nn.Parameter(torch.zeros(embed_size))  # 高斯分布 1 的均值 μ1 (D)
        self.std1 = nn.Parameter(torch.ones(embed_size))   # 高斯分布 1 的标准差 σ1 (D)

        # 高斯参数 2
        self.mean2 = nn.Parameter(torch.zeros(embed_size))  # 高斯分布 2 的均值 μ2 (D)
        self.std2 = nn.Parameter(torch.ones(embed_size))   # 高斯分布 2 的标准差 σ2 (D)

    def forward(self, x):
        """
        输入初始化权重并生成两个独立的高斯加权结果
        :param weights: 初始化权重 (B, D)
        :return: 两个高斯权重矩阵 (B, D, D), (B, D, D)
        """
        weights = self.att(x)
        B, D = weights.shape

        # Step 1: 创建索引 (0, 1, 2, ..., D-1)
        x_indices = torch.arange(D, device=weights.device).float()  # (D)

        # Step 2: 计算高斯矩阵 1
        x_diff1 = (x_indices.unsqueeze(1) - x_indices.unsqueeze(0))  # (D, D)
        gaussian_matrix1 = torch.exp(-((x_diff1 - self.mean1.view(-1, 1)) ** 2) / (2 * self.std1.view(-1, 1) ** 2))  # (D, D)
        gaussian_matrix1 = gaussian_matrix1.unsqueeze(0).expand(B, -1, -1)  # 扩展到 (B, D, D)

        # Step 3: 计算高斯矩阵 2
        x_diff2 = (x_indices.unsqueeze(1) - x_indices.unsqueeze(0))  # (D, D)
        gaussian_matrix2 = torch.exp(-((x_diff2 - self.mean2.view(-1, 1)) ** 2) / (2 * self.std2.view(-1, 1) ** 2))  # (D, D)
        gaussian_matrix2 = gaussian_matrix2.unsqueeze(0).expand(B, -1, -1)  # 扩展到 (B, D, D)

        # Step 4: 分别将权重广播并逐元素相乘
        weighted_output1 = weights.unsqueeze(-1) * gaussian_matrix1  # (B, D, D)
        weighted_output2 = weights.unsqueeze(-1) * gaussian_matrix2  # (B, D, D)

        return weighted_output1, weighted_output2


class MLPBasis(nn.Module):
    def __init__(self, embed_size, hidden_size):
        """
        MLP 基模块，用于生成两个权重矩阵并与输入权重相乘
        :param embed_size: 输入特征维度 D
        :param hidden_size: MLP 隐藏层大小
        """
        super(MLPBasis, self).__init__()

        self.att = LinearAttention(embed_size)
        
        # MLP 1
        self.mlp1 = nn.Sequential(
            nn.Linear(embed_size, hidden_size),  # 第一层全连接
            nn.ReLU(),                          # 激活函数
            nn.Linear(hidden_size, embed_size),  # 第二层全连接
            nn.ReLU()                           # 激活函数
        )

        # MLP 2
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_size, hidden_size),  # 第一层全连接
            nn.ReLU(),                          # 激活函数
            nn.Linear(hidden_size, embed_size),  # 第二层全连接
            nn.ReLU()                           # 激活函数
        )

    def forward(self, x):
        """
        输入初始化权重并生成两个 MLP 加权结果
        :param weights: 初始化权重 (B, D)
        :return: 两个 MLP 权重矩阵 (B, D, D), (B, D, D)
        """
        weights = self.att(x)
        B, D = weights.shape

        # Step 1: 使用 MLP1 生成权重矩阵 1
        mlp_weights1 = self.mlp1(weights)  # (B, D)
        mlp_matrix1 = mlp_weights1.unsqueeze(2) * mlp_weights1.unsqueeze(1)  # (B, D, D)

        # Step 2: 使用 MLP2 生成权重矩阵 2
        mlp_weights2 = self.mlp2(weights)  # (B, D)
        mlp_matrix2 = mlp_weights2.unsqueeze(2) * mlp_weights2.unsqueeze(1)  # (B, D, D)

        return mlp_matrix1, mlp_matrix2


class CosineSineBasis(nn.Module):
    def __init__(self, embed_size):
        """
        Cosine 和 Sine 基模块，用于生成两个权重矩阵
        :param embed_size: 输入特征维度 D
        """
        super(CosineSineBasis, self).__init__()
        
        self.att = LinearAttention(embed_size)
        # 可学习的频率参数和偏移量参数
        self.frequency_cos = nn.Parameter(torch.randn(embed_size))  # Cosine 的频率参数 (D)
        self.frequency_sin = nn.Parameter(torch.randn(embed_size))  # Sine 的频率参数 (D)
        self.phase_cos = nn.Parameter(torch.zeros(embed_size))      # Cosine 的偏移量参数 (D)
        self.phase_sin = nn.Parameter(torch.zeros(embed_size))      # Sine 的偏移量参数 (D)

    def forward(self, x):
        """
        输入初始化权重并生成两个独立的 Cosine 和 Sine 加权结果
        :param weights: 初始化权重 (B, D)
        :return: 两个权重矩阵 (B, D, D), (B, D, D)
        """
        weights = self.att(x)
        B, D = weights.shape

        # Step 1: 创建索引矩阵
        x_indices = torch.arange(D, device=weights.device).float()  # (D)
        x_matrix = x_indices.unsqueeze(1) - x_indices.unsqueeze(0)  # (D, D)

        # Step 2: 生成 Cosine 权重矩阵
        cosine_matrix = torch.cos(
            self.frequency_cos.view(-1, 1) * x_matrix + self.phase_cos.view(-1, 1)
        )  # (D, D)
        cosine_matrix = cosine_matrix.unsqueeze(0).expand(B, -1, -1)  # (B, D, D)

        # Step 3: 生成 Sine 权重矩阵
        sine_matrix = torch.sin(
            self.frequency_sin.view(-1, 1) * x_matrix + self.phase_sin.view(-1, 1)
        )  # (D, D)
        sine_matrix = sine_matrix.unsqueeze(0).expand(B, -1, -1)  # (B, D, D)

        # Step 4: 广播输入权重并逐元素相乘
        weighted_cosine = weights.unsqueeze(-1) * cosine_matrix  # (B, D, D)
        weighted_sine = weights.unsqueeze(-1) * sine_matrix      # (B, D, D)

        return weighted_cosine, weighted_sine



