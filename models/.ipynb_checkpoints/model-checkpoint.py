import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import FourierGraphConv, TimeGCN, TimeCNN, FreqCNN, TimeAttention, FreqAttention  # 不同的图卷积变体
from models.base_fun import LinearAttention, GaussianBasis, MLPBasis, CosineSineBasis  # 权重生成器，需要先经过LinearAttention 然后加一个base
from models.wtconv1d import WTConv1d, _ScaleModule


class EEGEmotionModel(nn.Module):
    def __init__(self, 
                 in_channels, 
                 embed_size, 
                 num_classes, 
                 use_wsd=True, 
                 use_graph=True, 
                 wavelet_level=2, 
                 graph_variant="time_gcn", 
                 graph_layer=2, 
                 weight_generator=None, 
                 kernel_size=5, 
                 stride=1, 
                 wt_type="db1",
                 add_noise=False, 
                 noise_std=0.1):
        """
        EEG 情感预测模型
        :param in_channels: 输入信号通道数
        :param embed_size: 嵌入特征维度
        :param num_classes: 分类类别数
        :param use_wsd: 是否使用小波分解
        :param use_graph: 是否使用图卷积
        :param wavelet_level: 小波分解层数
        :param graph_variant: 图卷积类型
        :param graph_layer: 图卷积层数
        :param weight_generator: 权重生成器（仅在频域图卷积中使用）
        :param kernel_size: 卷积核大小
        :param stride: 步幅
        :param wt_type: 小波类型
        :param add_noise: 是否对输入数据添加噪声
        :param noise_std: 噪声强度（标准差）
        """
        super(EEGEmotionModel, self).__init__()

        self.use_wsd = use_wsd
        self.use_graph = use_graph
        self.add_noise = add_noise
        self.noise_std = noise_std

        self.embed = nn.Linear(in_channels, embed_size)

        # 小波分解模块
        if use_wsd:
            self.wavelet = WTConv1d(embed_size, embed_size, wt_levels=wavelet_level, wt_type=wt_type)

        # 图卷积模块
        if use_graph:
            if graph_variant == "time_gcn":
                self.graph_conv = TimeGCN(embed_size, graph_layer)
            elif graph_variant == "time_cnn":
                self.graph_conv = nn.ModuleList([
                    TimeCNN(embed_size) for i in range(graph_layer)
                ])
            elif graph_variant == "time_att":
                self.graph_conv = nn.ModuleList([
                    TimeAttention(embed_size) for _ in range(graph_layer)
                ])
            elif graph_variant == "fft_cnn":
                self.graph_conv = nn.ModuleList([
                    FreqCNN(embed_size, weight_generator) for _ in range(graph_layer)
                ])
            elif graph_variant == "fft_att":
                self.graph_conv = nn.ModuleList([
                    FreqAttention(embed_size, weight_generator) for _ in range(graph_layer)
                ])
            elif graph_variant == "fft_hyp":
                self.graph_conv = nn.ModuleList([
                    FourierGraphConv(embed_size, weight_generator) for _ in range(graph_layer)
                ])
            else:
                raise ValueError(f"未知的 graph_variant: {graph_variant}")

        # 基础卷积和缩放模块
        self.base_conv = nn.Conv1d(embed_size, embed_size, kernel_size, padding="same", stride=1, bias=True)
        self.base_scale = _ScaleModule([1, embed_size, 1])

        # 步幅逻辑
        self.stride = stride
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(embed_size, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv1d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=embed_size)
        else:
            self.do_stride = None

        # 分类器 (三层全连接网络)
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, embed_size // 4),
            nn.ReLU(),
            nn.Linear(embed_size // 4, num_classes)
        )

    def forward(self, x):
        """
        前向传播
        :param x: 输入信号 (B, C, T)
        :return: 分类预测结果 (B, num_classes)
        """
        x = x.squeeze(1)  # 去掉多余的维度 (B, C, T)

        # Step 0: 添加噪声（可选）
        if self.add_noise:
            noise = torch.randn_like(x) * self.noise_std  # 生成高斯噪声
            x = x + noise

        x = self.embed(x.permute(0, 2, 1)).permute(0, 2, 1)
        B, C, T = x.shape

        # Step 1: 小波分解（可选）
        if self.use_wsd:
            x_ll_in_levels, x_h_in_levels, shapes_in_levels = self.wavelet.decompose(x)
        else:
            x_ll_in_levels, x_h_in_levels, shapes_in_levels = [x], [torch.zeros_like(x)], [x.shape]

        # Step 2: 图卷积（可选）
        if self.use_graph:
            x_ll_processed = []
            x_h_processed = []
            for x_ll, x_h in zip(x_ll_in_levels, x_h_in_levels):
                x_ll = x_ll.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
                for layer in self.graph_conv:
                    x_ll = layer(x_ll)
                x_ll_processed.append(x_ll.permute(0, 2, 1))

                x_h = x_h.permute(0, 2, 1)
                for layer in self.graph_conv:
                    x_h = layer(x_h)
                x_h_processed.append(x_h.permute(0, 2, 1))
        else:
            x_ll_processed = x_ll_in_levels
            x_h_processed = x_h_in_levels

        # Step 3: 信号还原
        if self.use_wsd:
            x_tag = self.wavelet.reconstruct(x_ll_processed, x_h_processed, shapes_in_levels)
        else:
            x_tag = x

        # Step 4: 基础卷积处理并融合
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        # 应用步幅
        if self.do_stride is not None:
            x = self.do_stride(x)

        # Step 5: 分类器
        x_flattened = x.mean(dim=-1)  # 时序特征全局平均池化 (B, C)
        logits = self.classifier(x_flattened)  # (B, num_classes)

        return logits

def build_model(config):
    """
    根据配置构建 EEG 情感预测模型
    :param config: 包含模型配置的字典，示例：
        {
            "graph_variant": "fft_hyp",  # 图卷积变体
            "use_wsd": True,            # 是否使用小波分解
            "use_graph": True,          # 是否使用图卷积
            "wavelet_level": 2,         # 小波分解层数
            "base_fun": "linear",       # 权重生成器
            "graph_layer": 2,           # 图卷积层数
            "kernel_size": 5,           # 卷积核大小
            "stride": 2,                # 步幅
            "add_noise": False,         # 是否添加噪声
            "noise_std": 0.1            # 噪声标准差
        }
    :return: EEGEmotionModel 实例
    """
    # 从配置中提取参数
    graph_variant = config.get("graph_variant", "time_gcn")
    use_wsd = config.get("use_wsd", True)
    use_graph = config.get("use_graph", True)
    wavelet_level = config.get("wavelet_level", 2)
    base_fun = config.get("base_fun", "sine")
    graph_layer = config.get("graph_layer", 2)
    kernel_size = config.get("kernel_size", 5)
    stride = config.get("stride", 1)
    add_noise = config.get("add_noise", False)  # 是否添加噪声
    noise_std = config.get("noise_std", 0.1)    # 噪声标准差

    # 输入和模型参数
    in_channels = 32
    embed_size = 64
    num_classes = 4
    wt_type = "db1"

    # 构建权重生成器
    weight_generator = None
    if graph_variant in ["fft_cnn", "fft_att", "fft_hyp"]:
        if base_fun == "linear":
            weight_generator = MLPBasis(embed_size, embed_size * 2)
        elif base_fun == "gauss":
            weight_generator = GaussianBasis(embed_size)
        elif base_fun == "sine":
            weight_generator = CosineSineBasis(embed_size)
        else:
            raise ValueError(f"未知的 base_fun: {base_fun}")

    # 实例化模型
    model = EEGEmotionModel(
        in_channels=in_channels,
        embed_size=embed_size,
        num_classes=num_classes,
        use_wsd=use_wsd,
        use_graph=use_graph,
        wavelet_level=wavelet_level,
        graph_variant=graph_variant,
        graph_layer=graph_layer,
        weight_generator=weight_generator,
        kernel_size=kernel_size,
        stride=stride,
        wt_type=wt_type,
        add_noise=add_noise,  # 添加噪声参数
        noise_std=noise_std   # 噪声强度参数
    )
    return model
    
if __name__ == '__main__':
    # 示例配置
    config = {
    "graph_variant": "fft_hyp",
    "use_wsd": True,
    "use_graph": True,
    "wavelet_level": 3,
    "base_fun": "sine",
    "graph_layer": 3,
    "kernel_size": 5,
    "stride": 2,
    "add_noise": True,  # 启用加噪声
    "noise_std": 0.05   # 噪声标准差
}

    # 构建模型
    model = build_model(config)

    # 示例输入
    B, C, T = 8, 32, 128  # 批量大小、通道数、时间序列长度
    x = torch.randn(B, 1, C, T)  # 输入信号

    # 前向传播
    logits = model(x)
    print(logits.shape)  # 输出: (B, num_classes)