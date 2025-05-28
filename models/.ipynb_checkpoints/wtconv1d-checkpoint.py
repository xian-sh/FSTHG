import pywt
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F

# copy from https://github.com/xuhangc/Plumbing/blob/main/blocks/wtconv1d.py

def create_wavelet_filter_1d(wave, in_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type)
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type)

    # Decomposition filters
    dec_filters = torch.stack([dec_lo, dec_hi], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1)

    # Reconstruction filters
    rec_filters = torch.stack([rec_lo, rec_hi], dim=0)
    rec_filters = rec_filters[:, None].repeat(in_size, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform_1d(x, filters):
    b, c, l = x.shape
    pad = filters.shape[2] // 2 - 1
    x = F.conv1d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 2, l // 2)
    return x

def inverse_wavelet_transform_1d(x, filters):
    b, c, _, l_half = x.shape
    pad = filters.shape[2] // 2 - 1
    x = x.reshape(b, c * 2, l_half)
    x = F.conv_transpose1d(x, filters, stride=2, groups=c, padding=pad)
    return x

# class WTConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
#         super(WTConv1d, self).__init__()

#         assert in_channels == out_channels

#         self.in_channels = in_channels
#         self.wt_levels = wt_levels
#         self.stride = stride

#         self.wt_filter, self.iwt_filter = create_wavelet_filter_1d(wt_type, in_channels, torch.float)
#         self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
#         self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

#         self.wt_function = partial(wavelet_transform_1d, filters=self.wt_filter)
#         self.iwt_function = partial(inverse_wavelet_transform_1d, filters=self.iwt_filter)

#         self.base_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', stride=1, bias=bias)
#         self.base_scale = _ScaleModule([1, in_channels, 1])

#         self.wavelet_convs = nn.ModuleList(
#             [nn.Conv1d(in_channels * 2, in_channels * 2, kernel_size, padding='same', stride=1, bias=False)
#              for _ in range(self.wt_levels)]
#         )
#         self.wavelet_scale = nn.ModuleList(
#             [_ScaleModule([1, in_channels * 2, 1], init_scale=0.1) for _ in range(self.wt_levels)]
#         )

#         if self.stride > 1:
#             self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1), requires_grad=False)
#             self.do_stride = lambda x_in: F.conv1d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
#         else:
#             self.do_stride = None

#     def forward(self, x):
#         x_ll_in_levels = []
#         x_h_in_levels = []
#         shapes_in_levels = []

#         curr_x_ll = x

#         for i in range(self.wt_levels):
#             curr_shape = curr_x_ll.shape
#             shapes_in_levels.append(curr_shape)
#             if curr_shape[2] % 2 > 0:
#                 curr_pads = (0, curr_shape[2] % 2)
#                 curr_x_ll = F.pad(curr_x_ll, curr_pads)

#             curr_x = self.wt_function(curr_x_ll)
#             curr_x_ll = curr_x[:, :, 0, :]

#             shape_x = curr_x.shape
#             curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 2, shape_x[3])
#             curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
#             curr_x_tag = curr_x_tag.reshape(shape_x)

#             x_ll_in_levels.append(curr_x_tag[:, :, 0, :])
#             x_h_in_levels.append(curr_x_tag[:, :, 1, :])

#         next_x_ll = 0

#         for i in range(self.wt_levels - 1, -1, -1):
#             curr_x_ll = x_ll_in_levels.pop()
#             curr_x_h = x_h_in_levels.pop()
#             curr_shape = shapes_in_levels.pop()

#             curr_x_ll = curr_x_ll + next_x_ll

#             curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h.unsqueeze(2)], dim=2)
#             next_x_ll = self.iwt_function(curr_x)

#             next_x_ll = next_x_ll[:, :, :curr_shape[2]]

#         x_tag = next_x_ll
#         assert len(x_ll_in_levels) == 0

#         x = self.base_scale(self.base_conv(x))
#         x = x + x_tag

#         if self.do_stride is not None:
#             x = self.do_stride(x)

#         return x

class WTConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv1d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        # 创建小波滤波器
        self.wt_filter, self.iwt_filter = create_wavelet_filter_1d(wt_type, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 小波变换与逆变换函数
        self.wt_function = partial(wavelet_transform_1d, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform_1d, filters=self.iwt_filter)

        # 基础卷积和缩放模块
        self.base_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', stride=1, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1])

        # 小波卷积和缩放模块（用于高频和低频特征提取）
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv1d(in_channels * 2, in_channels * 2, kernel_size, padding='same', stride=1, bias=False)
             for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 2, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        # 步幅逻辑
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv1d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        # 分解信号
        x_ll_in_levels, x_h_in_levels, shapes_in_levels = self.decompose(x)  # list

        # level = 2，low 2, (512, 256) high 2, (512, 256)
        # level = 3, low 3, (512, 256, 128) high 3, (512, 256, 128) 
        # level = 4, low 4, (512, 256, 128, 64) high 3, (512, 256, 128, 64) 
        # level = 5, low 5, (512, 256, 128, 64, 32) high 3, (512, 256, 128, 64, 32) 
        
        x_ll_processed = x_ll_in_levels 
        x_h_processed = x_h_in_levels

        # 还原信号
        x_tag = self.reconstruct(x_ll_processed, x_h_processed, shapes_in_levels)

        # 基础卷积处理并融合还原信号
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        # 应用步幅
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

    def decompose(self, x):
        """
        分解信号为高频和低频分量
        """
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # 如果信号长度为奇数，进行填充
            if curr_shape[2] % 2 > 0:
                curr_pads = (0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 小波分解
            curr_x = self.wt_function(curr_x_ll)

            # 提取低频和高频分量
            curr_x_ll = curr_x[:, :, 0, :]  # 低频
            curr_x_h = curr_x[:, :, 1, :]  # 高频

            # 保存分量
            x_ll_in_levels.append(curr_x_ll)
            x_h_in_levels.append(curr_x_h)

        return x_ll_in_levels, x_h_in_levels, shapes_in_levels

    def reconstruct(self, x_ll_in_levels, x_h_in_levels, shapes_in_levels):
        """
        根据高频和低频分量还原信号
        """
        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels[i]
            curr_x_h = x_h_in_levels[i]
            curr_shape = shapes_in_levels[i]

            # 融合低频分量
            curr_x_ll = curr_x_ll + next_x_ll

            # 合并高频和低频分量
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h.unsqueeze(2)], dim=2)

            # 逆小波变换
            next_x_ll = self.iwt_function(curr_x)

            # 恢复信号长度
            next_x_ll = next_x_ll[:, :, :curr_shape[2]]

        return next_x_ll

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class DepthwiseSeparableConvWithWTConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConvWithWTConv1d, self).__init__()

        self.depthwise = WTConv1d(in_channels, in_channels, kernel_size=kernel_size)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

if __name__ == '__main__':
    input_data = torch.randn(1, 64, 1024)  # Example input data (batch_size, channels, length)
    # wtconv = DepthwiseSeparableConvWithWTConv1d(in_channels=64, out_channels=128)
    wtconv = WTConv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, bias=True, wt_levels=3, wt_type='db1')
    output_data = wtconv(input_data)
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
