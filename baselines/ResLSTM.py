import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, hnshape):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(hnshape[1:])  # 仅指定 (seq_len, out_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, ht):
        # ht((batch_size, seq_len, out_dim))
        yt = self.layernorm(ht)
        yt = self.activation(yt)
        return yt


class MMResLstmBlock(nn.Module):
    def __init__(self, indim, outdim, num_layer, batch_size, seq_len, wh):
        super(MMResLstmBlock, self).__init__()
        self.batch_size = batch_size
        self.in_dim = indim
        self.out_dim = outdim
        self.num_layer = num_layer
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=self.out_dim, num_layers=self.num_layer,
                            batch_first=True, bidirectional=False)
        self.lstm.weight_hh_l0 = wh
        self.lstm.flatten_parameters()
        self.layernorm = LayerNorm((self.batch_size, self.seq_len, self.out_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, xt):
        batch_size = xt.size(0)  # 动态获取当前输入的批量大小
        h0 = torch.zeros((self.num_layer, batch_size, self.out_dim), device=xt.device)
        c0 = torch.zeros((self.num_layer, batch_size, self.out_dim), device=xt.device)
        out, (hn, cn) = self.lstm(xt, (h0, c0))  # 使用动态初始化的 h0 和 c0
        out = self.layernorm(out)
        out = self.dropout(out)
        return out

torch.autograd.set_detect_anomaly(True)


class ResLSTM(nn.Module):
    def __init__(self, num_node=62, seq_len=5, num_classes=3):
        super(ResLSTM, self).__init__()
        # 默认参数设置
        self.dim_eeg = num_node  # EEG 输入维度
        self.out_dim = 32  # LSTM 隐藏层输出维度
        self.num_layer = 1  # LSTM 层数
        self.batch_size = 18  # 默认 batch 大小
        self.num_block = 2  # 残差块数量
        self.seq_len = seq_len  # 时间序列长度
        self.num_node = num_node  # 输入节点数量 (EEG)
        self.eeg_blocks = nn.ModuleList()
        self.per1 = Permute()
        self.whs = []
        for i in range(self.num_block):
            self.whs.append(nn.Parameter(torch.Tensor(4 * self.out_dim, self.out_dim)))
        for i in range(self.num_block):
            if i == 0:
                self.eeg_blocks.append(
                    MMResLstmBlock(self.dim_eeg, self.out_dim, self.num_layer, self.batch_size, self.seq_len, self.whs[i]))
            else:
                self.eeg_blocks.append(
                    MMResLstmBlock(self.out_dim, self.out_dim, self.num_layer, self.batch_size, self.seq_len, self.whs[i]))
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(self.seq_len * self.out_dim, num_classes)
        self.act = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_block):
            nn.init.xavier_uniform_(self.whs[i])

    def forward(self, data):
        data = data.squeeze(1)
        x_T = data  # 输入形状 (b, 62, 5)
        x_T = self.per1(x_T)  # 转换为 (b, 5, 62)
        x_eeg = x_T  # EEG 数据 (b, 5, 62)
        out_eeg = x_eeg
        # 分不同的 block 来共享权重
        for i in range(self.num_block):
            if i == 0:
                out_eeg = self.eeg_blocks[i](out_eeg)
            else:
                last_eeg = out_eeg
                out_eeg = self.eeg_blocks[i](out_eeg)
                out_eeg += last_eeg
        out = self.flatten(out_eeg)
        out = self.act(self.dense1(out))
        return out


# 测试代码
if __name__ == "__main__":
    # 模拟输入数据 (batch_size=16, num_nodes=62, seq_len=5)
    batch_size = 16
    num_nodes = 62
    seq_len = 5
    input_data = torch.randn(batch_size, num_nodes, seq_len).to(device)

    # 初始化模型
    model = ResLSTM(num_node=num_nodes, seq_len=seq_len, num_classes=3).to(device)

    # 前向传播
    output = model(input_data)
    print("Output shape:", output.shape)  # 应输出 (batch_size, num_classes)