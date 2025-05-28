import torch as t
import torch.nn as nn


class ACRNN(nn.Module):
    def __init__(self, num_channels=62, time_steps=5, num_classes=3):
        super(ACRNN, self).__init__()
        # 配置
        self.num_channels = num_channels  # 输入的通道数
        self.time_steps = time_steps  # 输入的时间步长
        self.cnn_kernel = 40  # CNN卷积核个数
        self.cnn_kernel_width = 3  # CNN卷积核宽度
        self.hidden_dim = 64  # LSTM隐藏层维度
        self.dropout_rate = 0.5  # Dropout比率

        # 注册层
        self.Dropout = nn.Dropout(self.dropout_rate)
        self.BatchNorm1d = nn.BatchNorm1d(self.hidden_dim)
        self.BatchNorm2d_1 = nn.BatchNorm2d(1)  # 修改为1通道
        self.BatchNorm2d_2 = nn.BatchNorm2d(self.cnn_kernel)
        self.W1 = nn.Linear(self.num_channels, self.num_channels // 2)  # 降维FC
        self.W2 = nn.Linear(self.num_channels // 2, self.num_channels)  # 升维FC
        self.Tanh = nn.Tanh()
        self.Softmax = nn.Softmax(dim=1)
        self.CNN = nn.Conv2d(1, self.cnn_kernel, (self.num_channels, self.cnn_kernel_width))  # 卷积核跨通道
        self.ELU = nn.ELU()
        self.MaxPool2d = nn.MaxPool2d(kernel_size=(1, 2))  # 池化层核大小
        self.LSTM = nn.LSTM(self.cnn_kernel, self.hidden_dim, num_layers=2, batch_first=True)
        self.Wq = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Wk = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Wv = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W3 = nn.Linear(self.hidden_dim, 16)
        self.relu = nn.ReLU()
        self.W4 = nn.Linear(16, num_classes)  # 分类输出层

    def forward(self, data):
        # data shape: (batch_size, num_channels, time_steps)
        data = data.squeeze(1)
        batch_size = data.shape[0]

        ## 通道注意力机制 (Channel-wise attention)
        s_ave = t.mean(data, dim=-1)  # 平均时间步长 -> (batch_size, num_channels)
        s_tmp = self.relu(self.W1(s_ave))  # 降维 -> (batch_size, num_channels // 2)
        c_imp = self.Softmax(self.W2(s_tmp))  # 通道权重 -> (batch_size, num_channels)

        ## 根据权重调整通道
        c_imp = c_imp.unsqueeze(-1).expand(-1, self.num_channels, self.time_steps)  # 扩展时间步长维度
        X = data * c_imp  # 加权输入 -> (batch_size, num_channels, time_steps)
        X = X.unsqueeze(1)  # 增加通道维度 -> (batch_size, 1, num_channels, time_steps)

        ## 批归一化
        X = self.BatchNorm2d_1(X)

        ## CNN 和池化层
        X = self.relu(self.CNN(X))  # 卷积输出 -> (batch_size, cnn_kernel, 1, time_steps - cnn_kernel_width + 1)
        X = self.MaxPool2d(X)  # 池化 -> (batch_size, cnn_kernel, 1, pooled_time)
        X = self.BatchNorm2d_2(X)

        ## 将 CNN 输出调整为 LSTM 输入
        X = X.squeeze(2)  # 去掉单通道维度 -> (batch_size, cnn_kernel, pooled_time)
        X = X.permute(0, 2, 1)  # 调整维度顺序 -> (batch_size, pooled_time, cnn_kernel)

        ## LSTM
        X, _ = self.LSTM(X)  # LSTM输出 -> (batch_size, pooled_time, hidden_dim)
        X = X[:, -1, :]  # 取最后一个时间步的输出 -> (batch_size, hidden_dim)
        X = self.BatchNorm1d(X)

        ## 自注意力机制 (Self-attention)
        q = self.Wq(X)  # Query -> (batch_size, hidden_dim)
        k = self.Wk(X)  # Key -> (batch_size, hidden_dim)
        v = self.Wv(X)  # Value -> (batch_size, hidden_dim)
        att = self.Softmax(q * k / (self.hidden_dim ** 0.5))  # 注意力分数 -> (batch_size, hidden_dim)
        X = att * v  # 加权值 -> (batch_size, hidden_dim)

        ## 分类层
        X = self.Dropout(X)
        X = self.relu(self.W3(X))  # -> (batch_size, 16)
        X = self.W4(X)  # -> (batch_size, num_classes)
        return X


if __name__ == '__main__':
    # 测试代码
    num_channels = 62  # EEG 通道数
    time_steps = 5  # 时间步长
    num_classes = 3  # 分类数

    # 初始化模型
    net = Net(num_channels=num_channels, time_steps=time_steps, num_classes=num_classes)

    # 模拟输入数据 (batch_size=16, num_channels=62, time_steps=5)
    input_data = t.rand(16, num_channels, time_steps)

    # 前向传播
    output = net(input_data)
    print("Output shape:", output.shape)  # 应输出 (batch_size, num_classes)