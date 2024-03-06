import torch.nn as nn

class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
        # self.layers和self.net没有本质区别 不同的叫法

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # 模型的输出可能是一个包含单个元素的二维张量 实现(B,1) -> (B)
        return x
