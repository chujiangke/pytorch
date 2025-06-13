import torch
import torch.nn as nn

# 假设输入的特征图形状为 (batch_size=2, channels=3, height=4, width=4)
x = torch.randn(2, 3, 4, 4)

# 使用 nn.Flatten() 展平
flatten = nn.Flatten()
output = flatten(x)

print(output.shape)  # 输出形状为 (2, 48)，其中 48 = 3 * 4 * 4


