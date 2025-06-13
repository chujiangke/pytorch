import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将 1D 特征扩展为适合 CNN 的 2D 形式（转换为伪"图像"，如 2x2 或 1x4 的"灰度图像"）
X = X.reshape(-1, 1, 2, 2)  # 这里转换为单通道 2x2 图像（假设输入有 4 个特征）

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# 构建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2)  # 输入通道为1，输出通道为16，卷积核大小2x2
        self.pool = nn.MaxPool2d(kernel_size=1)       # 最大池化窗口为1x1（不改变尺寸）
        self.fc1 = nn.Linear(16 * 1 * 1, 64)         # 展平后输入为16，输出为64
        self.fc2 = nn.Linear(64, 3)                  # 输出层，类别数为3

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))         # 卷积 + 激活 + 池化
        x = torch.flatten(x, 1)                     # 展平
        x = F.relu(self.fc1(x))                     # 全连接 + 激活
        x = self.fc2(x)                             # 输出层
        return x

# 初始化模型
model = CNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()               # 梯度清零
        outputs = model(inputs)             # 前向传播
        loss = criterion(outputs, labels)   # 计算损失
        loss.backward()                     # 反向传播
        optimizer.step()                    # 更新权重
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # 获取最大值的索引
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# 将模型保存为 ONNX 格式
onnx_file_path = "iris_dataset.onnx"
# 从 train_loader 中获取一个 batch 数据
dummy_input, _ = next(iter(train_loader))  # 获取一个批次的输入数据
print(f"Dummy input shape: {dummy_input.shape}")

# 导出模型
torch.onnx.export(model,               # 要导出的模型
                  dummy_input,         # 一个样本输入数据，用于推导模型的输入形状
                  onnx_file_path,      # 保存路径
                  export_params=True,  # 是否保存模型参数
                  opset_version=12,    # ONNX 版本
                  do_constant_folding=True,  # 是否进行常量折叠优化
                  input_names=['input'],  # 输入名称
                  output_names=['output'],  # 输出名称
                  dynamic_axes={'input': {0: 'batch_size'},  # 动态批量大小
                                'output': {0: 'batch_size'}})

print(f"Model has been successfully saved to {onnx_file_path}")