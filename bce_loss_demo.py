
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch import optim


x_data = np.array([[i, 10 - i] for i in range(10)], dtype=np.float32)
y_data = np.array([[1] if i % 2 == 0 else [0] for i in range(10)], dtype=np.float32)

x = torch.tensor(x_data, dtype=torch.float32)
y = torch.tensor(y_data, dtype=torch.float32)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
model = SimpleNN()
model.train()
criterion = nn.BCELoss()  # 二元交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 训练模型
def train(model, criterion, optimizer, x, y, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()  # 清除梯度
        outputs = model(x)  # 前向传播
        loss = criterion(outputs, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return model

def plot_results(model, x, y):  
    model.eval()
    with torch.no_grad():
        predicted = model(x).cpu().numpy()  # 获取预测结果
    import matplotlib.pyplot as plt
    plt.scatter(x[:, 0].cpu().numpy(), y.cpu().numpy(), label='Actual Data', color='blue')
    plt.scatter(x[:, 0].cpu().numpy(), predicted, label='Predicted Data', color='red')
    plt.xlabel('Input Feature 1')
    plt.ylabel('Target Value')
    plt.title('Binary Classification Result')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    trained_model = train(model, criterion, optimizer, x, y, epochs=100)
    plot_results(trained_model, x, y)
    print('Training complete.')
# This code demonstrates a simple binary classification using a neural network with BCE loss.