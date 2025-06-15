import torch
import torch.utils
import torch.utils.data
import torchvision

from torch import nn
from torch.nn import  functional as F
from torch import optim
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from torch.nn import functional as F # 导入函数库
from loguru import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_data = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y_data = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
x = torch.tensor(x_data, dtype=torch.float32, device=device)
y = torch.tensor(y_data, dtype=torch.float32, device=device)

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1, bias=True)  # 输入特征维度为1，输出特征维度为1

    def forward(self, x):
        return self.linear(x)
    
model = LinearRegression().to(device)
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器
def train(model, criterion, optimizer, x, y, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()  # 清除梯度
        outputs = model(x)  # 前向传播
        loss = criterion(outputs, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return model

def plot_results(model, x, y):  
    model.eval()
    with torch.no_grad():
        predicted = model(x).cpu().numpy()  # 获取预测结果
    plt.scatter(x.cpu().numpy(), y.cpu().numpy(), label='Actual Data', color='blue')
    plt.plot(x.cpu().numpy(), predicted, label='Fitted Line', color='red')
    plt.xlabel('Input Feature')
    plt.ylabel('Target Value')
    plt.title('Linear Regression Result')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    trained_model = train(model, criterion, optimizer, x, y, epochs=100)
    plot_results(trained_model, x, y)
    logger.info('Training complete.')
    torch.save(trained_model.state_dict(), 'linear_regression_model.pth')
    logger.info('Model saved to linear_regression_model.pth')