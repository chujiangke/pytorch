import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models

# 1. 数据预处理修正：单通道转三通道
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 关键修正：1通道->3通道
    transforms.Resize(224),  # ResNet要求最小224x224输入
    transforms.ToTensor(),
])

# 2. 加载数据
train_set = datasets.FashionMNIST('./fashion_mnist_data', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST('./fashion_mnist_data', train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

# 3. 正确的迁移学习模型
class FashionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 加载预训练模型（关键修正！）
        self.net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 冻结卷积层权重（保留预训练特征）
        for param in self.net.parameters():
            param.requires_grad = False  # 仅当需要特征提取时使用
        
        # 替换全连接层（适配当前任务）
        self.net.fc = nn.Sequential(
            nn.Linear(self.net.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# 3. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 4. 统一训练/验证逻辑
def run_epoch(loader, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, correct = 0, 0
    
    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# 5. 训练循环
for epoch in range(10):
    train_loss, train_acc = run_epoch(train_loader)
    test_loss, test_acc = run_epoch(test_loader, is_train=False)
    
    print(f"Epoch {epoch+1}/10: "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2%}")

# 6. 模型保存
torch.save(model.state_dict(), "fashion_mnist_model.pth")