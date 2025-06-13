import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 精简数据加载
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.FashionMNIST('./fashion_mnist_data', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST('./fashion_mnist_data', train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

# 2. 优化模型结构（使用Sequential组织层）
class FashionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # 卷积块1
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            
            # 卷积块2
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            
            # 展平层
            nn.Flatten(),
            
            # 分类头
            nn.Linear(64 * 4 * 4, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Dropout(0.25),
            nn.Linear(200, num_classes)
        )
    
    def forward(self, x):
        return self.feature_extractor(x)

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