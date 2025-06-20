import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import numpy as np
import os
import matplotlib.pyplot as plt

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# 使用CIFAR-10中的两个类别（猫和狗）替代原始数据集
selected_classes = [3, 5]  # 3=猫, 5=狗
class_names = ['cat', 'dog']

# 数据预处理和增强
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 全局自定义数据集类
class CustomTransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        img, label = self.subset[index]
        # 重新映射标签：3->0(猫), 5->1(狗)
        mapped_label = 0 if label == 3 else 1
        if self.transform:
            img = self.transform(img)
        return img, mapped_label
        
    def __len__(self):
        return len(self.subset)

# 创建CIFAR-10数据集（只选择猫和狗两类）
def create_cifar10_subset(train=True):
    full_dataset = datasets.CIFAR10(root='./cifar10_data', train=train, download=True)
    
    # 筛选出猫(3)和狗(5)的样本
    indices = [i for i, (_, label) in enumerate(full_dataset) if label in selected_classes]
    
    # 创建子集
    subset = torch.utils.data.Subset(full_dataset, indices)
    
    return CustomTransformDataset(subset, transform=data_transforms['train' if train else 'valid'])

# 训练函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=15):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
    accuracy = corrects.double() / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# 推理示例
def predict_image(image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted[0]]

# 可视化函数
def imshow(inp, title=None):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

if __name__ == '__main__':
    # 创建数据集
    image_datasets = {
        'train': create_cifar10_subset(train=True),
        'valid': create_cifar10_subset(train=False)
    }

    print(f"训练集大小: {len(image_datasets['train'])}")
    print(f"验证集大小: {len(image_datasets['valid'])}")

    # 创建数据加载器
    batch_size = 32
    # 尝试使用4个工作进程，如果不行可以设置为0
    num_workers = 4
    # 如果在macOS上出现多进程问题，可以尝试将num_workers设置为0
    # num_workers = 0
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'valid': torch.utils.data.DataLoader(
            image_datasets['valid'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    # 检查GPU可用性
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载预训练ResNet18模型
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 冻结所有卷积层参数
    for param in model.parameters():
        param.requires_grad = False

    # 修改最后一层全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # 学习率调度器
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练模型
    num_epochs = 15
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs)

    # 保存模型
    os.makedirs('./models', exist_ok=True)
    torch.save(model, './models/cifar10_cat_dog_resnet18.pth')

    # 保存为TorchScript格式
    model.eval()
    example_input = torch.rand(1, 3, 224, 224).to(device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save('./models/cifar10_cat_dog_resnet18.pt')

    # 评估模型
    print("Evaluating final model...")
    val_accuracy = evaluate_model(model, dataloaders['valid'])

    # 随机样本预测
    sample_idx = torch.randint(0, len(image_datasets['valid']), (1,)).item()
    sample_image, sample_label = image_datasets['valid'][sample_idx]
    prediction = predict_image(sample_image)
    actual = class_names[sample_label]
    print(f"Prediction: {prediction}, Actual: {actual}")

    # 可视化样本
    plt.figure()
    imshow(sample_image, title=f'Predicted: {prediction}\nActual: {actual}')
    plt.savefig('./sample_prediction.png')
    plt.show()
