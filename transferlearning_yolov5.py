import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import shutil
import time
import yaml
from tqdm import tqdm
import cv2
import tarfile
import requests
from ultralytics import YOLO
from PIL import Image
import torch.nn.functional as F

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 创建项目目录结构
def create_project_structure():
    os.makedirs('datasets/images/train', exist_ok=True)
    os.makedirs('datasets/images/val', exist_ok=True)
    os.makedirs('datasets/labels/train', exist_ok=True)
    os.makedirs('datasets/labels/val', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs('cifar10_data', exist_ok=True)

# 创建数据集配置文件
def create_data_yaml():
    # 三个类别：牦牛、藏獒、雪豹
    class_names = ['yak', 'tibetan_mastiff', 'snow_leopard']
    
    data = {
        'train': os.path.abspath('datasets/images/train'),
        'val': os.path.abspath('datasets/images/val'),
        'nc': len(class_names),  # 三个类别
        'names': class_names  # 类别名称
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print("数据集配置文件已创建: data.yaml")
    return class_names

# 下载CIFAR-10数据集
def download_cifar10_dataset():
    print("下载CIFAR-10数据集...")
    
    # 创建目录
    os.makedirs('cifar10_data', exist_ok=True)
    
    # 下载训练集
    train_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    train_path = "cifar10_data/cifar-10-python.tar.gz"
    
    # 如果文件不存在则下载
    if not os.path.exists(train_path):
        print(f"下载训练集: {train_url}")
        response = requests.get(train_url, stream=True)
        with open(train_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    # 解压文件
    print("解压数据集...")
    with tarfile.open(train_path, 'r:gz') as tar:
        tar.extractall(path='cifar10_data')
    
    print("数据集下载完成!")

# 准备CIFAR-10数据集为YOLO格式
def prepare_cifar10_dataset():
    print("准备CIFAR-10数据集...")
    
    # 加载训练集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./cifar10_data', 
        train=True, 
        download=False,  # 我们已经下载过了
        transform=transforms.ToTensor()
    )
    
    # 加载测试集（作为验证集）
    val_dataset = torchvision.datasets.CIFAR10(
        root='./cifar10_data', 
        train=False, 
        download=False,  # 我们已经下载过了
        transform=transforms.ToTensor()
    )
    
    # 使用前三个类别：飞机(0)、汽车(1)、鸟(2) 分别代表牦牛、藏獒、雪豹
    class_indices = [0, 1, 2]
    
    # 保存图像和生成YOLO格式标签
    print("准备训练数据...")
    for dataset, dataset_type in zip([train_dataset, val_dataset], ['train', 'val']):
        # 创建对应的图像和标签目录
        img_dir = f'datasets/images/{dataset_type}'
        label_dir = f'datasets/labels/{dataset_type}'
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        
        # 进度条
        progress_bar = tqdm(total=len(dataset), desc=f'Processing {dataset_type} images')
        
        for idx, (image, label) in enumerate(dataset):
            # 只使用前三个类别
            if label not in class_indices:
                progress_bar.update(1)
                continue
                
            # 转换为PIL图像并保存
            img_path = os.path.join(img_dir, f'image_{idx:05d}.png')
            pil_image = transforms.ToPILImage()(image)
            pil_image.save(img_path)
            
            # 创建YOLO格式标签
            # 由于CIFAR-10没有边界框，我们将创建一个覆盖整个图像的边界框
            label_path = os.path.join(label_dir, f'image_{idx:05d}.txt')
            
            # YOLO格式: class_id x_center y_center width height
            # 这里我们创建一个覆盖整个图像的边界框
            class_id = label  # 0,1或2
            x_center = 0.5  # 图像中心x
            y_center = 0.5  # 图像中心y
            width = 1.0     # 整个宽度
            height = 1.0    # 整个高度
            
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center} {y_center} {width} {height}")
            
            progress_bar.update(1)
        
        progress_bar.close()
    
    print("数据集准备完成!")
    return len(train_dataset), len(val_dataset)

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=128, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
        
        # 加载图像并获取原始尺寸
        image = Image.open(img_path).convert('RGB')
        original_w, original_h = image.size
        
        # 应用变换
        image_tensor = self.transform(image)  # [3, H, W]
        
        # 加载并处理标签
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.split()
                    if len(data) == 5:
                        class_id, x_center, y_center, width, height = map(float, data)
                        # 关键：将归一化坐标缩放到当前img_size
                        x_center = x_center * self.img_size / original_w
                        y_center = y_center * self.img_size / original_h
                        width = width * self.img_size / original_w
                        height = height * self.img_size / original_h
                        labels.append([int(class_id), x_center, y_center, width, height])
        
        return {
            'image': image_tensor,
            'labels': labels,  # 列表的列表: [[class_id, x, y, w, h], ...]
            'img_path': img_path,
            'original_size': (original_h, original_w)
        }


# 替换原有检测层初始化代码
def initialize_detect_biases(model):
    """递归查找并初始化所有Detect层"""
    for name, module in model.named_modules():
        # 兼容v6.x和v7.0+的类名差异
        if hasattr(module, 'initialize_biases') or module.__class__.__name__ == "Detect":
            print(f"初始化检测层: {name}")
            # 统一初始化方法调用
            if hasattr(module, 'initialize_biases'):
                module.initialize_biases()
            # v7.0+的特殊处理
            elif hasattr(model, 'detect'):
                model.detect.initialize_biases()
                
def load_yolov5_model(num_classes=3, freeze_backbone=True):
    """兼容新版YOLOv5的迁移学习模型加载"""
    try:
        print("加载YOLOv5s预训练模型 (兼容版)...")
        # 1. 加载模型并跳过属性访问
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        cfg_data = {}
        # 2. 动态获取配置文件路径（兼容新旧版）
        if hasattr(model.model, 'yaml_file'):
            cfg_data = model.model.yaml  # v6.x
        else:
            # v7.0+ 配置文件固定在仓库中
            cfg_path = 'yolov5/models/yolov5s.yaml'  
            with open(cfg_path) as f:
                cfg_data = yaml.safe_load(f)

        # 3. 更新类别数
        cfg_data['nc'] = num_classes
        print(f"更新类别数: {num_classes}")

        # 4. 重建模型（关键修复点）
        from models.yolo import Model
        custom_model = Model(cfg_data)  
        
        # 5. 权重迁移（排除检测层）
        state_dict = model.state_dict()
        exclude = [k for k in state_dict if '24.' in k]  # 检测层前缀[8](@ref)
        csd = {k: v for k, v in state_dict.items() if k not in exclude}
        custom_model.load_state_dict(csd, strict=False)
        
        # 6. 初始化新检测层
        # 在权重迁移后调用
        initialize_detect_biases(custom_model)
        print("检测层初始化完成")
        
        # 7. 冻结Backbone（按模块名而非层序号）
        if freeze_backbone:
            for name, param in custom_model.named_parameters():
                if 'model.0.' in name:  # 只冻结卷积主干[4](@ref)
                    param.requires_grad = False
            print("冻结Backbone参数")
        
        # 参数统计与设备转移（略）
        custom_model.to(device)
        return custom_model
        
    except Exception as e:
        print(f"加载失败: {e}")
        print("建议直接使用官方训练命令:")
        print("python train.py --weights yolov5s.pt --data data.yaml --epochs 50 --freeze 10")
        return None

class YOLOLoss(nn.Module):
    def __init__(self, num_classes, anchors=(), autobalance=False):
        super().__init__()
        self.num_classes = num_classes
        self.box_gain = 0.05
        self.cls_gain = 0.5
        self.obj_gain = 1.0
        self.autobalance = autobalance
        
        # 锚点框
        self.anchors = anchors
        self.na = len(anchors[0]) // 2  # 每个尺度的锚点数量
        self.nl = len(anchors)  # 预测层数量
        
        # 自动平衡参数
        if autobalance:
            self.balance = {3: [4.0, 1.0, 0.4]}  # 示例平衡参数

    def forward(self, preds, targets):
        """
        preds: 模型输出列表，每个元素为[batch_size, anchors, grid_h, grid_w, box_attrs]
        targets: 目标张量 [num_targets, 6] (batch_idx, class_id, x, y, w, h)
        """
        # 统一输出格式处理
        if isinstance(preds, tuple):
            # 如果是元组，取第一个元素（通常是预测结果）
            preds = preds[0]
        
        if not isinstance(preds, list):
            preds = [preds]
        
        device = targets.device
        
        # 初始化损失分量
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        
        # 目标分配
        tcls, tbox, indices, anchors = self.build_targets(preds, targets)
        
        # 计算损失
        for i, pred in enumerate(preds):  # 遍历每个预测层
            b, a, gj, gi = indices[i]  # 图像索引, 锚点索引, 网格y, 网格x
            tobj = torch.zeros_like(pred[..., 0], device=device)  # 目标obj
            
            n = b.shape[0]  # 目标数量
            if n:
                # 提取预测值
                pxy = pred[b, a, gj, gi, :2].sigmoid() * 2 - 0.5
                pwh = (pred[b, a, gj, gi, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # 预测框
                
                # IoU计算
                iou = self.bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()  # iou损失
                
                # 分类损失
                t = torch.full_like(pred[b, a, gj, gi, 5:], 0.0, device=device)
                t[range(n), tcls[i]] = 1.0
                lcls += F.binary_cross_entropy_with_logits(
                    pred[b, a, gj, gi, 5:], t
                )
                
                # 目标置信度
                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = iou  # iou作为目标置信度
            
            # 对象存在损失
            obj_loss = F.binary_cross_entropy_with_logits(
                pred[..., 4], tobj
            )
            
            # 自动平衡
            if self.autobalance:
                balance = self.balance.get(self.nl, [1.0, 1.0, 1.0])[i]
                obj_loss *= balance
            
            lobj += obj_loss
        
        # 加权损失
        lbox *= self.box_gain
        lobj *= self.obj_gain
        lcls *= self.cls_gain
        
        # 总损失
        loss = lbox + lobj + lcls
        return loss, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, preds, targets):
        """分配目标到锚点框"""
        # 简化的目标分配逻辑
        # 实际实现更复杂，包含多尺度分配和网格对齐
        # 这里返回一个简化版本
        indices = []
        tbox = []
        tcls = []
        anchors = []
        
        for i, pred in enumerate(preds):
            # 实际实现需要根据目标位置分配到不同尺度和网格
            # 这里返回空值作为占位符
            indices.append((torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])))
            tbox.append(torch.tensor([]))
            tcls.append(torch.tensor([]))
            anchors.append(torch.tensor([]))
        
        return tcls, tbox, indices, anchors

    def bbox_iou(self, box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        """计算IoU、GIoU、DIoU或CIoU"""
        # 简化的IoU计算
        # 实际实现包含各种IoU变体
        # 这里使用标准IoU作为示例
        if xywh:
            box1 = torch.cat((box1[..., :2] - box1[..., 2:]/2, 
                             box1[..., :2] + box1[..., 2:]/2), dim=-1)
            box2 = torch.cat((box2[..., :2] - box2[..., 2:]/2, 
                             box2[..., :2] + box2[..., 2:]/2), dim=-1)
        
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)
        
        # 交集区域
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        
        # 并集区域
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = w1 * h1 + w2 * h2 - inter + eps
        
        iou = inter / union
        
        if CIoU or DIoU or GIoU:
            # 实际实现包含更复杂的计算
            return iou
        else:
            return iou


def train_model(model, train_loader, num_epochs=1000, initial_lr=0.01):
    """
    YOLOv5官方推荐的迁移学习训练函数
    包含冻结策略、学习率调度和最佳实践（已移除进度条操作）
    """
    # 验证模型配置
    print("验证模型配置...")
    
    # 初始化损失函数
    anchors = [
        [10,13, 16,30, 33,23],  # P3/8
        [30,61, 62,45, 59,119],  # P4/16
        [116,90, 156,198, 373,326]  # P5/32
    ]
    criterion = YOLOLoss(num_classes=3, anchors=anchors)
    
    # 优化器设置（官方推荐SGD）
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True
    )
    
    # 学习率调度器（余弦退火）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs, 
        eta_min=initial_lr * 0.1  # 最小学习率为初始值的10%
    )
    
    # 训练历史记录
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        # 渐进式解冻策略
        if epoch == num_epochs // 3:  # 1/3训练进度
            print("\n解冻中间层...") 
            # 解冻中间层（骨干网络后半部分）
            for name, param in model.named_parameters():
                if 'model.10.' in name or 'model.13.' in name or 'model.17.' in name:
                    param.requires_grad = True
        
        if epoch == 2 * num_epochs // 3:  # 2/3训练进度
            print("\n解冻所有层...")
            # 解冻所有层
            for param in model.parameters():
                param.requires_grad = True
        
        print(f'开始训练 Epoch {epoch+1}/{num_epochs}')
        start_time = time.time()
        
        # 训练阶段
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device, non_blocking=True)
            labels_list = batch['labels']  # list[list]: 每个元素是图像的标签列表
            batch_size = images.shape[0]
            
            targets = []  # 存储batch内所有目标
            for batch_idx in range(batch_size):
                img_labels = labels_list[batch_idx]  # 当前图像的标签 [[cls, x, y, w, h], ...]
                
                if len(img_labels) > 0:
                    # 直接在GPU创建Tensor (高效)
                    labels_tensor = torch.tensor(img_labels, dtype=torch.float32, device=device)
                    n_objects = labels_tensor.size(0)
                    
                    # 添加batch索引列 [batch_idx, class_id, x, y, w, h]
                    batch_col = torch.full((n_objects, 1), batch_idx, dtype=torch.float32, device=device)
                    img_targets = torch.cat([batch_col, labels_tensor], dim=1)
                    targets.append(img_targets)
            
            # 合并所有目标 (若无目标则创建空Tensor)
            targets = torch.cat(targets, dim=0) if targets else torch.zeros((0, 6), device=device)
            # 混合精度训练
            with torch.cuda.amp.autocast():
                # 前向传播
                raw_output = model(images)
                print(f"模型输出类型: {type(raw_output)}")  # 添加调试
                print(f"输出长度: {len(raw_output) if isinstance(raw_output, (list, tuple)) else 1}")  # 添加调试
                
                # 如果是元组，取第一个元素（通常是预测结果）
                if isinstance(raw_output, tuple):
                    pred = raw_output[0]
                else:
                    pred = raw_output
                
                # 计算损失
                loss, loss_items = criterion(pred, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            # 更新损失记录
            epoch_loss += loss.item()
            
            # 每10个batch打印一次训练状态
            if (i + 1) % 10 == 0:
                avg_loss = epoch_loss / (i + 1)
                elapsed = time.time() - start_time
                print(f'Batch {i+1}/{num_batches} | '
                      f'Loss: {avg_loss:.4f} | '
                      f'Time: {elapsed:.1f}s | '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 更新学习率
        scheduler.step()
        
       
def custom_collate_fn(batch):
    # batch 是一个列表，每个元素是数据集返回的字典
    images = [item['image'] for item in batch]
    labels_list = [item['labels'] for item in batch]  # 每个元素是标签列表（列表的列表）
    img_paths = [item['img_path'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]

    # 堆叠图像
    images = torch.stack(images, dim=0)

    # 其他数据保持为列表（因为标签长度不一致，无法直接堆叠成张量）
    return {
        'image': images,
        'labels': labels_list,   # 注意：这里是一个列表，每个元素是一个图像的标签列表（每个标签是一个包含5个元素的列表）
        'img_path': img_paths,
        'original_size': original_sizes
    }

# 主函数
def main():
    # 创建项目结构
    create_project_structure()
    
    # 创建数据集配置文件
    class_names = create_data_yaml()
    
    # 下载CIFAR-10数据集
    download_cifar10_dataset()
    
    # 准备CIFAR-10数据集
    train_size, val_size = prepare_cifar10_dataset()
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
    
    # 加载数据集
    print("加载自定义数据集...")
    train_dataset = YOLODataset(
        img_dir='datasets/images/train',
        label_dir='datasets/labels/train',
        img_size=128
    )
    
    val_dataset = YOLODataset(
        img_dir='datasets/images/val',
        label_dir='datasets/labels/val',
        img_size=128
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn,  num_workers=2)
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 加载模型
    print("加载YOLOv5模型...")
    model = load_yolov5_model(num_classes=3)
    
    # 训练模型
    print("开始训练...")
    trained_model = train_model(
        model, 
        val_loader
    )
    
    # 保存最终模型
    torch.save(trained_model.state_dict(), 'models/yolov5_cifar_final.pth')
    print("最终模型已保存")
    

if __name__ == "__main__":
    main()
