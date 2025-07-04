import sys
import os

import torch
from torch import nn, optim
from captcha_model import net
from captcha_data import dataloader, val_loader
from tqdm import tqdm

# 添加ONNX导出函数 - 适配单通道灰度图
def export_to_onnx(model, device, output_path="captcha_model.onnx"):
    model.eval()
    # 创建虚拟输入（与真实输入尺寸相同）: [batch, channels, height, width]
    dummy_input = torch.randn(1, 1, 60, 160).to(device)  # 单通道灰度图
    
    # 设置动态batch维度
    dynamic_axes = {
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    }
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    print(f"模型已导出为ONNX格式: {output_path}")

epoch_lr = [
    (1000, 0.1),
    (100, 0.01),
    (100, 0.001),
    (100, 0.0001),
]  # [(300,0.05),(100,0.001),(100,0.0001)]
# 自动检测GPU可用性并选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criteron = nn.MultiLabelSoftMarginLoss()


def train():
    net.to(device)
    best_val_acc = 0.0  # 记录最佳验证准确率
    best_model_path = "best_captcha_model.pth"
    # 创建保存模型的目录
    os.makedirs("saved_models", exist_ok=True)

    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []
    for n, (num_epoch, lr) in enumerate(epoch_lr):
        # 优化器也可以多尝试一下，一般来说使用SGD的对应的学习率会比Adam大一个数量级
        optimizer = optim.SGD(
            net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
        for epoch in range(num_epoch):
            epoch_loss = 0.0
            epoch_acc = 0.0
            for i, (img, label) in tqdm(enumerate(dataloader)):
                out = net(img.to(device))
                label = label.to(device)
                # 清空net里面所有参数的梯度
                optimizer.zero_grad()
                # 计算预测值与目标值之间的损失
                loss = criteron(out, label.to(device))
                # 计算梯度
                loss.backward()
                # 根据梯度调整net中的参数
                optimizer.step()
                # 整理输出，方便与标签进行对比
                predict = torch.argmax(out.view(-1, 36), dim=1)
                true_label = torch.argmax(label.view(-1, 36), dim=1)
                epoch_acc += torch.sum(predict == true_label).item()
                epoch_loss += loss.item()
            # 每训练三次验证一次
            if epoch % 3 == 0:
                # no_grad()模式不计算梯度，可以跑得快一点
                with torch.no_grad():
                    val_loss = 0.0
                    val_acc = 0.0
                    for i, (img, label) in tqdm_notebook(enumerate(val_loader)):
                        out = net(img.to(device))
                        label = label.to(device)
                        loss = criteron(out, label.to(device))
                        predict = torch.argmax(out.view(-1, 36), dim=1)
                        true_label = torch.argmax(label.view(-1, 36), dim=1)
                        val_acc += torch.sum(predict == true_label).item()
                        val_loss += loss.item()
                val_acc /= len(val_loader.dataset) * 4
                val_loss /= len(val_loader)
            epoch_acc /= len(dataloader.dataset) * 4
            epoch_loss /= len(dataloader)
            print(
                "epoch : {} , epoch loss : {} , epoch accuracy : {}".format(
                    epoch + sum([e[0] for e in epoch_lr[:n]]),
                    epoch_loss,
                    epoch_acc,
                )
            )
            if epoch % 3 == 0:
                print(
                    "epoch : {} , val loss : {} , val accuracy : {}".format(
                        epoch + sum([e[0] for e in epoch_lr[:n]]),
                        val_loss,
                        val_acc,
                    )
                )
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_path = f"saved_models/best_model_epoch_{epoch}_acc_{best_val_acc:.4f}.pth"
                    torch.save(net.state_dict(), best_model_path)
                    print(f"保存最佳模型，验证准确率: {best_val_acc:.4f}")
                    
                for i in range(3):
                    val_accuracies.append(val_acc)
                    val_losses.append(val_loss)
            accuracies.append(epoch_acc)
            losses.append(epoch_loss)
            
    # 训练结束后导出ONNX模型
    net.load_state_dict(torch.load(best_model_path))
    onnx_path = "saved_models/captcha_model.onnx"
    export_to_onnx(net, device, onnx_path)
    
    print(f"训练完成! 最佳模型已保存至: {best_model_path}")
    print(f"ONNX模型已导出至: {onnx_path}")

if __name__ == "__main__":
    train()
