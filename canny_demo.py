import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('track_image.jpg', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if image is None:
    print("图像加载失败，请检查文件路径")
    exit()

# 使用 GaussianBlur 进行平滑处理，减少噪声
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Canny 边缘检测，低阈值100，高阈值200
edges = cv2.Canny(blurred, 100, 200)

# 显示结果
plt.figure(figsize=(10, 6))

# 原图
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 边缘检测结果
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.show()
