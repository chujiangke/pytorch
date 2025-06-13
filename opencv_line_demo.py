import cv2
import numpy as np

# 创建一个黑色图像
height, width = 480, 640
image = np.zeros((height, width, 3), dtype=np.uint8)

# 定义线条的起点和终点
start_point = (100, 100)  # 起点坐标 (x, y)
end_point = (500, 400)    # 终点坐标 (x, y)

# 定义线条的颜色和厚度
color = (0, 255, 0)  # 绿色 (B, G, R)
thickness = 5        # 线条宽度

# 在图像上绘制线条
image = cv2.line(image, start_point, end_point, color, thickness)

# 显示结果图像
cv2.imshow("Line Image", image)

# 等待按键后关闭显示窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
