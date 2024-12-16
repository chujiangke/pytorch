import cv2
 
# 创建一个黑色的图像
 
 
img = cv2.imread('20240826142433.jpg')
#调整图片大小
new_size = (800, 600)
img = cv2.resize(img, new_size)
# 绘制一条红色的对角线，从左上角到右下角
cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (255, 0, 255), 3)
 
# 显示图像
cv2.imwrite("line_demo.jpg", img)
