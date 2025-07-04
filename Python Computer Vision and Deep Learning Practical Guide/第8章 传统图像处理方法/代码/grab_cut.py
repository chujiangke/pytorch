import numpy as np 
import cv2
from matplotlib import pyplot as plt

def show(img):
    img_ = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img_)
img = cv2.imread("/data/object_detection_segment/object_detection/011.jpg")
show(img)
def grabcut(img,mask,rect,iters=20):
    img_ = img.copy()
    bg_model = np.zeros((1,65),np.float64)
    fg_model = np.zeros((1,65),np.float64)
    cv2.grabCut(img.copy(),mask,rect,bg_model,fg_model,iters,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_ = img*mask2[:,:,np.newaxis]
    return img_

mask = np.zeros(img.shape[:2],np.uint8)
rect = (40,40,250,260)
img_copy = img.copy()
cv2.rectangle(img_copy,rect[:2],rect[2:],(0,255,0),3)
show(img_copy)
img = grabcut(img,mask,rect)
show(img)
plt.show()
