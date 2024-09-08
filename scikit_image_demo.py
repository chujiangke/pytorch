import skimage
import cv2

image = skimage.data.chelsea()
cv2.imshow("original", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

cv2.waitKey(0)
cv2.destoryAllWindows()

smallBlur = np.one((7, 7), dtype="float") * (1.0/(7*7))

convoletOutput = convolve(gray, smallBlur)
opencvOutput = cv2.filter2D(gray, -1, smallBlur)
cv2.imShow("litter Blur", convoletOutput)

largeBlur = np.ones((21, 21), dtype="float") * (1.0/(21*21))
convoletOutput = cv2.filter2D(gray, -1, largeBlur)
cv2.imshow("large Blur", convoletOutput)

cv2.waitKey(0)
cv2.destoryAllWindows()