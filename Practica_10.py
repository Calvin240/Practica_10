import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread("figure_r.jpg")

mask = np.zeros(image.shape[:2], np.uint8)

backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)

roi = cv2.selectROI(image)
roi_select = image[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]

cv2.grabCut(image,mask,roi,backgroundModel,foregroundModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 
   
image = image * mask2[:, :, np.newaxis]

cv2.imwrite('img_c.jpg',image)

img_c = cv2.imread('img_c.jpg')

gray = cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

height, width = dst.shape
color = (0,0,255)

for y in range(0,height):
    for x in range(0,width):
        if dst.item(y,x) > 0.01 * dst.max():
            cv2.circle(img_c,(x,y),3,color,cv2.FILLED,cv2.LINE_AA)

plt.imshow(image)
plt.show()

cv2.imshow('Corner',img_c)
cv2.waitKey(0)
cv2.destroyAllWindows()
