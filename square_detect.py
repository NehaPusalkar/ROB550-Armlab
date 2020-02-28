import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

image = cv2.imread("./data/rgb_image.png")
cv2.imshow("image",image)
# gaussian blur
blur = cv2.GaussianBlur(image,(5,5),5)
# convert to HSV

hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

# set limits on values for that color
lower_val =np.array([0, 120, 70])
upper_val = np.array([5,255,255])

lower_red = np.array([170,120,70])
upper_red = np.array([180,255,255])

mask1 = cv2.inRange(hsv,lower_val,upper_val)
mask2 = cv2.inRange(hsv,lower_red,upper_red)

mask = mask1 + mask2

# mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
# mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
#
# mask_final = cv2.bitwise_not(mask)
#
# res1 = cv2.bitwise_and(image,image, mask = mask_final)
output_img = image.copy()
output_img[np.where(mask==0)] = 0

output_hsv = hsv.copy()
output_hsv[np.where(mask==0)] = 0

cv2.imshow("image",output_img)
cv2.imshow("hsv_image",output_hsv)






#
#cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
