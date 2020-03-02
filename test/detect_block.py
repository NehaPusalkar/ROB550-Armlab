import cv2
import numpy as np

video_frame = cv2.cvtColor(cv2.imread("./data/rgb_image.png",cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2HSV)
depth_frame = cv2.imread("./data/raw_depth.png",0).astype(np.uint8)
#depth_frame = cv2.medianBlur(depth_frame, 3)
ret, th1 = cv2.threshold(depth_frame, 198, 255, cv2.THRESH_BINARY)
binary = cv2.Canny(th1, 10, 90)
image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
canvas = np.zeros((480,640,3)).astype(np.uint8)
canvas[...,0] = depth_frame
canvas[...,1] = depth_frame
canvas[...,2] = depth_frame
block_center = []
if(len(contours)!=0):
    for contour in contours:
        #cv2.drawContours(canvas, contours, -1, (255,0,255), 3)
        perimeter = cv2.arcLength(contour,True)
        epsilon = 0.1 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        overlap = False
        if(len(approx)==4):
            cur_center = np.sum(approx, axis=0)/4
            for center in block_center:
                if(np.sum((center - cur_center)**2)<100):
                    overlap = True
                    break
            if(not overlap and cv2.contourArea(approx) > 200):
                block_center.append(cur_center)
                cv2.drawContours(canvas, [approx], -1, (255,0,255), 2)
affine_matrix = np.array([[  9.29346844e-1,  -3.23494980e-03,   11.2347976],
                            [  1.48233033e-03,   8.74361534e-01,   31.8750435]])

block_center = np.array(block_center).reshape(4,2).T
block_center_h = np.concatenate((block_center, np.array([[1,1,1,1]])), axis=0)
block_center_in_rgb = np.dot(affine_matrix, block_center_h).T
for center in block_center_in_rgb:
    x=int(center[0])
    y=int(center[1])
    print(video_frame[y][x][0])

cv2.imshow("image",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()