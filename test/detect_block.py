import cv2
import numpy as np
import math

colors = {'red': [0,8,200,255,75,115], 'green':[40,70,100,255,45,85], 'blue':[100,130,100,200,60,90],
         'orange': [8,18,200,255,90,150], 'purple':[140,170,100,170,45,75], 'pink': [170,180,180,250,125,165],
         'yellow' : [20,30,200,255,100,255], 'black': [5,30,100,210,20,60]}

colors_center = {'red': [4,227.5,95], 'green' : [55,177.5,65], 'blue' : [115,150,75],
                 'orange' : [13,227.5,120], 'purple' :[155,135,60], 'pink': [175,215,145],
                 'yellow' : [25,227.5,177.5], 'black': [17.5,155,40]}

frame =cv2.imread("./data/rgb_image.png",cv2.IMREAD_UNCHANGED)

video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
video_frame = cv2.resize(video_frame, (640, 480))
depth_frame = cv2.imread("./data/raw_depth.png",0).astype(np.uint8)
depth_frame = cv2.medianBlur(depth_frame, 3)
th1 = cv2.inRange(depth_frame, 158, 200)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
th1 = cv2.erode(th1, kernel, iterations=2)
th1 = cv2.dilate(th1, kernel, iterations=1)
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
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cur_center = np.sum(approx, axis=0)/len(approx)
            for center in block_center:
                if(np.sum((center - cur_center)**2)<50):
                    overlap = True
                    break
            if(not overlap and cv2.contourArea(approx) > 200):
                block_center.append(cur_center)
                cv2.drawContours(canvas, [box], -1, (255,0,255), 2)

affine_matrix = np.array([[ 9.29346844e-1,  -3.23494980e-03,   11.2347976],
                        [  1.48233033e-03,   8.74361534e-01,   31.8750435]])

num = len(block_center)
block_center = np.array(block_center).reshape(num,2).T
block_center_h = np.concatenate((block_center, np.ones([1, num])), axis=0)
block_center_in_rgb = np.dot(affine_matrix, block_center_h).T
font = cv2.FONT_HERSHEY_SIMPLEX
for center in block_center_in_rgb:
    min_loss = 2000
    x=int(center[0])
    y=int(center[1])
    h = int(video_frame[y][x][0])
    s = int(video_frame[y][x][1])
    v = int(video_frame[y][x][2])
    for key in colors:
        # low_h = colors[key][0]
        # high_h = colors[key][1]
        # low_s = colors[key][2]
        # high_s = colors[key][3]
        # low_v =  colors[key][4]
        # high_v = colors[key][5]
        h_diff = abs(h - colors_center[key][0])
        s_diff = abs(s - colors_center[key][1])
        v_diff = abs(v - colors_center[key][2])
        
        loss = math.sqrt(0.9 * (h_diff**2) + 0.05 * (s_diff**2) + 0.05 * (v_diff**2)) 
      
        if (loss < min_loss):
            min_loss = loss
            color_detected = key
    #output_h = "H:{}".format(int(video_frame[y][x][0]*2))
    output_h = "{}".format(color_detected)
    cv2.putText(canvas, output_h, (x, y), font, 0.5, (0, 0, 0))

cv2.imshow("image",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()