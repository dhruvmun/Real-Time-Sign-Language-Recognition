import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
x, y, w, h = 400,100,200,200
i = 0

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))
    res = cv2.bitwise_and(img, img, mask=mask2)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    median = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel_square = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(median, kernel_square, iterations=2)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)
    ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)
    thresh = thresh[y:y + h, x:x + w]

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
    cv2.imshow("Frame", img)
    cv2.imshow("thresh", thresh)

    i+=1
    FILE = 'c_'+str(i)+'.png'
    cv2.imwrite(FILE, thresh)

    if i==100:
        break

    k = cv2.waitKey(10)
    if k == 27:
        break