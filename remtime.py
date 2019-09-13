import cv2
import numpy as np
import os
import tensorflow as tf


my_model = tf.keras.models.load_model('sign_language_self.h5')
'''
# Dictionary to convert numerical classes to alphabet
idx2alph = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',
            15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',22:'w',23:'x',24:'y',25:'z'}


# ====================== Live loop ======================
# =======================================================

video_capture = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, img = video_capture.read()
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

    # Draw rectangle
    x = 400
    y = 100
    w = 200
    h = 200
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)

    
    # Crop + process captured frame
    thresh = thresh[y:y+h, x:x+w]
    x = cv2.resize(thresh, (64, 64))
    x = np.array(x, dtype=float)
    x = x.reshape(1, 64, 64, 1)

    
    # Make prediction
    my_predict = my_model.predict(x, batch_size=1, verbose=0)
    top_prd = np.argmax(my_predict, axis=1)
    

    # Only display predictions with probabilities greater than 0.5
    if np.max(my_predict) >= 0.50:

        prediction_result = idx2alph[top_prd[0]]

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, prediction_result, (10,100), font, 4,(255,255,255),2)
       
    # Display the resulting frame
    cv2.imshow('Video', img)
    cv2.imshow('bw', thresh)

    # Press 'q' to exit live loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


# Release the capture
video_capture.release()
cv2.destroyAllWindows()
'''