import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
import pyautogui

last_time = time.time()

##for i in range(4)[::-1]:
##    print(i+1)
##    i-=1
##    time.sleep(1)
##    
##PressKey(W)
##time.sleep(2)
##ReleaseKey(W)

flag = True

def draw_lines(img, lines):
    try:
        for line in lines:
            coord = line[0]
            cv2.line(img, (coord[0],coord[1]), (coord[2],coord[3]), [255,255,255], 3)
    except:
        pass

def detect_vehicles(processed_img, original_image):
    ret,thresh = cv2.threshold(processed_img, 100, 255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
#square perimeter 84
    cv2.imshow('thrsh',processed_img)
    for cnt in contours:
        
        x,y,w,h = cv2.boundingRect(cnt)
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        perimeter = cv2.arcLength(cnt,True)
##        if(w > 45 and w < 50):
        cv2.drawContours(original_image, [cnt], -1, (255,0,0), 2)
        text = "none"
        print("w, h",w,h)
        print("approx ",len(approx))
        print("perimeter",perimeter)
        if(len(approx))==4 and (perimeter > 300 and perimeter < 500):
##            cv2.drawContours(original_image, [cnt], -1, (255,0,0), 2)
            text = "square"
            #print("cnt for square",cnt)
            #print("square",perimeter)
        elif(len(approx))==15:
            text = "circle"
            #print("circle")
        elif perimeter > 1000 and perimeter < 1500:
##            cv2.drawContours(original_image, [cnt], -1, (255,0,0), 2)
            #print("none",perimeter)
            pass
##        cv2.putText(original_image, text, (cnt[0][0][0], cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX,
##        0.5, (255, 255, 255), 2)
##    print(contours)
    return original_image

def roi(img, vertices):
    if(flag):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)

    return masked

def process_img(original_image):
    vertices = np.array([[0,830], [0,350], [480,350], [480,830]])
##    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_img = roi(original_image, [vertices])
    
    
##    cv2.imshow('processed_img_before', processed_img)
    #print(processed_img)
    #print("size ",process)
    #print(np.any([47,232,255]))
    #np.where(processed_img==[47,232,255], [166,163,162], processed_img)

##    yellow_removed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
##    print(np.where((yellow_removed_img == [32,217,255]).all(axis = 2)))
##    yellow_removed_img[np.where((yellow_removed_img == [32,217,255]).all(axis = 2))] = [0,0,255]#[166,163,162]
##    cv2.imshow('yellow_removed_img', yellow_removed_img)
##    #processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
##    cv2.imshow('processed_img', processed_img)
##    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
##
##    mask = cv2.inRange(hsv, (21, 39, 64), (40,255,255))
##    imask = mask>0
##    yellow = np.zeros_like(processed_img, np.uint8)
##    yellow[imask] = processed_img[imask]
    #cv2.imshow('yellow', processed_img)


    processed_img = cv2.Canny(processed_img, 200, 600)
    processed_img = cv2.dilate(processed_img, (3,3), iterations=4)
##    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
    detect_vehicles(processed_img, original_image)
    
##    
##
##    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 200, 10)
##    
##    draw_lines(processed_img, lines)
    
    return processed_img

original_window = "Captured Screen"
canny_window = "Edge Detected Screen"

##cv2.namedWindow( original_window, cv2.WINDOW_AUTOSIZE )
##cv2.namedWindow( canny_window, cv2.WINDOW_AUTOSIZE )

while(True):
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 480, 830)))
    new_screen = process_img(screen)
##    printscreen_numpy = np.array(printscreen_pil.getdata(), dtype='uint8')\
##    .reshape((printscreen_pil.size[1], printscreen_pil.size[0],3))
    cv2.imshow(original_window, cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    
##    cv2.imshow(canny_window, new_screen)
    print('this frame took ', time.time() - last_time)
    last_time = time.time()
    if(cv2.waitKey(25) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        break
