import cv2
import numpy as np
cap=cv2.VideoCapture('video.mp4')


min_width_rectangle= 80 #min width of the rectangle
min_height_rectangle= 80  #min height of the rectangle
count_line_pos=500
#Initialize subtractor function
algo=cv2.createBackgroundSubtractorMOG2()

# center red dot used while counting the vehicles within the rectangle
def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

#counter to keep track of total number of vehicles present in video
detect_counter=[]
offset=6 #allowable error between pixel
#counter to enumerate each vehicle in the video individually
counter=0



while True:
    ret,frame1=cap.read()
    grey= cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur= cv2.GaussianBlur(grey,(3,3),5)
    #applying on all frames
    img_sub= algo.apply(blur)
    dilate=cv2.dilate(img_sub,np.ones((3,3)))
    #morphological operation
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilate_struct=cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    dilate_struct=cv2.morphologyEx(dilate_struct,cv2.MORPH_CLOSE,kernel)
    count_vehicle,h= cv2.findContours(dilate_struct,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1,(25,count_line_pos),(1200,count_line_pos),(255,127,0),3)

    #surrounding vehicles with rectangles
    for (i,c) in enumerate(count_vehicle):
        (x,y,w,h)=cv2.boundingRect(c)
        validate_counter=(w>=min_width_rectangle) and (h>=min_height_rectangle)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"VEHICLE NUMBER :"+str(counter),(x,y-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,244,0),2)

        #placing the red dot in center for counting
        center=center_handle(x,y,w,h)
        detect_counter.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

        #loop to append total vehicle count
        for (x,y) in detect_counter:
            if y<(count_line_pos+offset) and y>(count_line_pos-offset):
                counter+=1
            cv2.line(frame1,(25,count_line_pos),(1200,count_line_pos),(0,127,255),3)
            detect_counter.remove((x,y))
            print("Vehicle Counter :"+str(counter))

    cv2.putText(frame1,"VEHICLE COUNTER :"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

    #cv2.imshow('Detector',dilate_struct)
    
    cv2.imshow('Original Video',frame1)
    if cv2.waitKey(1)==13:
        break

cv2.destroyAllWindows()
cap.release()




