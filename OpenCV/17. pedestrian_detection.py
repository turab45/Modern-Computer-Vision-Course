import numpy as np
import cv2

""" Padestring (Full body) detection"""

# create cascaded classifier with pre-learned weights
# For other objects, change the file here
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

cap = cv2.VideoCapture("videos/walking-in.mp4")

# get width and height of the frame
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("walking-out.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

while(True):
    ret, frame = cap.read()
    if not ret:
        print("No frame captured")

    #frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect face
    faces = face_cascade.detectMultiScale(gray)
    # plot results
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
    cv2.imshow('Padestrian detection', frame)
    # write video
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()