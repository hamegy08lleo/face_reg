import sqlite3
import cv2
import numpy as np

def face_reg(): 
    

    face_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        raise IOError('Unable to load the face cascade classifier xml file')

    cap = cv2.VideoCapture(0)

    recognizer = cv2.face.LBPHFaceRecognizer_create()


    recognizer.read('model.yml')

    scaling_factor = 1
    

    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in face_rects:
            roi_gray = gray[y:y+h, x:x+w]
            id, confidence = recognizer.predict(roi_gray)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 172, 189), 3)
            
            connection = sqlite3.connect("Person.db")
            cursor = connection.cursor() 
            cursor.execute("SELECT name FROM Persons WHERE personID = (?)", (id, ))
            name = cursor.fetchone()[0]
            
            connection.close()
            
            text = f'{name}'
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 172, 189), 2)
        cv2.imshow('Face Detector', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
