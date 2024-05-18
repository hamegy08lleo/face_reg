import sqlite3
import cv2
import os
import tkinter as tk
from tkinter import simpledialog, Button
import time
from face_reg import face_reg
from train import train


dataset_dir = 'dataset'

os.makedirs(dataset_dir, exist_ok=True)

scaling_factor = 1


def capture_images():
    cap = cv2.VideoCapture(0)
    
    start_time = time.time()
    
    root = tk.Tk()
    root.withdraw()
    name = simpledialog.askstring("Input", "Enter the name of the person",
                                   parent=root)
    if not name: 
        return
    
    
    connection = sqlite3.connect('Person.db')
    cursor = connection.cursor()
    cursor.execute("PRAGMA table_info('Persons')")
    table_exists = cursor.fetchall()

    if not table_exists:
        cursor.execute(
            '''
            CREATE TABLE Persons(
                personID integer primary key autoincrement,
                name text
            )
            '''
        )
        connection.commit()
    
    print(name)
    
    cursor.execute(
        "INSERT INTO Persons(name) values(?)", (name, )
    )
    connection.commit()
    noLoop = 100
    print(name)
    while noLoop > 0:
        _, frame = cap.read()
        
        cv2.imshow('Webcam', frame)

        cursor.execute(
            '''
                SELECT MAX(personID) FROM Persons 
                WHERE name = ?
            ''', (name, )
        )
        personID = cursor.fetchone()[0]
        cursor.execute( 
                       '''
                       SELECT name from Persons 
                       WHERE personID = ?
                       ''', (personID, )
                       )
        
        if time.time() - start_time > 0.1:
            img_name = os.path.join(dataset_dir, f'{personID}.{int(time.time())}.png')
            cv2.imwrite(img_name, frame)
            noLoop -= 1
            start_time = time.time()
            
        c = cv2.waitKey(1)
        if c == 27:
            break
    
    connection.close()
        

    cap.release()

    cv2.destroyAllWindows()
    train()

    
def quit(): 
    root.quit()
    root.destroy()
root = tk.Tk()

Button(root, text="Start Capturing", command=capture_images).pack()
Button(root, text="Start Face Recognition", command=face_reg).pack()
root.protocol("WM_DELETE_WINDOW", quit)
root.mainloop()
