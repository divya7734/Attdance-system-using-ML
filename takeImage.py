import csv
import os, cv2
import numpy as np
import pandas as pd
import datetime
import time

# ✅ FIXED: correct function name and parameters
def TakeImage(l1, l2, haarcasecade_path, trainimage_path, message, err_screen, text_to_speech):
    if (l1 == "") and (l2 == ""):
        t = 'Please Enter your Enrollment Number and Name.'
        text_to_speech(t)
        message.configure(text=t)
        return
    elif l1 == "":
        t = 'Please Enter your Enrollment Number.'
        text_to_speech(t)
        message.configure(text=t)
        return
    elif l2 == "":
        t = 'Please Enter your Name.'
        text_to_speech(t)
        message.configure(text=t)
        return

    try:
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(haarcasecade_path)
        Enrollment = l1
        Name = l2
        sampleNum = 0

        directory = Enrollment + "_" + Name
        path = os.path.join(trainimage_path, directory)
        if not os.path.exists(path):
            os.mkdir(path)

        while True:
            ret, img = cam.read()
            if not ret:
                message.configure(text="Camera not detected!")
                text_to_speech("Camera not detected!")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite(
                    f"{path}/{Name}_{Enrollment}_{sampleNum}.jpg",
                    gray[y:y+h, x:x+w]
                )
                cv2.imshow("Capturing Images...", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            elif sampleNum >= 50:
                break

        cam.release()
        cv2.destroyAllWindows()

        # ✅ Append student info to CSV
        csv_path = "StudentDetails/studentdetails.csv"
        if not os.path.exists("StudentDetails"):
            os.mkdir("StudentDetails")

        with open(csv_path, "a+", newline="") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([Enrollment, Name])

        res = f"Images Saved for ER No: {Enrollment}, Name: {Name}"
        message.configure(text=res)
        text_to_speech(res)

    except FileExistsError:
        msg = "Student data already exists!"
        text_to_speech(msg)
        message.configure(text=msg)
