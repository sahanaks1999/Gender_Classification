import cv2
import numpy as np
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)

cap.set(3, 480) #set width of the frame
cap.set(4, 640) #set height of the frame
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_list = ['Male', 'Female']
def video_detector(gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    while (cap.isOpened()):
      ret, image = cap.read()
           
      face_cascade = cv2.CascadeClassifier("/home/sahanaks/opencv/data/haarcascades/haarcascade_frontalface_alt.xml")
      if ret:
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          faces = face_cascade.detectMultiScale(gray, 1.1, 5)
          if(len(faces)>0):
             print("Found {} faces".format(str(len(faces))))
          for (x, y, w, h )in faces:
             cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
             #Get Face 
             face_img = image[y:y+h, h:h+w].copy()
             
             face_img = cv2.resize(face_img, (64,64))
             gender_preds = gender_net.predict(np.expand_dims(face_img, axis=0))
             print(gender_preds[0].argmax())
             gender = gender_list[gender_preds[0].argmax()]
             print("Gender : " + gender)
             
             overlay_text = "%s " % (gender)
             cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
             cv2.imshow('frame', image)  

      if cv2.waitKey(1) & 0xFF == ord('q'): 
             break
if __name__ == "__main__":
    gender_net = load_model('/home/sahanaks/gender2.model')
    video_detector(gender_net)
