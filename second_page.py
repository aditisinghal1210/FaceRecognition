from dl_code import IMG_SIZE
import cv2
import tensorflow
import numpy as np
import keras
from numpy import asarray


IMG_SIZE = IMG_SIZE
def start_recognition_exam():    
    our_model= keras.models.load_model('model.h5')

    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX
    #iniciate id counter
    id = 0
    names = ['None', 'Shraddha', 'Harshal', 'Aditi', 'Aishwarya'] 
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    
    while True:
        ret, img =cam.read()
        img = cv2.flip(img, 1) # Flip vertically
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # cvtColor() method is used to convert an image from one color space to another.
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            new_gray = gray[y:y+h,x:x+w]
            # img_data = cv2.re
            img_data = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            confidence = our_model.predict(img_data.reshape(1,IMG_SIZE,IMG_SIZE,1))
            id = np.argmax(confidence[0])
            print(id)
            name = names[id]
            
            
            cv2.putText(
                        img, 
                        str(name), 
                        (x+5,y-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                    )
        
        
        cv2.imshow('camera',img) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
    
    # Release video capture and destroy all windows
    cam.release()
    cv2.destroyAllWindows()
    
start_recognition_exam()