from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera
import cv2,os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import Counter
import sqlite3
from datetime import date,datetime


tf.keras.backend.clear_session()

app = Flask(__name__)

video_stream = VideoCamera()

VIDEO_TIME=6
STUDENT_ID = None
PERIOD = None

class_labels = {0:'ThuThao', 1:'ThuDiem', 2:'PhamNguyen',3:'ThuHien',4:'HuynhDuc'}

model = load_model('models/Face_trained_model_11_41_02_.h5', compile=False) #tốt 


@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
        start=int(datetime.now().strftime("%S"))
        global model, class_labels, STUDENT_ID
        labels = []
        while int(datetime.now().strftime("%S"))-start!=VIDEO_TIME:
                frame = camera.get_frame()
                # plt.imshow(frame)
                nparr = np.fromstring(frame, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faceDetect=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                # cv2.imwrite('img'+str(np.random.randint(100*1000))+'.png', img)
                faces=faceDetect.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(100,100),flags=cv2.CASCADE_SCALE_IMAGE)
                # res = None
                for(x,y,w,h) in faces:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                        res = cv2.resize(gray[y:y+h,x:x+w], (100, 100), interpolation = cv2.INTER_AREA)
                prediction =model.predict(res.reshape(-1, 100, 100, 1))
                labels.append(np.argmax(model.predict(res.reshape(-1, 100, 100, 1).astype(np.float32))))
                print(prediction)
                print(labels)
                predict_float = tf.cast(prediction, tf.float32)
                probabilities = tf.nn.softmax(predict_float).numpy()
                for i, prob in enumerate(probabilities[0]):
                    print( f'Class {i}: {prob:.4f}',)
                yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')
        print('*'*11 , ' '*5,max(set(labels), key = labels.count),  class_labels[max(set(labels), key = labels.count)] ,'--', ' '*5, '*'*11)
        STUDENT_ID = max(set(labels), key = labels.count)


                

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream), mimetype = 'multipart/x-mixed-replace;boundary=frame')


@app.route('/submit')
def submit():
    # details = get_details()
    if details:
        return render_template('details.html', data=details)
    else:
        return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=True)