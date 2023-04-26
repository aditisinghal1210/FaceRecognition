import sys
sys.setrecursionlimit(5000)
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from second_page import start_recognition_exam
from third_page import pose

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def gen_frames(face_id):  
    count = 0
    while True:
        success, frame = camera.read()
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 7)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                count += 1
                cv2.imwrite(f"data/User.{face_id}.{count}.jpg", gray[y:y+h,x:x+w])
                cv2.imshow('image', frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if count >= 30:
                camera.release()
                cv2.destroyAllWindows()
                return redirect(url_for('idex2'))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    face_id = request.form['user_id']
    return Response(gen_frames(face_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/exam')
def start_recognition_exam():
    return Response(start_recognition_exam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/proctor')
def proctoring():
    return Response(pose(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
