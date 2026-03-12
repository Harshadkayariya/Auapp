import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# 1. स्टाइल AI मॉडल लोड करें (YOLOv8)
model = YOLO('yolov8n.pt')

def generate_frames():
    camera = cv2.VideoCapture(0) # वेबकैम शुरू करें
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 2. AI से चीजों को पहचानवाएं
            results = model(frame, stream=True)
            
            for r in results:
                annotated_frame = r.plot() # फोटो पर बॉक्स बनाएं

            # 3. फ्रेम को वेब पर दिखाने के लिए एनकोड करें
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # एक बेसिक HTML पेज जो कैमरा फीड दिखाएगा
    return """
    <html>
      <head>
        <title>BS 4th Year Project - AI Vision</title>
        <style>
            body { background: #121212; color: white; text-align: center; font-family: sans-serif; }
            .container { margin-top: 50px; }
            img { border: 5px solid #00ff88; border-radius: 15px; width: 80%; max-width: 800px; }
        </style>
      </head>
      <body>
        <div class="container">
            <h1>AI Real-Time Object Detection System</h1>
            <p>Inspired by Matt Deitke Research | BS Final Year Project</p>
            <img src="/video_feed">
        </div>
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
