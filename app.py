
import cv2
from ultralytics import YOLO

try:
    model = YOLO('yolov8n.pt') 
    print("AI Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("--- AI Detection System Running ---")
print("Press 'q' to stop the program")

while True:
    success, frame = cap.read()
    
    if not success:
        print("Failed to grab frame")
        break
      
    results = model(frame, stream=True)
  
    for r in results:
        annotated_frame = r.plot() 
        
        cv2.imshow("Final Year Project: AI Object Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
