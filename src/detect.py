import cv2
import numpy as np
from tensorflow.keras.models import load_model

def detect_objects(model, live_stream=True, video_path=None):
    if live_stream:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (128, 128))
        frame_normalized = frame_resized / 255.0
        frame_input = np.expand_dims(frame_normalized, axis=0)
        
        predictions = model.predict(frame_input)
        class_id = np.argmax(predictions[0])
        
        cv2.putText(frame, f"Detected Class ID: {class_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
