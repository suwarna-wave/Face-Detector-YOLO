
import cv2
from ultralytics import YOLO


def main():
    model = YOLO('yolov12n-face.pt')
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return  
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("face Tracking Started - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run YOLO detection
        results = model(frame, verbose=True)
        
        # Draw detections on frame
        annotated_frame = results[0].plot()
        
        # Display FPS
        fps_text = f"Press 'q' to quit"
        cv2.putText(annotated_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('YOLO Hand Tracking', annotated_frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Hand Tracking Stopped")


if __name__ == "__main__":
    main()
