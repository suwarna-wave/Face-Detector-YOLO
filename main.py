
import cv2
from ultralytics import YOLO
from datetime import datetime


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


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

    focus_mode = False
    print("Face Tracking Started")
    print("Keys: [q] quit, [m] toggle focus mode, [s] save snapshot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Run YOLO detection
        results = model(frame, verbose=False)
        result = results[0]
        boxes = result.boxes

        face_count = 0
        lock_face = None
        lock_area = 0.0

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                face_count += 1
                area = (x2 - x1) * (y2 - y1)
                if area > lock_area:
                    lock_area = area
                    lock_face = (x1, y1, x2, y2, conf)

        if focus_mode and face_count > 0:
            # Keep faces sharp while softening background for a spotlight effect.
            blurred = cv2.GaussianBlur(frame, (31, 31), 0)
            display_frame = blurred
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1i = int(clamp(x1, 0, frame.shape[1] - 1))
                y1i = int(clamp(y1, 0, frame.shape[0] - 1))
                x2i = int(clamp(x2, 1, frame.shape[1]))
                y2i = int(clamp(y2, 1, frame.shape[0]))
                if x2i > x1i and y2i > y1i:
                    display_frame[y1i:y2i, x1i:x2i] = frame[y1i:y2i, x1i:x2i]
        else:
            display_frame = frame.copy()

        h, w = display_frame.shape[:2]
        center = (w // 2, h // 2)

        # Center guide to help frame a selfie.
        cv2.circle(display_frame, center, 70, (255, 255, 0), 1)

        if lock_face is not None:
            x1, y1, x2, y2, conf = lock_face
            x1i = int(clamp(x1, 0, w - 1))
            y1i = int(clamp(y1, 0, h - 1))
            x2i = int(clamp(x2, 1, w))
            y2i = int(clamp(y2, 1, h))

            # Lock-on style rectangle for the dominant face.
            cv2.rectangle(display_frame, (x1i, y1i), (x2i, y2i), (0, 255, 255), 2)

            fx = (x1i + x2i) // 2
            fy = (y1i + y2i) // 2
            cv2.line(display_frame, (fx - 12, fy), (fx + 12, fy), (0, 255, 255), 1)
            cv2.line(display_frame, (fx, fy - 12), (fx, fy + 12), (0, 255, 255), 1)

            conf_text = f"LOCK {conf * 100:.1f}%"
            cv2.putText(display_frame, conf_text, (x1i, max(20, y1i - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            near_center = abs(fx - center[0]) < 80 and abs(fy - center[1]) < 80
            large_enough = (x2i - x1i) * (y2i - y1i) > 14000
            if near_center and large_enough:
                cv2.putText(display_frame, "SELFIE READY", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

        top_text = f"Faces: {face_count}"
        mode_text = f"Mode: {'Focus' if focus_mode else 'Normal'}"
        cv2.putText(display_frame, top_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, mode_text, (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(display_frame, "[q] Quit  [m] Focus  [s] Snapshot", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        cv2.imshow('YOLO Face Tracking', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            focus_mode = not focus_mode
        if key == ord('s'):
            filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Saved {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Face Tracking Stopped")


if __name__ == "__main__":
    main()
