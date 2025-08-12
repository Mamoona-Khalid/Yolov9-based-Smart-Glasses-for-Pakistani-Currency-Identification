import cv2
import torch
import os
import argparse
import ultralytics
from ultralytics import YOLO
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()


def load_model(weights_path):
    # Load the YOLO model
    model = YOLO(weights_path)
    return model


def predict_and_display(frame, model, detected_objects):
    # Run inference on the frame
    results = model([frame], stream=True)

    # Set to track objects detected in the current frame
    current_detections = set()

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        class_names = result.names  # Class names from the model

        # Draw bounding boxes and labels on the frame
        for i, box in enumerate(boxes):
            if box.conf >= 0.3:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = class_names[int(box.cls)]
                confidence = box.conf.item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)

                # Add detected object to the current_detections set
                current_detections.add(label)

                # Speak out the label if it's a new detection in this frame
                if label not in detected_objects:
                    engine.say(f'Detected {label} Rupees Note')
                    engine.runAndWait()
                    detected_objects.add(label)

    return frame, current_detections


def save_image(frame, count):
    if not os.path.exists('results'):
        os.makedirs('results')
    filename = f'results/frame_{count}.jpg'
    cv2.imwrite(filename, frame)


def main():
    global engine
    parser = argparse.ArgumentParser(description='YOLO Real-time Object Detection')
    parser.add_argument('--weights', type=str, help='Path to the YOLO weights file', required=True)
    args = parser.parse_args()

    # Load YOLO model
    model = load_model(args.weights)

    option = input("Select input type (webcam/video/image): ").strip().lower()
    frame_count = 1
    frame_counter = 0
    DETECTION_RESET_INTERVAL = 30  # Reset detected_objects list after every 30 frames
    detected_objects = set()  # Set to track detected objects

    if option == 'webcam':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        out = cv2.VideoWriter('results/webcam_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result_frame, current_detections = predict_and_display(frame, model, detected_objects)
            cv2.imshow('YOLO Object Detection', result_frame)
            out.write(result_frame)

            # Add current detections to detected_objects set
            detected_objects.update(current_detections)

            # Increment frame counter
            frame_counter += 1

            # Reset detected_objects list after every DETECTION_RESET_INTERVAL frames
            if frame_counter == DETECTION_RESET_INTERVAL:
                detected_objects.clear()
                frame_counter = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
    elif option == 'video':
        video_path = input("Enter the path to the video file: ").strip()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
        out = cv2.VideoWriter('results/video_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0,
                              (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result_frame, current_detections = predict_and_display(frame, model, detected_objects)
            cv2.imshow('YOLO Object Detection', result_frame)
            out.write(result_frame)

            # Add current detections to detected_objects set
            detected_objects.update(current_detections)

            # Increment frame counter
            frame_counter += 1

            # Reset detected_objects list after every DETECTION_RESET_INTERVAL frames
            if frame_counter == DETECTION_RESET_INTERVAL:
                detected_objects.clear()
                frame_counter = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
    elif option == 'image':
        image_path = input("Enter the path to the image file: ").strip()
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image file: {image_path}")
            return
        result_frame, current_detections = predict_and_display(frame, model, detected_objects)
        cv2.imshow('YOLO Object Detection', result_frame)
        save_image(result_frame, frame_count)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Invalid option. Please select 'webcam', 'video', or 'image'.")
        return

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
