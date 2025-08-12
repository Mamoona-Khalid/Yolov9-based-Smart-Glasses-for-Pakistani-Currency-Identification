import cv2
import torch
import ultralytics
from ultralytics import YOLO
import argparse


def load_model(weights_path):
    # Load the YOLOv9 model
    model = YOLO('weights/best_20.pt')
    return model


def predict_and_display(frame, model):
    # Run inference on the frame
    results = model([frame], stream=True)

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

    return frame


def main():
    parser = argparse.ArgumentParser(description='YOLOv9 Real-time Object Detection')
    parser.add_argument('--weights', type=str, help='Path to the YOLOv9 weights file', required=True)
    args = parser.parse_args()

    # Load YOLOv9 model
    model = load_model(args.weights)

    option = input("Select input type (webcam/video/image): ").strip().lower()

    if option == 'webcam':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
    elif option == 'video':
        video_path = input("Enter the path to the video file: ").strip()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
    elif option == 'image':
        image_path = input("Enter the path to the image file: ").strip()
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image file: {image_path}")
            return
        result_frame = predict_and_display(frame, model)
        cv2.imshow('YOLOv9 Object Detection', result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        print("Invalid option. Please select 'webcam', 'video', or 'image'.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = predict_and_display(frame, model)
        cv2.imshow('YOLOv9 Object Detection', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
