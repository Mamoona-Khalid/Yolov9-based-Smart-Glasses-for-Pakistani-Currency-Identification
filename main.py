import ultralytics

from ultralytics import YOLO

# Load a model
model = YOLO("weights/best_30.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["Testing_data/a.jpg", "Testing_data/b.jpg"], stream=True)  # return a generator of Results objects
i=1
# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    print('boxes', boxes)
    masks = result.masks  # Masks object for segmentation masks outputs
    print('masks', masks)
    keypoints = result.keypoints  # Keypoints object for pose outputs
    print('key points', keypoints)
    probs = result.probs  # Probs object for classification outputs
    print('prob', probs)
    obb = result.obb  # Oriented boxes object for OBB outputs
    print('obb', obb)
    result.show()  # display to screen
    result.save(filename=f"result{i}.jpg")  # save to disk
    i = i+1