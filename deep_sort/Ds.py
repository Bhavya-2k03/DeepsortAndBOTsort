import cv2
import numpy as np
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# Load the neural network for feature extraction
model_filename = 'deep_sort/resources/networks/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

# DeepSORT parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# Initialize tracker
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection (replace this part with your preferred object detection method)
    # For simplicity, using dummy detections
    # bbox is a list of bounding boxes in the format [x, y, width, height]
    bbox = [[100, 100, 50, 50], [200, 200, 50, 50]]
    confidences = [0.9, 0.8]  # Confidence scores
    features = encoder(frame, bbox)

    detections = [Detection(bbox[i], confidences[i], features[i]) for i in range(len(bbox))]

    # Run non-maxima suppression
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.3, nms_max_overlap)

    detections = [detections[i[0]] for i in indices]

    # Update tracker
    tracker.predict()
    tracker.update(detections)

    # Draw bounding boxes and IDs
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlwh()
        track_id = track.track_id
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
                      (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('DeepSORT Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
