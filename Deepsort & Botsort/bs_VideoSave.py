from ultralytics import YOLO
import cv2

model=YOLO("yolov8s.pt")
input_video_path="sample2.mp4"

# cap=cv2.VideoCapture(0)
cap = cv2.VideoCapture(input_video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_video_path = "result.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

track_ids = -1

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(frame, persist=True)
    boxes = results[0].boxes.xyxy
    confs = results[0].boxes.conf
    class_ids = results[0].boxes.cls
    track_ids = results[0].boxes.id

    if track_ids is None:
      continue

    names=results[0].names 
    detections = []
    for i in range(len(class_ids)):
        class_id = int(class_ids[i])
        class_name = names[class_id]
        trk_id = int(track_ids[i])
        detections.append((boxes[i], confs[i], class_name,trk_id))

    if detections:
        for box, conf, class_name, trk_id in detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_name} {trk_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow('img', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
