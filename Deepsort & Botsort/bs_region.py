from ultralytics import YOLO
import cv2
import datetime


model = YOLO("yolov8s.pt")

cap=cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

track_ids = -1

roi_top_left = (100, 100)
roi_bottom_right = (300, 300)

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

    names = results[0].names 

    detections = []
    for i in range(len(class_ids)):
        class_id = int(class_ids[i])
        class_name = names[class_id]
        trk_id = int(track_ids[i])

        detections.append((boxes[i], confs[i], class_name, trk_id))

    if detections:
        for box, conf, class_name, trk_id in detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_name} {trk_id}"
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if (roi_top_left[0] < center_x < roi_bottom_right[0] and roi_top_left[1] < center_y < roi_bottom_right[1]):
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Person with ID: {trk_id} entered the region at time: {current_time}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2)
        cv2.imshow('img', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
