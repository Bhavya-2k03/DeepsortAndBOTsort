import cv2
import numpy as np
import sys
import glob
import time
import torch

class YoloDetector():
    def __init__(self, modelName):
        self.model=self.load_model(modelName)
        self.classes=self.model.names
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        print("using device: ", self.device)
    
    def load_model(self, modelName):
        if modelName:
            model=torch.hub.load('ultralytics/yolov5', 'custom', path=modelName, force_reload=True)
        else:
            model=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    
    def score_frame(self, frame):
        self.model.to(self.device)
        downscaleFactor=2
        width =int(frame.shape[1]/downscaleFactor)
        height= int(frame.shape[0]/downscaleFactor)
        frame=cv2.resize(frame,(width,height))
        
        results=self.model(frame)
        labels, cord= results.xyxyn[0][:,-1], results.xyxyn[0][:,:-1]
        return labels,cord 
    
    def class_to_label(self,x):
        return self.classes[int(x)]

    def plot_boxes(self,results, frame, height, width, confidence=0.4):
        labels, cord=results
        detections=[]
        n=len(labels)
        xShape, yShape=width, height

        for i in range(n):
            row=cord[i]

            if row[4]>=confidence:
                x1 , y1, x2, y2=int(row[0]*xShape), int(row[1]* yShape), int(row[2]*xShape), int(row[3]*yShape)

                myclass='person'
                if self.class_to_label(labels[i])==myclass:
                    xCenter=x1+(x2-x1)
                    yCenter=y1+((y2-y1)/2)
                    tlwh=np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype=np.float32)
                    confidence=float(row[4].item())
                    feature=myclass
                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), myclass))

        return frame, detections
    
    
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

modelName=None
detector=YoloDetector(modelName)

import os
os.environ["KMP_DUPLICATE_LIP_OK"]="TRUE"

from deep_sort_realtime.deepsort_tracker import DeepSort

objectTracker=DeepSort(max_age=5)

while cap.isOpened():
    success, img =cap.read()
    start=time.perf_counter()
    results=detector.score_frame(img)
    img,detections=detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5)
    tracks=objectTracker.update_tracks(detections, frame=img)

    for track in tracks:
        if not track.is_confirmed():
            continue
        trackID=track.track_id
        ltrb=track.to_ltrb()
        bbox=ltrb

        cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]),int(bbox[3])),(0,0,255),2)
        cv2.putText(img, "ID: "+str(trackID), (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0,255,0),2)

    end=time.perf_counter()
    totalTime=end-start
    fps=1/totalTime

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),2)
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF==27:
        break

cap.release()

cv2.destroyAllWindows()



