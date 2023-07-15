# Object Tracker with Kalman Filter Using Yolov8 As Detector

*This project implements kalman filter tracking on detected objects using Yolov8. This repo is particularly created to do such tasks using roboflow object detection models. Currently only available for single object tracking. This demo is implemented using soccer-ball detection model from roboflow.*

## To run

```
pip install -r requirements.txt
python object_track.py  --roboflow_api $your_roboflow_api --project_code $your_project_code --path $video_path --overlap $30 --confidence $40 --max_tracks $1
```



https://github.com/irfanbykara/Multi-Object-Tracking-with-Kalman-Filter/assets/63783207/a5dfeb17-f2ad-4fe1-a955-f97a16e9abee



# References -
Forked From: https://github.com/mabhisharma/Multi-Object-Tracking-with-Kalman-Filter
http://studentdavestutorials.weebly.com/multi-bugobject-tracking.html
