#!/bin/bash

export DISPLAY=:0


~/skip_sudo.sh
sudo chmod -R 777 /dev/ttyUSB0

#python3 get_keypoint_from_video.py --model=resnet --video='v4l2src io-mode=2 device=/dev/video1 ! video/x-raw, format=YUY2, width=1920, height=1080, framerate=60/1 !  nvvidconv ! video/x-raw(memory:NVMM), format=(string)I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink sync=false async=false drop=true'

#python3 get_keypoint_from_video.py --model=resnet --video='v4l2src io-mode=2 device=/dev/video0 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! nvjpegdec ! video/x-raw !  nvvidconv ! video/x-raw(memory:NVMM), format=(string)I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink sync=false async=false drop=true'



#python3 get_keypoint_from_video.py --model=densenet --video='v4l2src io-mode=2 device=/dev/video0 ! image/jpeg, width=1920, height=1080, framerate=30/1 ! nvjpegdec ! video/x-raw !  nvvidconv ! video/x-raw(memory:NVMM), format=(string)I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink sync=false async=false drop=true'


python3 get_keypoint_from_video.py --model=resnet --video='v4l2src io-mode=2 device=/dev/video0 ! video/x-raw, format=YUY2, width=1920, height=1080, framerate=60/1 !  nvvidconv ! video/x-raw(memory:NVMM), format=(string)I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink sync=false async=false drop=true'