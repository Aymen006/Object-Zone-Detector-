# Count People in Zone

## Overview

A video analysis tool that counts and highlights objects in specific zones of a video. Each zone is marked in a different color for easy visualization and counting.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./setup.sh
```

## Script Arguments

- `--source_weights_path` (optional): The path to the YOLO model's weights file. Defaults to `"yolov8x.pt"` if not specified.

- `--zone_configuration_path`: Path to the JSON file containing zone configurations. This file defines the polygonal areas in the video where objects will be counted.

- `--source_video_path`: The path to the source video file that will be analyzed.

- `--target_video_path` (optional): The path to save the output video with annotations. If not provided, the processed video will be displayed in real-time.

- `--confidence_threshold` (optional): Sets the confidence threshold for object detection to filter detections. Default is `0.3`.

- `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold for the model. Default is `0.7`.

## Configuration

**Zone Files:**
- `horizontal-zone-config.json` - Horizontal zones
- `multi-zone-config.json` - Multiple custom zones
- `quarters-zone-config.json` - Four equal quarters
- `vertical-zone-config.json` - Vertical zones

## Usage

```bash
python ultralytics_example.py \
    --zone_configuration_path data/multi-zone-config.json \
    --source_video_path data/market-square.mp4 \
    --confidence_threshold 0.3 \
    --iou_threshold 0.5
```
