# Standalone Color Camera Recording

## What This Is
Modified version of the slip detection capture script that records **color RGB images** instead of grayscale.

## Key Changes Made

### 1. Pixel Format (Line 179-180)
```python
# Changed from Mono8 to RGB8Packed for color
cam.MV_CC_SetEnumValueByString("PixelFormat", "RGB8Packed")
```

### 2. Image Reshaping (Line 237)
```python
# Reshape as RGB with 3 channels: (height, width, 3)
img = arr.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth, 3)).copy()
```

### 3. Exposure Time (Line 42)
```python
CAM_EXPOSURE_TIME = 10000.0  # Increased for better brightness
```

### 4. Xense Sensor Camera Disabled (Line 409-410)
```python
# Disable internal camera to avoid V4L2 conflict with HIKROBOT cameras
s1 = Sensor.create(INDEX_SENSOR_ID, use_camera=False)
s2 = Sensor.create(THUMB_SENSOR_ID, use_camera=False)
```

## Usage
Run with Xense Python environment:
```bash
cd ~/grasp_dataset/standalone/
/usr/share/bin/xensegui/bin/python3 slip_detection_capture_color.py --object grape
```

## Prerequisites
1. LEAP Hand ROS node must be running:
   ```bash
   roslaunch leap_hand leap.launch urdf_file:=leap_right.urdf
   ```
2. Connect Xense tactile sensors (index + thumb)
3. Connect HIKROBOT cameras (cam1, cam2)

## Result
Records color images matching what you see in the MVS viewer, instead of black and white grayscale images.

## Files
- `slip_detection_capture_color.py` - Main capture script with color support
