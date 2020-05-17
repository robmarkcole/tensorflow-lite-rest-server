# tensorflow-lite-rest-server
Expose tensorflow-lite models via a rest API, and currently object detection is supported. Can be hosted on any of the common platforms including RPi, linux desktop, Mac and Windows.

## Setup
In this process we create a virtual environment (venv), then install tensorflow-lite [as per these instructions](https://www.tensorflow.org/lite/guide/python) which is platform specific, and finally install the remaining requirements. Note on an RPi (only) it is necessary to manually install pip3, numpy, pillow.

All instructions for mac:
```
python3.7 -m venv venv
source venv/bin/activate
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl
pip3 install -r requirements.txt
```

## Models
Models are available [here](https://github.com/google-coral/edgetpu/tree/master/test_data) with a short description of the models [here](https://coral.ai/models/). For convenience are the `mobilenet_ssd_v2` model is included in this repo and used by default. This model can detect 90 types of object, with object labels listed in `labels/coco_labels.txt`

## Usage
Start the server:
```
python3 tflite-server.py
```

You can check that the app is running by visiting `http://ip:5000/` from any machine, where `ip` is the ip address of the host (`localhost` if querying from the same machine).

Post an image for processing via cURL:
```
curl -X POST -F image=@tests/people_car.jpg 'http://localhost:5000/v1/vision/detection'
```
Which should return:
```
{
  "predictions": [
    {
      "confidence": 0.93359375, 
      "label": "car", 
      "x_max": 619, 
      "x_min": 302, 
      "y_max": 348, 
      "y_min": 120
    }, 
    {
      "confidence": 0.7890625, 
      "label": "person", 
      "x_max": 363, 
      "x_min": 275, 
      "y_max": 323, 
      "y_min": 126
    },
.
.
.
'success': True}
```

## Deepstack & Home Assistant
This API can be used as a drop in replacement for [deepstack object detection](https://github.com/robmarkcole/HASS-Deepstack-object) in Home Assistant.

## Development
I am developing on a mac/pi4 using VScode. Install the dev requirements: `$ pip3 install -r requirements-dev.txt`. Sort requirements with `$ /home/pi/.local/bin/isort tflite-server.py`. Unfortunately appears black is not supported on pi4 yet.