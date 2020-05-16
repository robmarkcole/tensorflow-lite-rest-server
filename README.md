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
We are using `.tflite` model files from https://github.com/google-coral/edgetpu which for convenience are in this repo.

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
{'predictions': [
   {'confidence': 0.93359375, 
    'label': 'car', 
    'x_max': 0.6453010439872742, 
    'x_min': 0.31541913747787476, 
    'y_max': 0.7257205843925476, 
    'y_min': 0.25035059452056885},
.
.
.
'success': True}
```

## Development
I am developing on a mac/pi4 using VScode. Install the dev requirements: `$ pip3 install -r requirements-dev.txt`. Sort requirements with `$ /home/pi/.local/bin/isort tflite-server.py`. Unfortunately appears black is not supported on pi4 yet.