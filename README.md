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
For convenience a couple of models are included in this repo and used by default. A description of each model is included in its directory. Additional models are available [here](https://github.com/google-coral/edgetpu/tree/master/test_data)

## Usage
Start the server on port 5000 (default is port 5000):
```
(venv) $ python3 tflite-server.py --port 5000
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
I am developing on a mac/pi4 using VScode. On mac use a venv, on pi install system wide.

* First time only, create venv: `python3.7 -m venv venv`
* Activate venv: `source venv/bin/activate`
* Install the dev requirements: `pip3 install -r requirements.txt` & `pip3 install -r requirements-dev.txt`
* Sort requirements: `venv/bin/isort tflite-server.py`
* Black format: `venv/bin/black tflite-server.py`
* Run the `usage.ipynb` notebook: `venv/bin/jupyter notebook`

Unfortunately appears black is not supported on pi4 yet.