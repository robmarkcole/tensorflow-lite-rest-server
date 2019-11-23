# tensorflow-lite-rest-server
Expose tensorflow-lite models via a rest API. Equivalent of https://github.com/robmarkcole/coral-pi-rest-server but not requiring Coral hardware. Requires raspberry pi 4 (TBC)

## Setup
Manually install pip3, numpy, pillow then tensorflow-lite [as per these instructions](https://www.tensorflow.org/lite/guide/python). Then `$ pip3 install -r requirements.txt`

## Models
We require `.tflite` model files. Git clone https://github.com/google-coral/edgetpu and all models are in `edgetpu/test_data/`. On my machine the absolute path to this folder is `/home/pi/github/edgetpu/test_data/`

## Server
#Start the server:
```
$ python3 tflite-server.py
```
Submit a request via cURL:
```
$ curl -X POST -F image=@images/people_car.jpg 'http://localhost:5000/v1/vision/detection'
```

## Development
I am developing on a pi4 using VScode remote over SSH from my Mac. Install the dev requirements: `$ pip3 install -r requirements-dev.txt`. Sort requirements with `$ /home/pi/.local/bin/isort tflite-server.py`. Unfortunately appears black is not supported on pi4 yet.