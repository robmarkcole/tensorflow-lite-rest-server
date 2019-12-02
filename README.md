# tensorflow-lite-rest-server
Expose tensorflow-lite models via a rest API. Currently object detection is supported.

Equivalent of https://github.com/robmarkcole/coral-pi-rest-server but not requiring Coral hardware. Requires raspberry pi 4 (TBC)

## Setup
Manually install pip3, numpy, pillow then tensorflow-lite [as per these instructions](https://www.tensorflow.org/lite/guide/python). Then `$ pip3 install -r requirements.txt`

## Models
We are using `.tflite` model files from https://github.com/google-coral/edgetpu which for convenience are in this repo.

## Server
#Start the server:
```
$ python3 tflite-server.py
```
You can check that the app is running by visiting `http://pi_ip:5000/` from any machine, where `pi_ip` is the ip address of the rasberry pi (`localhost` for requests from the pi itself). 

Submit a request via cURL:
```
$ curl -X POST -F image=@images/people_car.jpg 'http://localhost:5000/v1/object/detection'
```
Which should return:
```json
{
  "image_height": 480, 
  "image_width": 960, 
  "objects": [
    {
      "box": [
        0.25783607363700867, 
        0.33058103919029236, 
        0.7246008515357971, 
        0.6428706645965576
      ], 
      "name": "car", 
      "score": 0.953125
    }, 
    {
      "box": [
        0.28432977199554443, 
        0.28714725375175476, 
        0.6788091659545898, 
        0.37872299551963806
      ], 
      "name": "person", 
      "score": 0.73046875
    }, 
    {
      "box": [
        0.524590253829956, 
        0.22533127665519714, 
        0.8592637777328491, 
        0.3502572476863861
      ], 
      "name": "bicycle", 
      "score": 0.66015625
    }, 
    {
      "box": [
        0.2879002094268799, 
        0.2565079927444458, 
        0.6752387285232544, 
        0.36057573556900024
      ], 
      "name": "person", 
      "score": 0.58203125
    }, 
    {
      "box": [
        0.30700039863586426, 
        0.871345043182373, 
        0.8164434432983398, 
        0.9612630605697632
      ], 
      "name": "person", 
      "score": 0.33984375
    }
  ], 
  "
```

## Development
I am developing on a pi4 using VScode remote over SSH from my Mac. Install the dev requirements: `$ pip3 install -r requirements-dev.txt`. Sort requirements with `$ /home/pi/.local/bin/isort tflite-server.py`. Unfortunately appears black is not supported on pi4 yet.

## Jupyterlab
I have installed jupyterlab on the pi to assist with prototyping - note you can use vscode to connect to jupyter on the pi but I have found that very slow so I just connect to the standard jupyterlab interface using the ssh approach below. 
* `$ pip3 install jupyterlab `
* [Connect from remote machine via SSH port forwarding](https://www.blopig.com/blog/2018/03/running-jupyter-notebook-on-a-remote-server-via-ssh/) -> first run  `jupyter notebook --generate-config` then set default password using `jupyter notebook password`. Can then run notebook or lab (`jupyter lab --port=9000 --no-browser &`) and connect with ssh: `ssh -N -f -L 9000:localhost:9000 pi@ip` and visit `http://localhost:9000`