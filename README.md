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
You can check that the app is running by visiting `http://pi_ip:5000/` from any machine, where `pi_ip` is the ip address of the rasberry pi (`localhost` for requests from the pi itself). 

Submit a request via cURL:
```
$ curl -X POST -F image=@images/people_car.jpg 'http://localhost:5000/v1/vision/detection'
```

## Summary
Mobilenet v1 is a classification model it is working fine. V2 are object (COCO) and face but I do not know how to interpret the outputs of the model.

## Development
I am developing on a pi4 using VScode remote over SSH from my Mac. Install the dev requirements: `$ pip3 install -r requirements-dev.txt`. Sort requirements with `$ /home/pi/.local/bin/isort tflite-server.py`. Unfortunately appears black is not supported on pi4 yet.

## Jupyterlab
I have installed jupyterlab on the pi to assist with prototyping. 
* `$ pip3 install jupyterlab `
* [Connect from remote machine via SSH port forwarding](https://www.blopig.com/blog/2018/03/running-jupyter-notebook-on-a-remote-server-via-ssh/) -> first run  `jupyter notebook --generate-config` then set default password using `jupyter notebook password`. Can then run notebook or lab (`jupyter lab --port=9000 --no-browser &`) and connect with ssh: `ssh -N -f -L 9000:localhost:9000 pi@ip` and visit `http://localhost:9000`