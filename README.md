# tensorflow-lite-rest-server
Expose tensorflow-lite models via a rest API

Equivalent of https://github.com/robmarkcole/coral-pi-rest-server but not requiring Coral hardware

Requires raspberry pi 4

## Setup
Install pip3, numpy, pillow then tensorflow-lite [as per these instructions](https://www.tensorflow.org/lite/guide/python).

## `label_image.py`
* https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python
* Get the data - I have changed these slightly

```
# Get photo
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp
# Get model
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
tar zxvf mobilenet_v1_1.0_224.tgz
# Get labels
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz
tar zxvf mobilenet_v1_1.0_224_frozen.tgz
```

Run the demo:
```
python3 label_image.py \
  --model_file mobilenet_v1_1.0_224.tflite \
  --label_file mobilenet_v1_1.0_224/labels.txt \
  --image grace_hopper.bmp
```
