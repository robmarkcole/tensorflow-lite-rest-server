# Start the server:
# 	python3 tflite-server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@face.jpg 'http://localhost:5000/v1/vision/detection'

import argparse
import numpy as np
import io
import logging

import flask
import tflite_runtime.interpreter as tflite
from PIL import Image

app = flask.Flask(__name__)

LOGFORMAT = "%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s"
logging.basicConfig(filename='tflite-server.log', level=logging.DEBUG, format=LOGFORMAT)

interpreter = None
labels = None
input_details = None
input_shape = None
output_details = None
floating_model = None

input_mean = 127.5
input_std = 127.5

ROOT_URL = "/v1/vision/detection"


def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

@app.route("/")
def info():
    info_str = "Flask app exposing tensorflow lite model {}".format(MODEL)
    return info_str


@app.route(ROOT_URL, methods=["POST"])
def predict():
    data = {"success": False}

    if not flask.request.method == "POST":
        return

    if flask.request.files.get("image"):
        model_height = input_shape[1]
        model_width = input_shape[2]

        image_file = flask.request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes)).resize((model_width, model_height))

        # add N dim
        input_data = np.expand_dims(img, axis=0)
        # Normalise the pixel data if floating point model
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        print("XXXXX")

        results = np.squeeze(output_data)

        top_k = results.argsort()[-5:][::-1]

        for i in top_k:
            print(i)
            #print(results[i], labels[i])
            
            #if floating_model:
            #    print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            #else:
            #    print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

        return flask.jsonify(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing tflite models")
    parser.add_argument(
        "--models_directory",
        default="/home/pi/github/edgetpu/test_data/",
        help="the directory containing the model & labels files",
    )
    parser.add_argument(
        "--model",
        default="mobilenet_ssd_v2_coco_quant_postprocess.tflite",
        help="model file",
    )
    parser.add_argument(
        "--labels", default="coco_labels.txt", help="labels file of model"
    )
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    global MODEL
    MODEL = args.model
    model_file = args.models_directory + args.model
    labels_file = args.models_directory + args.labels

    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    print("\n Loaded model : {}".format(model_file))
    if floating_model:
        print("\n The model is a floating_model")
    labels = load_labels(labels_file)
    app.run(host="0.0.0.0", debug=True, port=args.port)
