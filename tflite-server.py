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
labels_file = None
input_details = None
output_details = None
model_height = None
model_width = None
floating_model = None

ROOT_URL = "/v1/vision/detection"


def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

# Function to read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
    return ret


@app.route("/")
def info():
    info_str = "Flask app exposing tensorflow lite model {}".format(MODEL)
    return info_str


@app.route(ROOT_URL, methods=["POST"])
def predict():
    data = {"success": False}

    if not flask.request.method == "POST":
        return
    try:
        if flask.request.files.get("image"):
            image_file = flask.request.files["image"]
            image_bytes = image_file.read()
            img = Image.open(io.BytesIO(image_bytes)).resize((model_width, model_height))

            # add N dim
            input_data = np.expand_dims(img, axis=0)

            if floating_model:
                input_mean = 127.5
                input_std = 127.5
                input_data = (np.float32(input_data) - args.input_mean) / args.input_std

            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            results = np.squeeze(output_data)
            # Run inference.
            top_k = results.argsort()[-5:][::-1]
            labels = load_labels(labels_file)
            for i in top_k:
                if floating_model:
                    print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
                else:
                    print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

            return flask.jsonify(data)

    except Exception as exc:
        data['error'] = str(exc)
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

    model_height = input_details[0]['shape'][1]
    model_width = input_details[0]['shape'][2]

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    print("\n Loaded model : {}".format(model_file))
    labels = ReadLabelFile(labels_file)
    app.run(host="0.0.0.0", debug=True, port=args.port)
