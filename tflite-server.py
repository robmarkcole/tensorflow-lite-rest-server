"""
Expose a tflite object detection model via a rest API.
"""
import argparse
import io
import logging

import flask
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

from helpers import read_coco_labels

app = flask.Flask(__name__)

LOGFORMAT = "%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s"
logging.basicConfig(filename="tflite-server.log", level=logging.DEBUG, format=LOGFORMAT)

interpreter = None
coco_labels = None
input_height = None
input_width = None

OBJ_DETECTION_URL = "/v1/vision/detection"
MODEL = "models/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
LABELS = "labels/coco_labels.txt"
CONFIDENCE_THRESHOLD = 0.6


@app.route("/")
def info():
    info_str = "Flask app exposing tensorflow lite model: {} \n".format(
        MODEL.split("/")[-1]
    )
    return info_str


@app.route(OBJ_DETECTION_URL, methods=["POST"])
def predict():
    data = {"success": False}
    if not flask.request.method == "POST":
        return

    if flask.request.files.get("image"):
        image_file = flask.request.files["image"]
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_width = image.size[0]
        image_height = image.size[1]
        resized_image = image.resize((input_width, input_height))

        input_data = np.expand_dims(resized_image, axis=0)
        interpreter.set_tensor(input_details[0]["index"], input_data)

        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]["index"])[
            0
        ]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]["index"])[
            0
        ]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]["index"])[
            0
        ]  # Confidence of detected objects

        objects = []
        for i in range(len(scores)):
            if (scores[i] > CONFIDENCE_THRESHOLD) and (scores[i] <= 1.0):
                single_object = {}
                single_object["label"] = coco_labels[int(classes[i])]
                single_object["confidence"] = float(scores[i])
                single_object["y_min"] = int(float(boxes[i][0]) * image_height)
                single_object["x_min"] = int(float(boxes[i][1]) * image_width)
                single_object["y_max"] = int(float(boxes[i][2]) * image_height)
                single_object["x_max"] = int(float(boxes[i][3]) * image_width)
                objects.append(single_object)

        data["predictions"] = objects
        data["success"] = True
        return flask.jsonify(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing tflite models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    interpreter = tflite.Interpreter(model_path=MODEL)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height = input_details[0]["shape"][1]
    input_width = input_details[0]["shape"][2]
    coco_labels = read_coco_labels(LABELS)

    app.run(host="0.0.0.0", debug=True, port=args.port)
