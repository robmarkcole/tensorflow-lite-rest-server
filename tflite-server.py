"""
Expose a tflite models via a rest API.
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

OBJ_DETECTION_URL = "/v1/vision/detection"
OBJ_MODEL = "models/object_detection/mobilenet_ssd_v2_coco/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
OBJ_LABELS = "models/object_detection/mobilenet_ssd_v2_coco/coco_labels.txt"


@app.route("/")
def info():
    return (
        f"""Flask app exposing object detection model: {OBJ_MODEL.split("/")[-2]} \n"""
    )


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
        resized_image = image.resize((obj_input_width, obj_input_height))

        input_data = np.expand_dims(resized_image, axis=0)
        obj_interpreter.set_tensor(obj_input_details[0]["index"], input_data)

        obj_interpreter.invoke()
        boxes = obj_interpreter.get_tensor(obj_output_details[0]["index"])[0]
        classes = obj_interpreter.get_tensor(obj_output_details[1]["index"])[0]
        scores = obj_interpreter.get_tensor(obj_output_details[2]["index"])[0]

        objects = []
        for i in range(len(scores)):
            single_object = {}
            single_object["label"] = obj_labels[int(classes[i])]
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

    obj_interpreter = tflite.Interpreter(model_path=OBJ_MODEL)
    obj_interpreter.allocate_tensors()
    obj_input_details = obj_interpreter.get_input_details()
    obj_output_details = obj_interpreter.get_output_details()
    obj_input_height = obj_input_details[0]["shape"][1]
    obj_input_width = obj_input_details[0]["shape"][2]
    obj_labels = read_coco_labels(OBJ_LABELS)

    app.run(host="0.0.0.0", debug=True, port=args.port)
