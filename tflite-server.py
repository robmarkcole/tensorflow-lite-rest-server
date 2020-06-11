"""
Expose a tflite models via a rest API.
"""
import argparse
import io
import logging
import sys

import flask
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

from helpers import read_labels, set_input_tensor, classify_image

app = flask.Flask(__name__)

LOGFORMAT = "%(asctime)s %(levelname)s %(name)s : %(message)s"
logging.basicConfig(
    # filename="tflite-server.log", # select filename or stream
    stream=sys.stdout,
    level=logging.DEBUG,
    format=LOGFORMAT,
)

MIN_CONFIDENCE = 0.1  # The absolute lowest confidence for a detection.

FACE_DETECTION_URL = "/v1/vision/face"
FACE_MODEL = "models/face_detection/mobilenet_ssd_v2_face/mobilenet_ssd_v2_face_quant_postprocess.tflite"

OBJ_DETECTION_URL = "/v1/vision/detection"
OBJ_MODEL = "models/object_detection/mobilenet_ssd_v2_coco/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
OBJ_LABELS = "models/object_detection/mobilenet_ssd_v2_coco/coco_labels.txt"

SCENE_URL = "/v1/vision/scene"
SCENE_MODEL = "models/classification/dogs-vs-cats/model.tflite"
SCENE_LABELS = "models/classification/dogs-vs-cats/labels.txt"


@app.route("/")
def info():
    return f"""
        Object detection model: {OBJ_MODEL.split("/")[-2]} \n
        Face detection model: {FACE_MODEL.split("/")[-2]} \n
        Scene model: {SCENE_MODEL.split("/")[-2]} \n
        """.replace(
        "\n", "<br>"
    )


@app.route(FACE_DETECTION_URL, methods=["POST"])
def predict_face():
    data = {"success": False}
    if not flask.request.method == "POST":
        return

    if flask.request.files.get("image"):
        # Open image and get bytes and size
        image_file = flask.request.files["image"]
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))  # A PIL image
        image_width = image.size[0]
        image_height = image.size[1]

        # Format data and send to interpreter
        resized_image = image.resize((face_input_width, face_input_height))
        input_data = np.expand_dims(resized_image, axis=0)
        face_interpreter.set_tensor(face_input_details[0]["index"], input_data)

        # Process image and get predictions
        face_interpreter.invoke()
        boxes = face_interpreter.get_tensor(face_output_details[0]["index"])[0]
        classes = face_interpreter.get_tensor(face_output_details[1]["index"])[
            0
        ]
        scores = face_interpreter.get_tensor(face_output_details[2]["index"])[0]

        faces = []
        for i in range(len(scores)):
            if not classes[i] == 0:  # Face
                continue
            single_face = {}
            single_face["confidence"] = float(scores[i])
            single_face["userid"] = "unknown"
            single_face["y_min"] = int(float(boxes[i][0]) * image_height)
            single_face["x_min"] = int(float(boxes[i][1]) * image_width)
            single_face["y_max"] = int(float(boxes[i][2]) * image_height)
            single_face["x_max"] = int(float(boxes[i][3]) * image_width)
            if single_face["confidence"] < MIN_CONFIDENCE:
                continue
            faces.append(single_face)

        data["predictions"] = faces
        data["success"] = True
        return flask.jsonify(data)


@app.route(OBJ_DETECTION_URL, methods=["POST"])
def predict_object():
    data = {"success": False}
    if not flask.request.method == "POST":
        return

    if flask.request.files.get("image"):
        # Open image and get bytes and size
        image_file = flask.request.files["image"]
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))  # A PIL image
        image_width = image.size[0]
        image_height = image.size[1]

        # Format data and send to interpreter
        resized_image = image.resize((obj_input_width, obj_input_height))
        input_data = np.expand_dims(resized_image, axis=0)
        obj_interpreter.set_tensor(obj_input_details[0]["index"], input_data)

        # Process image and get predictions
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

            if single_object["confidence"] < MIN_CONFIDENCE:
                continue
            objects.append(single_object)

        data["predictions"] = objects
        data["success"] = True
        return flask.jsonify(data)


@app.route(SCENE_URL, methods=["POST"])
def predict_scene():
    data = {"success": False}
    if not flask.request.method == "POST":
        return

    if flask.request.files.get("image"):
        # Open image and get bytes and size
        image_file = flask.request.files["image"]
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))  # A PIL image
        # Format data and send to interpreter
        resized_image = image.resize(
            (scene_input_width, scene_input_height), Image.ANTIALIAS
        )
        results = classify_image(scene_interpreter, image=resized_image)

        print(
            f"results[0]: {results[0]}", file=sys.stderr,
        )
        label_id, prob = results[0]

        data["label"] = scene_labels[label_id]
        data["confidence"] = prob
        data["success"] = True
        return flask.jsonify(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask app exposing tflite models"
    )
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    # Setup object detection
    obj_interpreter = tflite.Interpreter(model_path=OBJ_MODEL)
    obj_interpreter.allocate_tensors()
    obj_input_details = obj_interpreter.get_input_details()
    obj_output_details = obj_interpreter.get_output_details()
    obj_input_height = obj_input_details[0]["shape"][1]
    obj_input_width = obj_input_details[0]["shape"][2]
    obj_labels = read_labels(OBJ_LABELS)

    # Setup face detection
    face_interpreter = tflite.Interpreter(model_path=FACE_MODEL)
    face_interpreter.allocate_tensors()
    face_input_details = face_interpreter.get_input_details()
    face_output_details = face_interpreter.get_output_details()
    face_input_height = face_input_details[0]["shape"][1]  # 320
    face_input_width = face_input_details[0]["shape"][2]  # 320

    # Setup face detection
    scene_interpreter = tflite.Interpreter(model_path=SCENE_MODEL)
    scene_interpreter.allocate_tensors()
    scene_input_details = scene_interpreter.get_input_details()
    scene_output_details = scene_interpreter.get_output_details()
    scene_input_height = scene_input_details[0]["shape"][1]
    scene_input_width = scene_input_details[0]["shape"][2]
    scene_labels = read_labels(SCENE_LABELS)

    app.run(host="0.0.0.0", debug=True, port=args.port)
