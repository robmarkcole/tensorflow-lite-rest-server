# Start the server:
# 	python3 tflite-server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@people-car.jpg 'http://localhost:5000/v1/object/detection'

import argparse
import numpy as np
import io
import logging

import flask
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageDraw

app = flask.Flask(__name__)

LOGFORMAT = "%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s"
logging.basicConfig(filename='tflite-server.log', level=logging.DEBUG, format=LOGFORMAT)

interpreter = None
labels = None
input_height = None
input_width =  None

ROOT_URL = "/v1/object/detection"


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

    if flask.request.files.get("image"):
        image_file = flask.request.files["image"]
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        resized_image = image.resize((input_width, input_height))

        input_data = np.expand_dims(resized_image, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

        min_conf_threshold = 0.3
        draw = ImageDraw.Draw(image)
        img_width = image.size[0]
        img_height = image.size[1]

        objects = []

        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                single_object = {}
                y_min = boxes[i][0]
                x_min = boxes[i][1]
                y_max = boxes[i][2]
                x_max = boxes[i][3]
                box = (y_min, x_min, y_max, x_max)

                single_object['name'] = labels[int(classes[i])]
                single_object['box'] = box
                single_object['score'] = scores[i]
                objects.append(single_object)

        data['objects'] = objects
        return flask.jsonify(str(data))


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
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]


    print("\n Loaded model : {}".format(model_file))
    labels = ReadLabelFile(labels_file)
    app.run(host="0.0.0.0", debug=True, port=args.port)
