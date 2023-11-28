"""
Expose tflite models via a rest API.
"""
import io
import sys

import numpy as np
import tflite_runtime.interpreter as tflite
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from helpers import classify_image, read_labels, set_input_tensor

from os.path import exists

app = FastAPI()

# Settings
MIN_CONFIDENCE = 0.1  # The absolute lowest confidence for a detection.
# URL
FACE_DETECTION_URL = "/v1/vision/face"
OBJ_DETECTION_URL = "/v1/vision/detection"
SCENE_URL = "/v1/vision/scene"
# Models and labels
FACE_MODEL = "models/face_detection/mobilenet_ssd_v2_face/mobilenet_ssd_v2_face_quant_postprocess.tflite"
OBJ_MODEL = "models/object_detection/mobilenet_ssd_v2_coco/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
OBJ_LABELS = "models/object_detection/mobilenet_ssd_v2_coco/coco_labels.txt"
SCENE_MODEL = "models/classification/dogs-vs-cats/model.tflite"
SCENE_LABELS = "models/classification/dogs-vs-cats/labels.txt"
ADDITIONAL_PREFIX = "models/additional/"
ADDITIONAL_MODEL = "$$MODEL_NAME$$/model.tflite"
ADDITIONAL_LABELS = "$$MODEL_NAME$$/labels.txt"

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
face_input_height = face_input_details[0]["shape"][1]
face_input_width = face_input_details[0]["shape"][2]

# Setup face detection
scene_interpreter = tflite.Interpreter(model_path=SCENE_MODEL)
scene_interpreter.allocate_tensors()
scene_input_details = scene_interpreter.get_input_details()
scene_output_details = scene_interpreter.get_output_details()
scene_input_height = scene_input_details[0]["shape"][1]
scene_input_width = scene_input_details[0]["shape"][2]
scene_labels = read_labels(SCENE_LABELS)


def build_interpreter(model_name):
    model_path = ADDITIONAL_MODEL.replace("$$MODEL_NAME$$", model_name)
    model_labels = ADDITIONAL_LABELS.replace("$$MODEL_NAME$$", model_name)
    return inner_interpreter_builder(ADDITIONAL_PREFIX+model_path, ADDITIONAL_PREFIX+model_labels)

def inner_interpreter_builder(model_path, model_labels):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height = input_details[0]["shape"][1]
    input_width = input_details[0]["shape"][2]
    file_exists = exists(model_labels)
    if file_exists:
        labels = read_labels(model_labels)
    else:
        labels = None
    return interpreter, input_details, output_details, input_height, input_width, labels

@app.get("/")
async def info():
    return """tflite-server docs at ip:port/docs"""


@app.post(FACE_DETECTION_URL)
async def predict_face(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        image_width = image.size[0]
        image_height = image.size[1]

        # Format data and send to interpreter
        resized_image = image.resize((face_input_width, face_input_height), Image.ANTIALIAS)
        input_data = np.expand_dims(resized_image, axis=0)
        face_interpreter.set_tensor(face_input_details[0]["index"], input_data)

        # Process image and get predictions
        face_interpreter.invoke()
        boxes = face_interpreter.get_tensor(face_output_details[0]["index"])[0]
        classes = face_interpreter.get_tensor(face_output_details[1]["index"])[0]
        scores = face_interpreter.get_tensor(face_output_details[2]["index"])[0]

        data = {}
        faces = []
        for i in range(len(scores)):
            if not classes[i] == 0:  # Face
                continue
            single_face = {}
            single_face["userid"] = "unknown"
            single_face["confidence"] = float(scores[i])
            single_face["y_min"] = int(float(boxes[i][0]) * image_height)
            single_face["x_min"] = int(float(boxes[i][1]) * image_width)
            single_face["y_max"] = int(float(boxes[i][2]) * image_height)
            single_face["x_max"] = int(float(boxes[i][3]) * image_width)
            if single_face["confidence"] < MIN_CONFIDENCE:
                continue
            faces.append(single_face)

        data["predictions"] = faces
        data["success"] = True
        return data
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


@app.post(OBJ_DETECTION_URL)
async def predict_object(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        image_width = image.size[0]
        image_height = image.size[1]

        # Format data and send to interpreter
        resized_image = image.resize((obj_input_width, obj_input_height), Image.ANTIALIAS)
        input_data = np.expand_dims(resized_image, axis=0)
        obj_interpreter.set_tensor(obj_input_details[0]["index"], input_data)

        # Process image and get predictions
        obj_interpreter.invoke()
        boxes = obj_interpreter.get_tensor(obj_output_details[0]["index"])[0]
        classes = obj_interpreter.get_tensor(obj_output_details[1]["index"])[0]
        scores = obj_interpreter.get_tensor(obj_output_details[2]["index"])[0]

        data = {}
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
        return data
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


@app.post(SCENE_URL)
async def predict_scene(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        resized_image = image.resize((scene_input_width, scene_input_height), Image.ANTIALIAS)
        results = classify_image(scene_interpreter, image=resized_image)
        label_id, prob = results[0]
        data = {}
        data["label"] = scene_labels[label_id]
        data["confidence"] = prob
        data["success"] = True
        return data
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/vision/detection/{model_name}")
async def predict_additional_vision_detection(model_name: str, image: UploadFile = File(...)):
    try:
        interpreter, input_details, output_details, input_height, input_width, labels = build_interpreter(model_name)
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        image_width = image.size[0]
        image_height = image.size[1]

        # Format data and send to interpreter
        resized_image = image.resize((input_width, input_height), Image.ANTIALIAS)
        input_data = np.expand_dims(resized_image, axis=0)
        interpreter.set_tensor(input_details[0]["index"], input_data)

        # Process image and get predictions
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        classes = interpreter.get_tensor(output_details[1]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]["index"])[0]

        data = {}
        items = []
        for i in range(len(scores)):
            if not classes[i] == 0:  # Item
                continue
            single_item = {}
            single_item["userid"] = "unknown"
            if labels is not None:
                single_item["label"] = labels[int(classes[i])]
            single_item["confidence"] = float(scores[i])
            single_item["y_min"] = int(float(boxes[i][0]) * image_height)
            single_item["x_min"] = int(float(boxes[i][1]) * image_width)
            single_item["y_max"] = int(float(boxes[i][2]) * image_height)
            single_item["x_max"] = int(float(boxes[i][3]) * image_width)
            if single_item["confidence"] < MIN_CONFIDENCE:
                continue
            items.append(single_item)

        data["predictions"] = items
        data["success"] = True
        return data
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/vision/classification/{model_name}")
async def predict_additional_vision_classification(model_name: str, image: UploadFile = File(...)):
    try:
        interpreter, input_details, output_details, input_height, input_width, labels = build_interpreter(model_name)
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        resized_image = image.resize((input_width, input_height), Image.ANTIALIAS)
        results = classify_image(interpreter, image=resized_image)
        label_id, prob = results[0]
        data = {}
        data["label"] = labels[label_id]
        data["confidence"] = prob
        data["success"] = True
        return data
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))

