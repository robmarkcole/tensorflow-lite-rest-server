"""Perform processing of test images."""
import pprint

import requests

print("-----------------------------")
OBJ_DETECTION_URL = "http://localhost:5000/v1/vision/detection"
OBJ_DETECTION_TEST_IMAGE = "people_car.jpg"
image_data = open(OBJ_DETECTION_TEST_IMAGE, "rb").read()
response = requests.post(OBJ_DETECTION_URL, files={"image": image_data}).json()
print(OBJ_DETECTION_TEST_IMAGE)
pprint.pprint(response)

print("-----------------------------")
FACE_DETECTION_URL = "http://localhost:5000/v1/vision/face"
FACE_DETECTION_TEST_IMAGE = "faces.jpg"
image_data = open(FACE_DETECTION_TEST_IMAGE, "rb").read()
response = requests.post(FACE_DETECTION_URL, files={"image": image_data}).json()
print(FACE_DETECTION_TEST_IMAGE)
pprint.pprint(response)

print("-----------------------------")
SCENE_URL = "http://localhost:5000/v1/vision/scene"
SCENE_TEST_IMAGE = "dog.jpg"
image_data = open(SCENE_TEST_IMAGE, "rb").read()
response = requests.post(SCENE_URL, files={"image": image_data}).json()
print(SCENE_TEST_IMAGE)
pprint.pprint(response)
