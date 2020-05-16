import requests

API_URL = "http://localhost:5000/v1/vision/detection"

image_data = open("people_car.jpg", "rb").read()

response = requests.post(API_URL, files={"image": image_data}).json()

print(response)
