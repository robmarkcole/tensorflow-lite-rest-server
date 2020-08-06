FROM python:3.7

EXPOSE 5000

WORKDIR /app

ADD requirements.txt /app/
RUN pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl
RUN pip install -r requirements.txt

ADD . /app

CMD [ "uvicorn", "tflite-server:app", "--reload", "--port", "5000", "--host", "0.0.0.0" ]
