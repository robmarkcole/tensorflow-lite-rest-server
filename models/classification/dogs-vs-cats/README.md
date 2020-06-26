## dogs vs cats
* Custom model trained on [teachable machine image classification](https://teachablemachine.withgoogle.com/train/image)
* Dataset: 958 cat and 1019 dog images from [kaggle dogs vs cats](https://www.kaggle.com/c/dogs-vs-cats)
* Input size: 224x224
* Type: tensorflow lite quantized
* Training time: 1 hour

## Notebooks
Also in this folder are notebooks which show both transfer and full training of a classification model, and export to tflite. Note that these notebooks take a long time to run, even on colab with GPU, expect the full training example to run for hours whilst transfer learning about 30 mins. Also I need to optimise the model compression, as my generated .tflite models are 80MB (full) and 40MB (quantized).