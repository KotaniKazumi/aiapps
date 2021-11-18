from django.db import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.core.protobuf import graph_debug_info_pb2
from tensorflow.keras.models import load_model
from PIL import Image
import io, base64

graph = tf.get_default_graph()

class Photo(models.Model):
    image = models.ImageField(upload_to='photos')

    IMAGE_SIZE = 224
    MODEL_FILE_PATH = './carbike/ml_models/vgg16_transfer.h5'
    classes = ["car", "motorbike"]
    num_classes = len(classes)

    def prefict(self):
        model =None
        global graph
        with graph.as_default():
            model = load_model(self.MODEL_FILE_PATH)

            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)

            image = Image.open(img_bin)
            image = image.convert("RGB")
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            data = np.asarray(image) / 255.0
            X = []
            X.append(data)
            X = np.array(X)
            
            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            print(self.classes[predicted], percentage)

