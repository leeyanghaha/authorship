from PIL import Image
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
import utils.function_utils as fu
import utils.key_utils as ku
import os
import sklearn.utils as sku

class ResNet:
    def __init__(self):
        self.model = self.net()

    def net(self):
        input = Input((256, 256, 3), name='image_input')
        output = ResNet50(weights='imagenet', include_top=False, pooling='max')(input)
        model = Model(inputs=input, outputs=output)
        return model

    def get_feature(self, x):
        x = preprocess_input(x)
        return self.model.predict(x)


def load_photo(image_file):
    try:
        image = Image.open(image_file)
        image = image.resize([256, 256], Image.ANTIALIAS)
        image = np.array(image)
        image = image[..., :3]
    except OSError:
        image = None
    return image


