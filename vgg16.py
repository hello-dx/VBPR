from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


base_model = VGG16(weights='imagenet',include_top=True)
# model = VGG16(weights='imagenet')
model = Model(  
    input=base_model.input, output=base_model.get_layer('fc2').output)


def extract_feature(img_path):

    img = image.load_img(img_path, target_size=(224, 224))  # 224,224
    # except:
    #     img_path='./images/000000.jpg'
    #     img = image.load_img(img_path, target_size=(224, 224))  # 224,224
    x = image.img_to_array(img)  # (3, 224, 224)
    x = np.expand_dims(x, axis=0)  # (1, 3, 224, 224)
    x = preprocess_input(x)
    features = model.predict(x)  # fc2
    # print(features.shape)
    # number = np.argmax(features)
    # print(words[number])
    return features
