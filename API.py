import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class FaceBeautyModel(object):
    def __init__(self):
        self.makeup_model_path = "Export\\MakeupEncoder.h5"
        self.generator_model_path = "Export\\Generator.h5"

        self.MakeupEncoder = self.loadMakeupEncoder()
        self.Generator = self.loadGenerator()

        self.image_size = (224, 224)
        pass

    def loadMakeupEncoder(self):
        return tf.keras.models.load_model(self.makeup_model_path)

    def loadGenerator(self):
        return tf.keras.models.load_model(self.generator_model_path, custom_objects = {'InstanceNormalization':tfa.layers.InstanceNormalization})

    def getMakeupCode(self, images):
        images = self.preprocessingImages(images)
        makeup_code = self.MakeupEncoder.predict(images)
        return makeup_code

    def transfer(self, images, makeup_codes, predict_batch  = 10):
        images = self.preprocessingImages(images)
        if(len(images) > len(makeup_codes[0])):
            makeup_codes[0] = np.repeat(makeup_codes[0], len(images), axis = 0)
            makeup_codes[1] = np.repeat(makeup_codes[1], len(images), axis = 0)
        else:
            makeup_codes[0] = makeup_codes[0][:len(images)]
            makeup_codes[1] = makeup_codes[1][:len(images)]


        transfer_images = self.Generator.predict([images, makeup_codes[0], makeup_codes[1]], batch_size = predict_batch)
        transfer_images = self.postprocessingImages(transfer_images)
        return transfer_images

    def preprocessingImages(self, images):
        for i in range(len(images)):
            images[i] = cv2.resize(images[i], self.image_size)
        images = np.array(images, dtype = np.float32)
        images = (images / 255.0  - 0.5) * 2
        return images

    def postprocessingImages(self, images):
        images = np.array(images, dtype = np.float32)
        images = (images / 2 + 0.5) * 255.0
        return images