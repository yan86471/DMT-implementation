import os
import cv2
import glob
import shutil
import numpy as np

from API import FaceBeautyModel

output_path = "Transfer"
images_path = "data\\RawData\\images\\non-makeup\\*"
reference_image_path = "data\\RawData\\images\\makeup\\vFG234.png"

def images_demo(face_beauty_model, reference_image_path, images_path):
    # reference image
    ref_image = cv2.imread(reference_image_path)

    # input images
    images_data_path = glob.glob(images_path, recursive=True)
    images = []
    name = []
    for image_path in images_data_path:
        image = cv2.imread(image_path)
        images.append(image)
        name.append(image_path[image_path.rfind("\\")+1: -4])
    
    # transfer
    makeup_codes = face_beauty_model.getMakeupCode([ref_image])
    transfer_images = face_beauty_model.transfer(images, makeup_codes)
    return name, transfer_images

if __name__ == "__main__":
    face_beauty_model = FaceBeautyModel()

    # create output dir
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok = True)

    name, transfer_images = images_demo(face_beauty_model, reference_image_path, images_path)

    # save images
    for i in range(len(transfer_images)):
        path = os.path.join(output_path, '{}.png'.format(name[i]))
        cv2.imwrite(path, transfer_images[i])
    pass

