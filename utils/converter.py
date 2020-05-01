import os
import cv2
import glob
import shutil
import skimage
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from skimage import io

import numpy as np

class Converter(object):
    def __init__(self, raw_images_path, split_rate, train_data_path = 'data\\Train', test_data_path = 'data\\Test', image_types = ['jpg', 'jpeg', 'png'], shard_size = 5120):
        self.raw_images_path = raw_images_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        self.split_rate = split_rate
        self.shard_size = shard_size
        self.image_types = image_types
        
    def getImagesPath(self):
        '''
        Get all images path in the raw data path.

        Returns:
            A list that contains all image path in the raw_images_path.
        '''
        images_path_list = []
        for path in self.raw_images_path:
            for image_type in self.image_types:
                images_path = os.path.join(path, '*.{}'.format(image_type))
                data_path = glob.glob(images_path, recursive=True)
                images_path_list.extend(data_path)
        return images_path_list

    def getData(self, image_path):
        '''
        Get a data from the image path.

        Args:
            image_path : the path of one image.
        
        Returns:
            A dictionary that contains a image and a mask.
        '''
        mask_path = image_path.replace('images', 'segs')

        image = self.readImage(image_path, 1)
        mask = self.readImage(mask_path, 0)

        m_height, m_width = mask.shape[:2]
        data = {"image" : image, "mask" : mask}
        return data

    def toExmaple(self, data):
        '''
        Convert a data to a example

        Args:
            A dictionary that contains a image and a mask.
        
        Returns:
            A tensorflow example that contains the information of one data.
        '''
        image, mask = data["image"], data["mask"]

        im_height, im_width = image.shape[:2]
        m_height, m_width = mask.shape[:2]
        # create example
        example = tf.train.Example(features=tf.train.Features(feature={
                'image': self._bytes_feature(image.tostring()),
                'im_height': self._int64_feature(im_height),
                'im_width': self._int64_feature(im_width),
                'mask': self._bytes_feature(mask.tostring()),
                'm_height': self._int64_feature(m_height),
                'm_width': self._int64_feature(m_width),
                }))
        return example

    def convert(self, path_list, output_path):
        '''
        Convert all data to the TFRecords and write to file. 

        Args:
            path_list : A list that contains the path of all images
            output_path : The path for TFRecord Writer
        
        Returns:
            The number of TFRecord that is actually written.
        '''
        print("[Converter] Converting data to {}....".format(output_path))
        shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path)  

        shard_id = 0
        shard_size_count = 0
        write_count = 0
        data_list = []
        for image_path in tqdm(path_list):
            # Create a writer
            if(shard_size_count == 0):
                shard_path = os.path.join(output_path, '%04d.tfrecords' % shard_id)
                writer = tf.io.TFRecordWriter(shard_path)

            # get a image data
            data = self.getData(image_path)
            if(len(data["image"]) == 0):
                continue

            # convert to a example
            example = self.toExmaple(data)

            # write a example
            writer.write(example.SerializeToString())
            write_count += 1
            shard_size_count += 1

            # Counting
            if(shard_size_count == self.shard_size):
                shard_id += 1
                shard_size_count = 0
                writer.close()
        return write_count

    def split(self):
        '''
        Split dataset to train dataset and test dataset.
        
        Returns:
            train_path_list : a list contains the path of all images in train dataset.
            test_path_list : a list contains the path of all images in test dataset.
        '''
        print("[Converter] Split data....")
        images_path_list = self.getImagesPath()

        # train test split
        num_split = int(len(images_path_list) * (1.0 - self.split_rate))
        train_path_list = images_path_list[-num_split:]
        test_path_list = images_path_list[:-num_split]

        print('[Converter] Train data : {}'.format(len(train_path_list)))
        print('[Converter] Test data : {}'.format(len(test_path_list)))
        print('[Converter] All data : {}'.format(len(images_path_list)))
        return train_path_list, test_path_list
    
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def _float32_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    def readImage(self, path, mode):
        '''
        Read a image from the path.

        Args:
            path : the path of a image.
            mode : cv2 imread mode.
        
        Returns:
            A image array.
        '''
        try:
            # chech image if is broken.
            skimage.io.imread(path)
        except Exception as e:
            print("[Converter] {} : {}".format(path, e))
            image = []
        else:
            image = cv2.imread(path, mode)
        return image

    def start(self):
        train_path_list, test_path_list = self.split()
        train_dataset_size = self.convert(train_path_list, self.train_data_path)
        test_dataset_size = self.convert(test_path_list, self.test_data_path)

        print('[Converter] Summary :')
        print('[Converter] Train data : {}'.format(train_dataset_size))
        print('[Converter] Test data : {}'.format(test_dataset_size))




