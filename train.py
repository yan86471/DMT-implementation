import tensorflow as tf
from utils.dataset import Dataset
from utils.model import NNModel

params = {
        "epochs" : 100,
        "batch_size" : 2,

        "image_shape" : (224, 224, 3),
        "classes" : {"face" : [1, 6, 11, 12, 13], "brow" : [2, 3], "eye" : [2, 3, 4, 5], "lip" : [7, 9], "non-makeup" : [0, 4, 5, 8, 10], 
                     "hair" : [10]},

        "logs_path" : 'logs\\01\\',
        "pretrained_model_path" : None,#'logs\\pretrained\\0100.ckpt',

        "train_dataset_path" : {'source' : r'.\\data\\source\\Train', 'reference' : r'.\\data\\reference\\Train'},
        "train_dataset_size" : [3450, 2447],
        "test_dataset_path" : [r'.\\data\\source\\Test', r'.\\data\\reference\\Test'],
        }


if __name__ == "__main__":
    train_dataset = Dataset(params['train_dataset_path'], image_shape = params['image_shape'], classes = params['classes'], batch_size = params['batch_size'], dataset_size = params['train_dataset_size'], isTraining = True)

    model = NNModel(input_shape = params['image_shape'], logs_path = params['logs_path'], batch_size = params['batch_size'], classes = params['classes'])

    model.train(train_dataset, params["epochs"], pretrained_model_path = params['pretrained_model_path'])

