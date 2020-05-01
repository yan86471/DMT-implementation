import os
import shutil

from train import params
from utils.model import NNModel

save_model_path = "logs\\01\\checkpoint\\0100.ckpt"
export_path = "Export\\"

if __name__ == "__main__":
    os.makedirs(export_path, exist_ok = True)
    model = NNModel(input_shape = params['image_shape'], logs_path = params['logs_path'], batch_size = params['batch_size'], classes = params['classes'])

    model.export_model(save_model_path, export_path)





