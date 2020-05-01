from utils.converter import Converter

split_rate = 0.1

# Source dataset
raw_images_path = ['data\\RawData\\images\\non-makeup', 'data\\RawData\\images\\makeup']
train_dataset_path = '.\\data\\source\\Train'
test_dataset_path = '.\\data\\source\\Test'

data_converter = Converter(raw_images_path, split_rate, train_dataset_path, test_dataset_path)
data_converter.start()

# Reference dataset
raw_images_path = ['data\\RawData\\images\\makeup']
train_dataset_path = '.\\data\\reference\\Train'
test_dataset_path = '.\\data\\reference\\Test'

data_converter = Converter(raw_images_path, split_rate, train_dataset_path, test_dataset_path)
data_converter.start()
