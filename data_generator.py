import numpy as np
from sklearn.utils import shuffle


class DataGenerator:
    def __init__(self, train_folder_path, test_folder_path):
        self.x_train, self.y_train, self.x_test, self.y_test = self.get_data(train_folder_path, test_folder_path)

    @staticmethod
    def load_label_names(self):
        return ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                'worm']

    def get_images(self, raw):
        raw_float = np.array(raw, dtype=float)
        images = raw_float.reshape([-1, 3, 32, 32])
        images = images.transpose([0, 2, 3, 1])
        return images

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            try:
                dict = pickle.load(fo)
            except UnicodeDecodeError:
                fo.seek(0)
                dict = pickle.load(fo, encoding='latin1')
        return dict

    def normalize(self, x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def one_hot_encode(self, x):
        encoded = np.zeros((len(x), 100))

        for idx, val in enumerate(x):
            encoded[idx][val] = 1

        return encoded

    def data_processing(self, features, labels):
        return self.normalize(features), self.one_hot_encode(labels)

    def get_data(self, data_train_path, data_test_path):
        x_train = self.get_images(self.unpickle(data_train_path)['data'])
        y_train = self.unpickle(data_train_path)['fine_labels']
        x_test = self.get_images(self.unpickle(data_test_path)['data'])
        y_test = self.unpickle(data_test_path)['fine_labels']
        x_train, y_train = self.data_processing(x_train, y_train)
        x_test, y_test = self.data_processing(x_test, y_test)
        print('Train Data Shape is ', x_train.shape, y_train.shape)
        print('Test Data Shape is ', x_test.shape, y_test.shape)
        x_train, y_train = shuffle(x_train, y_train)
        x_test, y_test = shuffle(x_test, y_test)
        return x_train, y_train, x_test, y_test

    def next_batch(self, batch_size):
        idx = np.random.choice(len(self.x_train), batch_size)
        yield self.x_train[idx], self.y_train[idx]

    def get_validation(self):
        return self.x_test, self.y_test

    def get_train(self):
        return self.x_train, self.y_train
