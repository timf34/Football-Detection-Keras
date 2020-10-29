import tensorflow as tf
import os
import sys
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
import math
import numpy as np
from constants import *
import imgaug.augmenters as iaa
from PIL import Image
from tensorflow.keras import preprocessing

def get_coords(data):
    """set up txt file as
    - image_name /n
    - x y /n """
    x = float(data[0]) * OUTPUT_WIDTH
    y = float(data[1]) * OUTPUT_HEIGHT
    x = int(x)
    y = int(y)
    return x, y

def heatmap_splat(x, y):
    #takes x, y coordinates of the centre of the ball and returns the ground truth heatmap
    ball_base = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 1), np.float)
    background_base= np.ones((OUTPUT_HEIGHT, OUTPUT_WIDTH, 1), np.float32)
    y_true = np.concatenate((ball_base, background_base), axis=-1)
    #if (x<5)
    #TODO: need to account for edge cases where the ball is near or at the edge of the image
    #array[(x-2):(x+3):1, y-3:y+4:1, 0] = 0.55
    #array[(x-3):(x+4):1, y-2:y+3:1, 0] = 0.55
    #array[(x-2):(x+3):1, y-2:y+3:1, 0] = 0.9
    #ball channel
    #y_true[(x-1):(x+2):1, y-1:y+2:1, 0] = 1.
    y_true[x, y, 0] = 1.
    #background channel
    y_true[(x - 1):(x + 2):1, y - 1:y + 2:1, 1] = 0.
    y_true[x, y, 1] = 0.

    y_true = y_true.astype('float32')

    return y_true

class DataGenerator(Sequence):

    def __init__(self, file_path, config_path, debug=False, augment=True):

        self.coords = []
        self.image_path = file_path
        self.debug = debug
        self.config = config_path
        self.augment = augment

        if not os.path.isfile(config_path):
            print("File path {} does not exist. Exiting...".format(config_path))
            sys.exit()

        if not os.path.isdir(file_path):
            print("Images folder path {} does not exist. Exiting...".format(file_path))
            sys.exit()

        with open(config_path) as fp:
            image_name = fp.readline()
            cnt = 1
            while image_name:
                co__ords = fp.readline().split(' ')
                x, y = get_coords(co__ords)

                self.coords.append((image_name.strip(), x, y))
                image_name = fp.readline()
                cnt += 1

    def __len__(self):
        return math.ceil(len(self.coords) / BATCH_SIZE)

    def __getitem__(self, index):
        co_ords = self.coords[index * BATCH_SIZE :(index + 1) * BATCH_SIZE]

        batch_images = np.zeros((len(co_ords), INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.float32)
        batch_heatmaps = np.zeros((len(co_ords), OUTPUT_HEIGHT, OUTPUT_WIDTH, 2), dtype=np.float32)

        for i, row in enumerate(co_ords):
            images_path, x, y = row

            proc_image = image.load_img(self.image_path + images_path, target_size=(INPUT_HEIGHT, INPUT_WIDTH))
            proc_image = image.img_to_array(proc_image, dtype='uint8')

            heatmap = heatmap_splat(y, x)  # y is height and x is width!!
            heatmap = np.expand_dims(heatmap, axis=0)

            aug_list = iaa.OneOf([
              iaa.Dropout([0.05, 0.1]),
              iaa.Sharpen((0.0, 1.0)),
              iaa.MultiplyHue((0.7,1.4)),
              iaa.MultiplyBrightness((0.7,1.4)),
            ])

            aug = iaa.Sequential([aug_list, iaa.Fliplr(0.5)], random_order=True)

            proc_image, heatmap = aug.augment(image=proc_image, heatmaps=heatmap)

            proc_image = np.expand_dims(proc_image, axis=0)
            proc_image = proc_image / 255. #just for now try without normalising
            batch_images[i] = proc_image
            batch_heatmaps[i] = heatmap

        return batch_images, batch_heatmaps
