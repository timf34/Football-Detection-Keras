import tensorflow as tf
import numpy as np
from keras import backend as K
from loss import*
import imgaug.augmenters as iaa
import imgaug.augmenters.meta as iam
from constants import*
from PIL import Image
from tensorflow.keras.preprocessing import image
import imageio
import imgaug as ia
import numpy as np
from matplotlib import pyplot as plt


proc_image = image.load_img(TEST_PHOTO, target_size=(INPUT_HEIGHT, INPUT_WIDTH))
proc_image = image.img_to_array(proc_image, dtype='uint8')
#imageio.imwrite("example_segmaps.jpg", proc_image)
#proc_image = np.expand_dims(proc_image, axis=0)

heatmap = np.zeros((1, OUTPUT_HEIGHT, OUTPUT_WIDTH, 1), dtype=np.float32)
heatmap[0, 20, 140, 0] = 1.
#heatmap =heatmap[:,:,:,0]
#plt.imshow(heatmap, cmap='gray')
#plt.show()


aug_list = iaa.OneOf([
  #iaa.Dropout([0.02, 0.1]),
  #iaa.Sharpen((0.0, 1.0)),
  iaa.MultiplyHue((0.7, 1.4)),
  #iaa.MultiplyBrightness((0.7, 1.4))
])

aug = iaa.Sequential([aug_list, iaa.Fliplr(0.5)], random_order=True)

proc, hm= aug.augment(image=proc_image, heatmaps=heatmap)

hm = hm[0,:,:,0]
plt.imshow(hm, cmap='gray')
plt.show()

imageio.imwrite("example_segmaps.jpg", proc)



"""kvar1 = tf.keras.backend.variable(np.array([[[False, False, False], [False, False, False], [False, False, False]]]),
                                 dtype='bool')
kvar2 = tf.keras.backend.variable(np.array([[[[0, 1, 1], [1, 0,0]]]]),
                                 dtype='float32')
gt = tf.keras.backend.variable(np.array([[[[0, 0, 1], [0, 0,0]]]]),
                                 dtype='float32')

x=deepball_loss_function(gt, kvar2)"""

"""c= K.cast(K.greater(kvar2, 0.99), 'float32')
print ('c')
print(c)
g = K.sum(gt * kvar2)
print('g')
print (g)
totalp = K.max(c, axis=(1, 2))
print('tot')
print(totalp)

prec = g / tf.maximum(1.0, totalp)
print(prec)"""