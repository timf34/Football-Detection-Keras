from model import create_model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers, metrics, callbacks, preprocessing, models
from tensorflow.keras.preprocessing import image
from loss import*
from data import DataGenerator
from constants import*
from PIL import Image
from accuracy import*
from keras import callbacks
import matplotlib.pyplot as plt
import cv2
import math
import time
from Video import*

def compile_model():
    model = create_model()

    model.compile(optimizer=optimizers.Adam(), loss=deepball_loss_function, metrics=([deepball_precision]))

    model_checkpoint = callbacks.ModelCheckpoint(filepath='footballcnn.h5', verbose=1)

    train_datagen = DataGenerator(file_path=IMAGE_PATH, config_path=CONFIG_PATH)

    model.fit(x=train_datagen, epochs=2, callbacks=[model_checkpoint])

    model.save_weights('footballcnn.h5')

    """"img = preprocessing.image.load_img('1frame1199.jpg', target_size=(360, 640, 3))
    input1 = preprocessing.image.img_to_array(img)
    input1 = input1.reshape([1, 360, 640, 3])
    input1 = input1 / 255.
    b = model.predict(input1)
    print(b.shape)
    b = b[0, :, :, 0]
    b = np.expand_dims(b, axis=2)
    preprocessing.image.save_img('pred.jpg', b)
    c = np.unravel_index(b.argmax(), b.shape)
    print(c)"""

    return

def predict():

    #customObj = {'deepball_loss_function': deepball_loss_function, 'deepball_precision': deepball_precision}
    #testmodel = keras.models.load_model('./deepballlocal.h5', custom_objects=customObj)
    model = create_model()
    model.load_weights('weights.40.hdf5')

    img = image.load_img('4frame95.jpg', target_size=(INPUT_HEIGHT, INPUT_WIDTH))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.

    pred = model.predict(img, batch_size=1, verbose=1)

    ball_hm = pred[0,:,:,0]

    jpg = np.expand_dims(ball_hm,axis=-1)

    preprocessing.image.save_img('pred.jpg', jpg)

    pos = np.unravel_index(ball_hm.argmax(), ball_hm.shape)

    yp, xp = pos
    print(xp, yp)
    xp = xp * 12
    yp = yp* 12
    print(xp, yp)
    print('Peak ball C : {}'.format(ball_hm[pos]))
    """plt.figure(figsize=(8, 10))
    plt.subplot(3, 1, 1)
    #plt.imshow(bg_cm, cmap='gray')
    #plt.subplot(3, 1, 2)
    plt.imshow(ball_hm, cmap='gray')
    plt.subplot(3, 1, 3)
    cv2.circle(img, (xp, yp), 16, (43, 0, 255), thickness=2)
    plt.imshow(img)
    plt.show()"""
    return

def display_ball_box(image, x, y):
    if x < 0:
        print("*** No ball detected ***")
    else:
        ball_pos = np.array([y,x])
        cv2.circle(image, (x, y), 16, (255,0,0), thickness=2)
    return image

def video(vid, output):
    i = 0
    frame_rate_divider = 1
    model = create_model()
    model.load_weights('weights.40.hdf5')
    while (vid.isOpened()):
        stime = time.time()
        ret, frame = vid.read()
        if ret:
            if i % frame_rate_divider == 0:

                (h, w) = frame.shape[:2]
                img = cv2.resize(frame.astype(np.float32), (INPUT_WIDTH, INPUT_HEIGHT))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0) / 255.
                pred = model.predict(img, batch_size=1, verbose=1)

                ball_hm = pred[0, :, :, 0]
                pos = np.unravel_index(ball_hm.argmax(), ball_hm.shape)
                y, x = pos #height first -> y first
                x = -1 if ball_hm[y,x] < THRESHOLD_VAL else x

                real_x = x*OG_DOWNSAMPLE_SIZE
                real_y = y*OG_DOWNSAMPLE_SIZE

                frame = display_ball_box(frame, real_x, real_y)

                #output.write(frame)
                cv2.imshow('frame', frame)
                i += 1

            print('FPS {:.1f}'.format(1 / (time.time() - stime)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    output.release()
    cv2.destroyAllWindows()


vid = cv2.VideoCapture(VID6_PATH)
size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('Seq6Ballco4.avi',codec,25.0,size)
outputSeg = cv2.VideoWriter('Seq6Seg.avi', codec,25.0,size)

video(vid, output)






