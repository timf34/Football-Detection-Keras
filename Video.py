from constants import*
import time
import cv2
import numpy as np
from model import*

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
                x = -1 if pos[y,x] < THRESHOLD_VAL else x

                real_x = x*OG_DOWNSAMPLE_SIZE
                real_y = y*OG_DOWNSAMPLE_SIZE

                frame = display_ball_box(frame, real_x, real_y)

                write(frame)
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



