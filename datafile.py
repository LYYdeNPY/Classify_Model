import numpy as np
import cv2
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


save_path0 = "./train/airplane"
save_path1 = "./train/automobile"

for i in range(1, 6):
    test = unpickle('cifar-10-batches-py/data_batch_' + str(i))
    for im_index, im_data in enumerate(test[b'data']):
        if test[b'labels'][im_index] == 0:
            im_filename = test[b'filenames'][im_index]
            im_data = np.reshape(im_data, [3, 32, 32])
            im_data = np.transpose(im_data, (1, 2, 0))
            # cv2.imshow("im_filename", cv2.resize(im_data, (200, 200)))
            # cv2.waitKey(0)
            if not os.path.exists(save_path0):
                os.mkdir(save_path0)
            cv2.imwrite("{}/{}".format(save_path0, im_filename.decode("utf-8")), im_data)
        elif test[b'labels'][im_index] == 1:
            im_filename = test[b'filenames'][im_index]
            im_data = np.reshape(im_data, [3, 32, 32])
            im_data = np.transpose(im_data, (1, 2, 0))
            # cv2.imshow("im_filename", cv2.resize(im_data, (200, 200)))
            # cv2.waitKey(0)
            if not os.path.exists(save_path1):
                os.mkdir(save_path1)
            cv2.imwrite("{}/{}".format(save_path1, im_filename.decode("utf-8")), im_data)

test = unpickle('cifar-10-batches-py/test_batch')
for im_index, im_data in enumerate(test[b'data']):
    if test[b'labels'][im_index] == 0:
        im_filename = test[b'filenames'][im_index]
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, (1, 2, 0))
        if not os.path.exists("test/airplane"):
            os.mkdir("test/airplane")
        cv2.imwrite("{}/{}".format("test/airplane", im_filename.decode("utf-8")), im_data)
    elif test[b'labels'][im_index] == 1:
        im_filename = test[b'filenames'][im_index]
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, (1, 2, 0))
        if not os.path.exists("test/automobile"):
            os.mkdir("test/automobile")
        cv2.imwrite("{}/{}".format("test/automobile", im_filename.decode("utf-8")), im_data)
