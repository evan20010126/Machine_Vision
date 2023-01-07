# 1. Gauss + different color coordinate + tensorflow + 雪山youtube
# ? 2. gray-level mapping
# ? 3. histogram equalization
# ? 4. high-pass

import cv2
from cv2 import COLOR_RGB2GRAY
from cv2 import IMWRITE_JPEG2000_COMPRESSION_X1000
from cv2 import COLOR_GRAY2RGB
from cv2 import COLOR_GRAY2BGR
from scipy.fft import ifft
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pafy
import time


def check_video_get(videos):
    for v in videos:
        if videos is None:
            print("Videos lost\n")
            exit()
    else:
        print("Videos are obtained.\n")


def gray_level_mapping(img):
    # Gamma correction (gamma is 0.5)

    Look_up_table = [(i / 255.0) ** 0.5 * 255 for i in range(0, 256)]
    # https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/ 看別人如何正規化

    for i in range(0, 256):
        img[i] = Look_up_table[i]

    return img


def histogram_equalization(img):
    # histogram equalization
    # Look_up_table = [0] * 256
    # for i in np.nditer(img):
    #     print(i)
    #     Look_up_table[i] += 1
    # for i in range(1, 256):
    #     Look_up_table[i] += Look_up_table[i-1]
    # for i in range(0, 256):
    #     img[i] = Look_up_table[i]
    # return img
    # Gamma correction (gamma is 0.5)

    Look_up_table = [(i / 255.0) ** 0.5 * 255 for i in range(0, 256)]
    # https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/ 看別人如何正規化

    for i in range(0, 256):
        img[i] = Look_up_table[i]

    return img


def high_pass(img, freq):  # 頻率高的通過
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    cy, cx = fshift.shape[0]/2, fshift.shape[1]/2
    h = np.arange(fshift.shape[0]).reshape((-1, 1)) - cy
    w = np.arange(fshift.shape[1]).reshape((1, -1)) - cx
    freq = freq**2
    fshift = np.where(h**2+w**2 >= freq, fshift, 0)
    ifft_f = np.fft.ifft2(np.fft.fftshift(fshift))
    return ifft_f


def gauss_and_color_coordinate_process(img):
    # color coordinate of HSV 提升飽和度
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = (hsv[:, :, 1].astype(float)*1.5)
    s[s > 255] = 255
    hsv2 = hsv.copy()
    hsv2[:, :, 1] = s
    hsv_result_img = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

    gauss_input_img = hsv_result_img.astype(np.float32)
    gkernel = cv2.getGaussianKernel(9, 9.)
    gfilter2d = np.dot(gkernel, gkernel.T)  # 轉成 dimension=2 的 矩陣
    # 準備 3通道的 kernal
    # kernal型態: 9, 9, 3;  Output: 3 通道
    filters = np.zeros((9, 9, 3, 3))  # 宣告張量
    for i in range(3):
        filters[:, :, i, i] = gfilter2d

    output_img = tf.squeeze(tf.nn.conv2d(
        gauss_input_img[tf.newaxis, ...], filters, 1, 'SAME')).numpy().astype(np.uint8)
    return output_img


videos = list()  # all videos
imgs = [None] * 4

# Process img_1 -> Gaussian filter + tensorflow + 匯入雪山 YouTube video
url = "https://www.youtube.com/watch?v=PHqhEgkGfrs"
ty_video = pafy.new(url, basic=False, gdata=False)
best = ty_video.getbest(preftype="mp4")
videos.append(cv2.VideoCapture(best.url))
videos.append(cv2.VideoCapture("./homework_1_test_video.mp4"))
# writer = cv2.VideoWriter(
# './samplevideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1080, 7040))
# check_video_get(videos)

while True:
    ret, frame = videos[0].read()
    ret1, frame1 = videos[1].read()
    imgs[0] = frame.copy()
    imgs[0] = gauss_and_color_coordinate_process(imgs[0])
    imgs[1] = cv2.cvtColor(frame1.copy(), COLOR_RGB2GRAY)
    imgs[1] = gray_level_mapping(imgs[1])
    imgs[2] = cv2.cvtColor(frame1.copy(), COLOR_RGB2GRAY)
    imgs[2] = histogram_equalization(imgs[2])
    imgs[3] = cv2.cvtColor(frame1.copy(), COLOR_RGB2GRAY)
    imgs[3] = high_pass(imgs[3], 50)
    HIGHT_OF_ANSWER = max(
        imgs[0].shape[0], imgs[1].shape[0], imgs[2].shape[0], imgs[3].shape[0])

    WIDTH_OF_IMAGE = imgs[0].shape[1] + \
        imgs[1].shape[1] + imgs[2].shape[1] + imgs[3].shape[1]
    answer_img = np.zeros((HIGHT_OF_ANSWER, WIDTH_OF_IMAGE, 3), dtype=np.uint8)

    answer_img[0:imgs[0].shape[0], :imgs[0].shape[1], :] = imgs[0]

    # answer_img[0:imgs[1].shape[0], imgs[0].shape[1]:imgs[1].shape[1],:] = convert_img = cv2.cvtColor(imgs[1], COLOR_GRAY2BGR)

    for c in range(0, 3):
        for i in range(0, imgs[1].shape[0]):
            for j in range(imgs[0].shape[1], imgs[1].shape[1]):
                answer_img[i, j, c] = imgs[1][i][j]

    # answer_img[0:imgs[2].shape[0], imgs[1].shape[1]:imgs[2].shape[1], :] = cv2.cvtColor(imgs[2], COLOR_GRAY2BGR)
    for c in range(0, 3):
        for i in range(0, imgs[2].shape[0]):
            for j in range(imgs[1].shape[1], imgs[2].shape[1]):
                answer_img[i, j, c] = imgs[2][i][j]

    # answer_img[0:imgs[3].shape[0], imgs[2].shape[1]:, :] = cv2.cvtColor(imgs[3], COLOR_GRAY2BGR)
    for c in range(0, 3):
        for i in range(0, imgs[3].shape[0]):
            for j in range(imgs[2].shape[1], imgs[3].shape[1]):
                answer_img[i, j, c] = imgs[3][i][j]

    # answer_img = cv2.resize(answer_img, (3508, 540))
    # cv2.imshow('youtube live', answer_img)
    # writer.write(answer_img)
    k = cv2.waitKey(50)
    if k == 27:
        break

# writer.release()
cv2.destroyAllWindows()
videos[0].release()
videos[1].release()
