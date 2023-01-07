# 1. Gauss + different color coordinate + tensorflow + 雪山youtube
# ? 2. gray-level mapping
# ? 3. histogram equalization
# ? 4. high-pass

from unittest import case
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
from PIL import Image, ImageDraw, ImageFont


def gray_level_mapping(img):
    # Gamma correction (gamma is 0.5)

    Look_up_table = [(i / 255.0) ** 0.5 * 255 for i in range(0, 256)]
    # https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/ 看別人如何正規化

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = Look_up_table[img[i][j]]

    return img


def histogram_equalization(img):
    # histogram equalization

    # I implement:
    # produce table
    # Look_up_table = [0] * 256
    # for i in np.nditer(img):
    #     print(i)
    #     Look_up_table[i] += 1
    # for i in range(1, 256):
    #     Look_up_table[i] += Look_up_table[i-1]

    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         img[i][j] = Look_up_table[img[i][j]]

    # alternatives: opencv-function
    # https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html
    img = cv2.equalizeHist(img)
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


def cv2ImgAddText(img, text, pos, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判斷是否OpenCV圖片類型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "C:\\Windows\Fonts\\simsun.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


urls = [
    ("https://www.youtube.com/watch?v=PHqhEgkGfrs", "雪山"),
    ("./homework_1_test_video.mp4", "Video1"),
    ("./homework_1_test_video.mp4", "Video2"),
    ("./homework_1_test_video.mp4", "Video3")
]

videos = []
fps = 100

for url in urls:
    try:
        if url[0][:5] == "https":
            video = pafy.new(url[0])
            best = video.getbest(preftype="mp4")
            video = cv2.VideoCapture(best.url)
            if video is not None:
                fps = min([fps, int(video.get(cv2.CAP_PROP_FPS))])
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print('Frame rate:', fps, 'Frame width:',
                      width, 'Frame height:', height)
                videos.append((video, url[1]))
        else:
            video = cv2.VideoCapture(url[0])
            videos.append((video, url[1]))
    except:
        print("no")
        pass


CASE0 = 0
CASE1 = 1
CASE2 = 2
CASE3 = 3

count = CASE0
if videos:
    cv2.namedWindow('00957202_homework1')
    big_frame = np.zeros(((len(videos)+3)//4*180, (4 if len(videos)
                         >= 4 else len(videos))*320, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(
        './samplevideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (big_frame.shape[1], big_frame.shape[0]))

    go_on = True
    while go_on:
        for i, video in enumerate(videos):
            grabbed, frame = video[0].read()
            if grabbed == False:
                go_on = False
            if grabbed:
                frame = cv2.resize(frame, (320, 180))
                if count == CASE0:
                    frame = gauss_and_color_coordinate_process(frame)
                elif count == CASE1:
                    frame = cv2.cvtColor(frame, COLOR_RGB2GRAY)
                    frame = gray_level_mapping(frame)
                elif count == CASE2:
                    frame = cv2.cvtColor(frame, COLOR_RGB2GRAY)
                    frame = histogram_equalization(frame)
                    pass
                elif count == CASE3:
                    frame = cv2.cvtColor(frame, COLOR_RGB2GRAY)
                    frame = high_pass(frame, 20)
                else:
                    pass

                row_idx = i//4
                col_idx = i % 4

                if count == CASE0:
                    big_frame[row_idx*frame.shape[0]:(row_idx+1)*frame.shape[0],
                              col_idx*frame.shape[1]:(col_idx+1)*frame.shape[1], :] = frame[:, :, :]
                # answer_img[0:imgs[3].shape[0], imgs[2].shape[1]:, :] = cv2.cvtColor(imgs[3], COLOR_GRAY2BGR)
                elif count == CASE1 or count == CASE2 or count == CASE3:
                    for c in range(0, 3):
                        f1 = -1
                        for i in range(row_idx*frame.shape[0], (row_idx+1)*frame.shape[0]):
                            f1 += 1
                            f2 = -1
                            for j in range(col_idx*frame.shape[1], (col_idx+1)*frame.shape[1]):
                                f2 += 1
                                big_frame[i, j, c] = frame[f1][f2]

                count += 1
                count %= 4
        cv2.imshow('00957202_homework1', big_frame)
        writer.write(big_frame)
        print("Running: " + str(count))

    cv2.destroyAllWindows()
    for video in videos:
        video[0].release()
    writer.release()
