# 示範怎麼把高斯的kernal擺上去
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 9x9的矩陣來放 吐一塊9x9的高斯kernal 呼叫opencv幫我填高斯的kernal 目的: 三個通道每個通道分別做高斯模糊
gkernel = cv2.getGaussianKernel(9, 9.)
gfilter2d = np.dot(gkernel, gkernel.T)
print(gkernel)
