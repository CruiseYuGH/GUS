import numpy as np
import cv2
import os ,torch
import random
import time
from sklearn.metrics import f1_score

def get_now_time():
    now =  time.localtime()
    now_time = time.strftime("%Y_%m_%d_%H_%M_%S", now)
    return now_time
    
def mean_f1(preds,targets):
    f1=[]
    temp_exp_pred = np.array(preds)
    temp_exp_target = np.array(targets)
    temp_exp_pred = torch.eye(6)[temp_exp_pred]
    temp_exp_target = torch.eye(6)[temp_exp_target]
    for i in range(0, 6):
        exp_pred = temp_exp_pred[:, i]
        exp_target = temp_exp_target[:, i]
        f1.append(f1_score(exp_pred, exp_target))
    print(f1)
    #logger.info(str(f1))
    return np.mean(f1)

def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def color2gray(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_img_3d = image_array.copy()
    gray_img_3d[:, :, 0] = gray
    gray_img_3d[:, :, 1] = gray
    gray_img_3d[:, :, 2] = gray
    return gray_img_3d


def data_augment(image, brightness):
    factor = 1.0 + random.uniform(-1.0*brightness, brightness)
    table = np.array([(i / 255.0) * factor * 255 for i in np.arange(0, 256)]).clip(0,255).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image