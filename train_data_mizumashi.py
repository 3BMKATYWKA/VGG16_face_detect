import os
import glob
from PIL import Image
from datetime import datetime as dt
import cv2
import numpy as np


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


train_list = []
with open('train_list.txt') as f:
    train_list = [s.strip() for s in f.readlines()]
print('= = train_list = =')
print(train_list)

# 回転
rotate = [15, 30, 45, 60, 300, 315, 330]

# ハイコントラスト、ローコントラスト
min_table = 50
max_table = 205
diff_table = max_table - min_table
LUT_HC = np.arange(256, dtype = 'uint8' )
LUT_LC = np.arange(256, dtype = 'uint8' )
# ハイコントラストLUT作成
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table
for i in range(max_table, 255):
    LUT_HC[i] = 255
# ローコントラストLUT作成
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255

LUT_G1 = np.arange(256, dtype = 'uint8' )
LUT_G2 = np.arange(256, dtype = 'uint8' )
gamma1 = 0.75
gamma2 = 1.5
for i in range(256):
    LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
    LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

for train_kind in train_list:
    OUTPUT_DIR = '{}/faces/train/{}'.format(os.getcwd(), train_kind)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # print(OUTPUT_DIR)
    # print(train_kind)
    file_list = glob.glob(f'faces/train/{train_kind}/*.jpg')
    for train_file in file_list:
        # print(train_file)
        name = basename_without_ext = os.path.splitext(os.path.basename(train_file))[0]
        img = Image.open(train_file)  # 画像読み込み
        filename = '{}-flip.jpg'.format(name)
        # 左右反転
        img_flr = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_flr = pil2cv(img_flr)
        cv2.imwrite('{}/{}'.format(OUTPUT_DIR, filename), img_flr)
        # 回転
        for d in rotate:
            img_rotate = img.rotate(d)
            img_rotate = pil2cv(img_rotate)
            filename = '{}-{}.jpg'.format(name, d)
            cv2.imwrite('{}/{}'.format(OUTPUT_DIR, filename), img_rotate) # 画像保存
        # ハイコントラスト
        img = cv2.imread(train_file, 1)
        # img = pil2cv(img)
        high_cont_img = cv2.LUT(img, LUT_HC)
        filename = '{}-hi.jpg'.format(name)
        cv2.imwrite('{}/{}'.format(OUTPUT_DIR, filename), high_cont_img)  # 画像保存
        # ローコントラスト
        low_cont_img = cv2.LUT(img, LUT_LC)
        filename = '{}-low.jpg'.format(name)
        cv2.imwrite('{}/{}'.format(OUTPUT_DIR, filename), low_cont_img)  # 画像保存