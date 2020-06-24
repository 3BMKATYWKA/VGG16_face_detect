# 画像処理プログラム
import os
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import cv2

def examine_cat_breeds(image, model, train_list):
    # 行列に変換
    img_array = img_to_array(image)
    # 3dim->4dim
    img_dims = np.expand_dims(img_array, axis=0)
    # Predict class（preds：クラスごとの確率が格納された行列(クラス数×1)）
    preds = model.predict(preprocess_input(img_dims))
    # print('preds')
    # print(preds)
    preds_reshape = preds.reshape(-1,preds.shape[0])
    # print("preds_reshape")
    # print(preds_reshape)
    # train_list(リスト)を12×1行列に変換
    # print('train_list')
    # print(train_list)
    cat_array = np.array(train_list).reshape(len(train_list),-1)
    # print('cat_array')
    # print(cat_array)
    # 確率高い順にソートする
    preds_sort = preds_reshape[np.argsort(preds_reshape[:, 0])[::-1]]
    # 確率の降順に合わせて猫の順番も変える
    cat_sort = cat_array[np.argsort(preds_reshape[:, 0])[::-1]]
    # print("cat_sort")
    # print(cat_sort)
    # preds_reshape と cat_arrayを結合
    set_result = np.concatenate([cat_sort, preds_sort], 1)
    # print("set_result")
    # print(set_result)
    return set_result


if __name__ == '__main__':

    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    WINDOW_NAME = "detect"
    #FILE_NAME = "detect.avi"

    DEVICE_ID = 0

    # 分類器の指定
    cascade_file = "haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # 保存ビデオファイルの準備
    #rec = cv2.VideoWriter(FILE_NAME, cv_fourcc('X', 'V', 'I', 'D'), FRAME_RATE, (width, height), True)

    # ウィンドウの準備
    cv2.namedWindow(WINDOW_NAME)

    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 1.5

    model = load_model('model.h5')
    train_list = []
    with open('train_list.txt') as f:
        train_list = [s.strip() for s in f.readlines()]
    print('= = train_list = =')
    print(train_list)

    # 変換処理ループ
    while end_flag == True:

        img = cv2.flip(c_frame, 1)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))

        for (pos_x, pos_y, w, h) in face_list:

            img_face = img[pos_y:pos_y+h, pos_x:pos_x+w]

            img_face = cv2.resize(img_face, (299, 299))

            results = examine_cat_breeds(img_face, model, train_list)
            print('results')
            print(results)
            text = results[0,0]

            color = (0, 0, 225)
            pen_w = 2
            cv2.putText(img,text,(pos_x,pos_y - 10),font,font_size,(255,255,0))
            cv2.rectangle(img, (pos_x, pos_y), (pos_x+w, pos_y+h), color, thickness = pen_w)

        # フレーム表示
        cv2.imshow(WINDOW_NAME, img)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()
    cv2.destroyAllWindows()
    cap.release()