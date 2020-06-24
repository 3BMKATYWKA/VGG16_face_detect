# 画像処理プログラム
import os
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf




# def get_batch_list(l, batch_size):
#     # [1, 2, 3, 4, 5,...] -> [[1, 2, 3], [4, 5,..]]
#     return [np.asarray(l[_:_+batch_size]) for _ in range(0, len(l), batch_size)]
#
# def weight_variable(shape):
#     initial = tf.compat.v1.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# def inference(images_placeholder, keep_prob):
#
#     x_image = tf.reshape(images_placeholder, [-1, 32, 32, 3])
#
#     # Convolution layer
#     W_conv1 = weight_variable([5, 5, 3, 32])
#     b_conv1 = bias_variable([32])
#     h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#
#     # Pooling layer
#     h_pool1 = max_pool_2x2(h_conv1)
#
#     # Convolution layer
#     W_conv2 = weight_variable([5, 5, 32, 64])
#     b_conv2 = bias_variable([64])
#     h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#
#     # Pooling layer
#     h_pool2 = max_pool_2x2(h_conv2)
#
#     # Full connected layer
#     W_fc1 = weight_variable([8 * 8 * 64, 1024])
#     b_fc1 = bias_variable([1024])
#     h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
#     h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#     # Dropout
#     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
#     # Full connected layer
#     W_fc2 = weight_variable([1024, len(label_dict)])
#     b_fc2 = bias_variable([len(label_dict)])
#
#     return tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#
# # cv2.cv.CV_FOURCC
# def cv_fourcc(c1, c2, c3, c4):
#     return (ord(c1) & 255) + ((ord(c2) & 255) << 8) + \
#         ((ord(c3) & 255) << 16) + ((ord(c4) & 255) << 24)


def examine_cat_breeds(image, model, train_list):
    # 行列に変換
    img_array = img_to_array(image)
    # 3dim->4dim
    img_dims = np.expand_dims(img_array, axis=0)
    # Predict class（preds：クラスごとの確率が格納された行列(クラス数×1)）
    preds = model.predict(preprocess_input(img_dims))
    preds_reshape = preds.reshape(-1,preds.shape[0])
    # train_list(リスト)を12×1行列に変換
    cat_array = np.array(train_list).reshape(len(train_list),-1)
    # 確率高い順にソートする
    preds_sort = preds_reshape[np.argsort(preds_reshape[:, 0])[::-1]]
    # 確率の降順に合わせて猫の順番も変える
    cat_sort = cat_array[np.argsort(preds_reshape[:, 0])[::-1]]
    # preds_reshape と cat_arrayを結合
    set_result = np.concatenate([cat_sort, preds_sort], 1)
    return set_result


if __name__ == '__main__':

    # path = "./faces/train"
    # dirs = os.listdir(path)
    # dirs = [f for f in dirs if os.path.isdir(os.path.join(path, f))]
    #
    # label_dict = {}
    # i = 0
    #
    # for dirname in dirs:
    #     label_dict[dirname] = i
    #     i += 1
    #
    # names = dirs

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


    # print("setup tensorflow")
    # x = tf.placeholder('float', shape=[None, 32 * 32 * 3])  # 32 * 32, 3 channels
    # keep_prob = tf.placeholder('float')
    # y_conv = inference(x, keep_prob)
    # sess = tf.compat.v1.InteractiveSession()
    # sess.run(tf.compat.v1.global_variables_initializer())
    #
    # print("loading model data")
    # tf.train.Saver().restore(sess, "model_face/model.ckpt")

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
        #img_gray = img
        face_list = cascade.detectMultiScale(img_gray, minSize=(150, 150))

        for (pos_x, pos_y, w, h) in face_list:

            img_face = img[pos_y:pos_y+h, pos_x:pos_x+w]

            img_face = cv2.resize(img_face, (299, 299))

            # test_images = []
            # test_images.append(img_face.flatten().astype(np.float32) / 255.0)
            # test_images = np.asarray(test_images)

            # results = y_conv.eval(feed_dict={ x: test_images, keep_prob: 1.0 })
            results = examine_cat_breeds(img_face, model, train_list)
            text = results[0,0]

            color = (0, 0, 225)
            pen_w = 2
            cv2.putText(img,text,(pos_x,pos_y - 10),font,font_size,(255,255,0))
            cv2.rectangle(img, (pos_x, pos_y), (pos_x+w, pos_y+h), color, thickness = pen_w)

        # フレーム表示
        cv2.imshow(WINDOW_NAME, img)

        # フレーム書き込み
        #rec.write(img)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    # sess.close()

    cv2.destroyAllWindows()

    cap.release()
    #rec.release()