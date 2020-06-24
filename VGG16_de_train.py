import glob
import numpy as np
import pandas as pd
# kerasのload_imgには手動でpillowのinstall必要
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

x_size = 299
y_size = 299
kind_label = []
cat_img = []
input_shape = (224, 224, 3)

# 識別するクラスのリスト
train_list = []
with open('train_list.txt') as f:
    train_list = [s.strip() for s in f.readlines()]
print('= = train_list = =')
print(train_list)

# データセットのロード
for train_kind in train_list:
    file_list = glob.glob(f'faces/train/{train_kind}/*.jpg')
    for train_file in file_list:
        img = load_img(train_file, target_size=(x_size, y_size))
        x = img_to_array(img)
        x = preprocess_input(x)
        cat_img.append(x)
        kind_label.append(train_kind) 

# 品種ラベルをダミー化
Y_dummy = pd.get_dummies(kind_label)

X_train, X_test, y_train, y_test = train_test_split(
    cat_img, Y_dummy, test_size=0.2, random_state=42)

# model読み込み
model = InceptionV3(weights='imagenet')

# 中間層を出力するモデル
intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[311].output)

# Denseレイヤーを接続
x = intermediate_layer_model.output
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(train_list), activation='softmax')(x)

# 転移学習モデル
transfer_model = Model(inputs=intermediate_layer_model.input, outputs=predictions)

# 一旦全レイヤーをフリーズ
for layer in transfer_model.layers:
    layer.trainable = False

# 最終段のDenseだけ再学習する
transfer_model.layers[312].trainable = True
transfer_model.layers[313].trainable = True

transfer_model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

#転移学習
transfer_model.fit(np.array(X_train), np.array(y_train), epochs=10,
                    validation_data=(np.array(X_test), np.array(y_test)))

#精度確認用に出力（必要に応じて）
loss, acc = transfer_model.evaluate(np.array(X_test), np.array(y_test))
print('Loss {}, Accuracy {}'.format(loss, acc))

transfer_model.save("./model.h5")