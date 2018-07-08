from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy
import os
import tensorflow as tf
import torch
# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

x = torch.rand(5,3)
print(x)


df_pre = pd.read_csv('./dataset/wine.csv', header=None)
df = df_pre.sample(frac=0.15)

dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

# 모델 저장 폴더 만들기
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
   os.mkdir(MODEL_DIR)

modelpath="./model/study_180606_2.hdf5"

# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=50)


model.fit(X, Y, validation_split=0.2, epochs=1000, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])
#model.fit(X_train, Y_train, epochs=1000, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])

print("\n Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
prediction = model.predict(X_test)
result = [round(x[0]) for x in prediction]
print(result)