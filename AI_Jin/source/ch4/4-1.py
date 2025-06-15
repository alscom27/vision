from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


#이미지 데이터 제네레이터를 만들기
train_generator = ImageDataGenerator(rotation_range=10, #랜덤 회전 각도
                                     zoom_range=0.2,  # 줌인 비율
                                     horizontal_flip=True,  # 가로 반전
                                     rescale=1/255)  # 정규화 작업

train_dataset = train_generator.flow_from_directory(directory='fer2013/train',
                                                    target_size=(48, 48),  
                                                    class_mode='categorical',
                                                    batch_size=16,  
                                                    shuffle=True,  
                                                    seed=10)

# 테스트 제네레이터 만들기
test_generator = ImageDataGenerator(rescale=1/255)
test_dataset = test_generator.flow_from_directory(directory='fer2013/test',
                                                  target_size=(48, 48),
                                                  class_mode='categorical',
                                                  batch_size=1,
                                                  shuffle=False,
                                                  seed=10)

# FER 데이터셋은 7가지의 감정 분류를 가짐, 디텍터는 32개 이미지는 48X48 셋을 가지고 있음
num_classes = 7
num_detectors = 32
width, height = 48, 48

# 이를 반영하는 망을 만들어 보도록 하자
network = Sequential()
network.add(Conv2D(filters=num_detectors, kernel_size=3, activation='relu', padding='same', input_shape=(width, height, 3)))
network.add(BatchNormalization())
network.add(Conv2D(filters=num_detectors, kernel_size=3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))
network.add(Conv2D(2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))
network.add(Conv2D(2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))
network.add(Conv2D(2*2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*2*num_detectors, 3, activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))
network.add(Flatten())
network.add(Dense(2*2*num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))
network.add(Dense(2*num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))
network.add(Dense(num_classes, activation='softmax'))
network.summary()


# 이제 네트워크를 컴파일 하기
network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습시 오버피팅을 방지하기 위해 정확도를 모니터링하여 조기에 학습을 종료시키는 인스턴스 정의 
monitor_val_acc = EarlyStopping(monitor='val_accuracy', patience=5)
# loss 제일 낮을 때 가중치 저장
filename = 'emotion_best.h5'
checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                             verbose=1,            # 로그를 출력합니다
                             save_best_only=True   # 가장 best 값만 저장합니다
                            )
epochs = 70
history= network.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[checkpoint, monitor_val_acc]).history
print('학습종료!')
network.save(filename)

score = network.evaluate(test_dataset)
print('Test loss:', score[0]) 
print('Test accuracy:', score[1]*100)

import pandas as pd #데이터를 분석 및 조작하기 위한 소프트웨어 라이브러리
import matplotlib.pyplot as plt #다양한 데이터를 많은 방법으로 도식화 할 수 있도록 하는 파이썬 라이브러리

plt.figure(figsize=(16,5)) 
# 만들어진 모델에 대해 train dataset과 validation dataset의 loss 를 그래프로 표현
plt.subplot(1, 2, 1) 
plt.plot(history['loss']) 
plt.plot(history['val_loss']) 
plt.title('model loss') 
plt.ylabel('loss') 
plt.xlabel('epoch') 
plt.legend(['train', 'validation'], loc='upper left')

# 만들어진 모델에 대해 train dataset과 validation dataset의 accuracy 를 그래프로 표현
plt.subplot(1, 2, 2) 
plt.plot(history['accuracy']) 
plt.plot(history['val_accuracy']) 
plt.title('model accuracy') 
plt.ylabel('accuracy') 
plt.xlabel('epoch') 
plt.legend(['train', 'validation'], loc='upper left')
