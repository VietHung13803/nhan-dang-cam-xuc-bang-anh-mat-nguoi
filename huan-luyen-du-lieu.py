import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ

# Khởi tạo mô hình CNN
model = Sequential()

# Thêm các lớp tích chập và lớp pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))  # Ảnh grayscale kích thước 48x48
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 cảm xúc khác nhau

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Khởi tạo ImageDataGenerator để tiền xử lý dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2, shear_range=0.2, brightness_range=[0.8, 1.2])
validation_datagen = ImageDataGenerator(rescale=1./255)

# Chuẩn bị dữ liệu huấn luyện và validation
train_generator = train_datagen.flow_from_directory(
    'F:/Programing/Visual Studio Code/Python/emotion-detection-master/data/train',  # Thay bằng đường dẫn tới dữ liệu huấn luyện
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'F:/Programing/Visual Studio Code/Python/emotion-detection-master/data/validation',  # Thay bằng đường dẫn tới dữ liệu validation
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 64,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 64,
    callbacks=[early_stopping]
)

# Vẽ biểu đồ accuracy của train và validation
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linestyle='--')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
plt.title('Train vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Lưu mô hình đã huấn luyện
model.save('F:/Programing/Visual Studio Code/Python/emotion-detection-master/data/emotion1_detection_model.h5')

# Đánh giá mô hình trên tập validation
loss, accuracy = model.evaluate(validation_generator)

# In kết quả đánh giá ra màn hình
print("Kết quả đánh giá trên tập validation:")
print(f"Loss (mất mát): {loss}")
print(f"Accuracy (độ chính xác): {accuracy}")
