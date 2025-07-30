import os
import shutil
from sklearn.model_selection import train_test_split

# Đường dẫn tới thư mục CK+ đã giải nén
data_dir = 'F:/Programing/Visual Studio Code/Python/emotion-detection-master/data/datatrain'
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Tạo thư mục lưu ảnh cho train và validation
os.makedirs('F:/Programing/Visual Studio Code/Python/emotion-detection-master/data/train', exist_ok=True)
os.makedirs('F:/Programing/Visual Studio Code/Python/emotion-detection-master/data/validation', exist_ok=True)
for label in emotion_labels:
    os.makedirs(f'F:/Programing/Visual Studio Code/Python/emotion-detection-master/data/train/{label}', exist_ok=True)
    os.makedirs(f'F:/Programing/Visual Studio Code/Python/emotion-detection-master/data/validation/{label}', exist_ok=True)

# Giả sử đã phân loại dữ liệu, bạn có thể sắp xếp vào train và validation
for label in emotion_labels:
    # Đường dẫn tới thư mục của cảm xúc
    label_dir = os.path.join(data_dir, label)
    images = os.listdir(label_dir)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    # Copy ảnh vào thư mục train và validation
    for img in train_images:
        shutil.copy(os.path.join(label_dir, img), f'F:/Programing/Visual Studio Code/Python/emotion-detection-master/data/train/{label}/{img}')
    for img in val_images:
        shutil.copy(os.path.join(label_dir, img), f'F:/Programing/Visual Studio Code/Python/emotion-detection-master/data/validation/{label}/{img}')

print("Dữ liệu đã sẵn sàng cho huấn luyện và validation")
