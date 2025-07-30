import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, Button, filedialog, Label, LEFT, RIGHT, Frame
import threading
from datetime import datetime
from collections import deque
import os

# Load pre-trained emotion detection model
model = load_model('F:/Programing/Visual Studio Code/Python/emotion-detection-master/data/emotion1_detection_model_temp.h5')

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Global stop flag
stop_flag = False
emotion_label = None
accuracy_label = None
file_emotion_label = None

# Create folder for saving results
SAVE_FOLDER = "F:/Programing/Visual Studio Code/Python/emotion-detection-master/data/save_result"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Log file for emotion detection
LOG_FILE = os.path.join(SAVE_FOLDER, "emotion_log.txt")

# Function to log emotion detection results
def log_emotion(emotion, timestamp):
    with open(LOG_FILE, "a") as log:
        log.write(f"{timestamp} - Emotion: {emotion}\n")

# Function to save detected image
def save_detected_image(image, emotion_label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_FOLDER, f"{emotion_label}_{timestamp}.jpg")
    cv2.imwrite(filename, image)
    print(f"Saved image: {filename}")

# Function to update emotion display
def update_emotion_display(detected_emotion, file_emotion, is_correct):
    if emotion_label and accuracy_label and file_emotion_label:
        emotion_label.config(text=f"Cảm xúc nhận dạng được: {detected_emotion}")
        file_emotion_label.config(text=f"Cảm xúc kiểm tra: {file_emotion}")
        accuracy_label.config(text=f"Đã nhận dạng: {'Đúng' if is_correct else 'Sai'}")

# Function to extract emotion from file name
def extract_emotion_from_filename(filename):
    # Extract file name without extension
    base_name = os.path.basename(filename)
    emotion_from_file = os.path.splitext(base_name)[0]
    # Get the part before the underscore
    emotion_from_file = emotion_from_file.split("_")[0]
    return emotion_from_file.lower()

# Function to detect emotion from an image
def detect_emotion_from_image(image_path):
    file_emotion = extract_emotion_from_filename(image_path)
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    detected_emotion = "Không xác định"
    is_correct = False

    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi, verbose=0)[0]
        max_index = np.argmax(prediction)
        detected_emotion = emotion_labels[max_index]

        log_emotion(detected_emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        cv2.putText(image, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

        # Check if detected emotion matches file emotion
        is_correct = detected_emotion.lower() == file_emotion

    update_emotion_display(detected_emotion, file_emotion, is_correct)
    save_detected_image(image, detected_emotion)
    cv2.imshow('Emotion Detection - Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to detect emotion from video or webcam
def detect_emotion_from_camera_or_video(video_path=None):
    global stop_flag
    stop_flag = False
    cap = cv2.VideoCapture(video_path if video_path else 0)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(SAVE_FOLDER, f"output_{timestamp}.avi")
    out = None
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if cap.isOpened():
        out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
        print(f"Saving video to {video_filename}")

    emotion_buffer = deque(maxlen=10)
    stable_emotion = None

    # Variables for FPS calculation
    frame_count = 0
    start_time = datetime.now()

    while cap.isOpened():
        if stop_flag:
            break
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi, verbose=0)[0]
            max_index = np.argmax(prediction)
            emotion_label = emotion_labels[max_index]

            log_emotion(emotion_label, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            emotion_buffer.append(emotion_label)

            if emotion_buffer:
                most_common_emotion = max(set(emotion_buffer), key=emotion_buffer.count)
                if emotion_buffer.count(most_common_emotion) > len(emotion_buffer) / 2:
                    stable_emotion = most_common_emotion

        for (x, y, w, h) in faces:
            if stable_emotion:  
                cv2.putText(frame, stable_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = (datetime.now() - start_time).total_seconds()
        if elapsed_time > 0:
            fps_display = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        if out:
            out.write(frame)
        cv2.imshow('Emotion Detection - Real-Time', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

# Function to stop detection
def stop_detection():
    global stop_flag
    stop_flag = True

# Function to open file dialog for selecting an image
def choose_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if file_path:
        threading.Thread(target=detect_emotion_from_image, args=(file_path,)).start()
 
# Function to open file dialog for selecting a video
def choose_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    threading.Thread(target=detect_emotion_from_camera_or_video, args=(file_path,)).start()

# Function to start real-time emotion detection from the webcam
def start_real_time_detection():
    threading.Thread(target=detect_emotion_from_camera_or_video).start()

# Create Tkinter GUI
def create_gui():
    global emotion_label, accuracy_label, file_emotion_label

    root = Tk()
    root.title("Emotion Detection")
    root.geometry("700x500")

    # Tiêu đề ở trên cùng
    title_label = Label(
        root,
        text="Chương trình nhận dạng cảm xúc mặt người",
        font=("Helvetica", 14, "bold"),
        pady=20
    )
    title_label.pack()

    # Khung chứa các chức năng
    function_frame = Frame(root)
    function_frame.pack(pady=20)

    # Dòng chức năng 1
    label1 = Label(function_frame, text="Nhận diện từ ảnh:", font=("Helvetica", 10))
    label1.grid(row=0, column=0, padx=10, pady=5, sticky="w")
    button1 = Button(function_frame, text="Chọn ảnh", command=choose_image, width=15)
    button1.grid(row=0, column=1, padx=10, pady=5)

    # Thông tin hiển thị kết quả
    emotion_label = Label(
        root,
        text="Cảm xúc nhận dạng được: Không xác định",
        font=("Helvetica", 12),
        fg="blue"
    )
    emotion_label.pack(pady=10)

    file_emotion_label = Label(
        root,
        text="Cảm xúc kiểm tra: Không xác định",
        font=("Helvetica", 12),
        fg="orange"
    )
    file_emotion_label.pack(pady=10)

    accuracy_label = Label(
        root,
        text="Đã nhận dạng: Không",
        font=("Helvetica", 12),
        fg="green"
    )
    accuracy_label.pack(pady=10)

    # Dòng chức năng 2
    label2 = Label(function_frame, text="Nhận diện từ video:", font=("Helvetica", 10))
    label2.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    button2 = Button(function_frame, text="Chọn video", command=choose_video, width=15)
    button2.grid(row=1, column=1, padx=10, pady=5)

    # Dòng chức năng 3
    label3 = Label(function_frame, text="Nhận diện từ camera:", font=("Helvetica", 10))
    label3.grid(row=2, column=0, padx=10, pady=5, sticky="w")
    button3 = Button(function_frame, text="Mở camera", command=start_real_time_detection, width=15)
    button3.grid(row=2, column=1, padx=10, pady=5)

    # Nút dừng ở dưới cùng
    stop_button = Button(
        root,
        text="Dừng nhận dạng (Q)",
        command=stop_detection,
        bg="red",
        fg="white",
        width=30
    )
    stop_button.pack(pady=20)

    root.mainloop()

# Run the GUI
if __name__ == "__main__":
    create_gui()
