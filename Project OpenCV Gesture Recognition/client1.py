import cv2
import mediapipe as mp
from mediapipe.tasks import python
import paho.mqtt.client as mqtt
import random
import time

# MQTT Configuration
mqtt_broker = "broker.emqx.io"
mqtt_port = 1883
topic_publish = "communicationhandgesture/client1"
topic_subscribe = "communicationhandgesture/client2"
client_id = f"python-mqtt-{random.randint(0, 1000)}"

# Inisialisasi variabel
gesture_name = "none"
gesture_score = 0
statusLed = 0

publish_time = time.time()
time_elapsed = 0.0


gesture_buffer = []

def get_most_common_gesture(gesture_buffer):
    if len(gesture_buffer) > 0:
        return max(set(gesture_buffer), key=gesture_buffer.count)
    return "none"

# On Message MQTT Function
def on_message(client, userdata, message):
    global statusLed
    global time_elapsed
    print("Received message '" + str(message.payload) +
          "' on topic '" + message.topic + "'")
    if message.topic == topic_subscribe:
        statusLed = int(message.payload.decode("utf-8"))
        current_time = time.time()
        time_elapsed = current_time - publish_time
        print(f"Waktu dari publish hingga on_message: {time_elapsed} detik")

# Inisialisasi MQTT Client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect
client.on_message = on_message
client.connect(mqtt_broker, mqtt_port)
client.loop_start()

# Inisialisasi variabel warna
green = (138, 188, 41)
red = (119, 70, 232)
black = (178, 123, 117)
white = (255, 255, 255)
color_landmark = (0, 0, 255)
color_connection = (0, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX

# Inisialisasi variabel
publish_on = False
publish_off = False
prev_time = time.time()

# Hitung FPS
def calculate_fps(prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return current_time, fps

# Inisialisasi model gesture recognition
model_path = "gesture_recognizer.task"
base_options = python.BaseOptions(model_asset_path=model_path)
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Get Result Callback Function for Gesture Recognition
def get_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global gesture_name, gesture_score
    if result is not None and any(result.gestures):
        for single_hand_gesture_data in result.gestures:
            gesture_name = single_hand_gesture_data[0].category_name
            gesture_score = single_hand_gesture_data[0].score

# Inisialisasi Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result)
recognizer = GestureRecognizer.create_from_options(options)

# Inisialisasi MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65)

# Inisialisasi OpenCV dan variabel timestamp
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
timestamp = 0

# Looping untuk membaca frame dari kamera
while True:
    # Proses frame dari kamera
    try:
        ret, frame = cap.read()
        if not ret:
            break

        # Proses frame dan hasil pengenalan gestur
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        np_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hitung FPS
        prev_time, fps = calculate_fps(prev_time)

        if not results.multi_hand_landmarks:
            gesture_name = "none"
            gesture_score = 0.0
            
        # Proses hasil pengenalan gestur
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = frame.shape

                # Gambar landmark tangan
                draw_spec_landmark = mp_drawing.DrawingSpec(
                    color=black, thickness=1, circle_radius=2)
                draw_spec_connection = mp_drawing.DrawingSpec(
                    color=color_connection, thickness=1)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, draw_spec_landmark, draw_spec_connection)

                # Crop gambar tangan
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=np_array)

                # Proses pengenalan gestur
                results = recognizer.recognize_async(mp_image, timestamp)
                timestamp += 1

        # Tampilkan FPS pada frame
        cv2.putText(frame, f"FPS: {fps:.2f} Huruf: {gesture_name}",
                    (50, 50), font, 1, black, 2, cv2.LINE_AA)

        # Publikasikan hasil gesture ke topik MQTT
         # Tambahkan hasil gesture ke buffer
        if gesture_name != "none":
            gesture_buffer.append(gesture_name)
        
        # Jika buffer sudah berisi 50 data, hitung gesture paling umum
        if len(gesture_buffer) >= 50:
            most_common_gesture = get_most_common_gesture(gesture_buffer)
            
            # Ganti 'none' dengan spasi
            if most_common_gesture == "none":
                most_common_gesture = " "
            
            # Publikasikan hasil gesture ke topik MQTT
            client.publish(topic_publish, payload=most_common_gesture)
            print(f"Gesture published: {most_common_gesture}")
            
            # Kosongkan buffer setelah pengiriman
            gesture_buffer = []

        cv2.imshow('Client 1', frame)

        # Break loop jika tombol ESC ditekan
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Tangkap error jika terjadi
    except Exception as e:
        print(f"Error: {e}")

# Tutup koneksi MQTT dan OpenCV
client.loop_stop()
cap.release()
cv2.destroyAllWindows()
