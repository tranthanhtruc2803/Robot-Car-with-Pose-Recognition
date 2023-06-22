import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import onnxruntime as rt
import time
import serial

u=0
last_label = 5
label = "x"
n_time_steps = 20
lm_list = []

#Action = [
#            "FORWARD",
#            "BACKWARD",
#            "TURN LEFT",
#            "TURN RIGHT",
#            "STOP"
#         ]
Action = [
            "f",
            "b",
            "l",
            "r",
            "x"
         ]
# Start MediaPipe library
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

output_path = "/home/tttruc/LSTM20.onnx"
output_names = ['dense_2']
providers = ['CPUExecutionProvider']

model = rt.InferenceSession('LSTM20.onnx', providers=providers)

# Initialize UART
print("UART Initialize...")

serial_port = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE
)

time.sleep(1)


# Camera
cap = cv2.VideoCapture("/dev/video0")
# used to record the time when we processed last frame
prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (221,160,221), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (104,34,139)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    lm_list = lm_list.astype('float32') # astype(np.float32)
    results = model.run(output_names, {"input": lm_list})
    predict = results[0]
    #print(predict)
    predict = predict.tolist()
    predict = predict[0]
    action = predict.index(max(predict))
    #print(max(predict))
    if max(predict) < 0.7:
        label = "x"
        return label
    label = Action[action]
    print(label)
    return label


i = 0
warmup_frames = 1

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # Draw landmark
    if results.pose_landmarks:
        c_lm = make_landmark_timestep(results)
        img = draw_landmark_on_image(mpDraw, results, img)
        lm_list.append(c_lm)
    # Interfere
    if len(lm_list) == n_time_steps:
        t = threading.Thread(target=detect, args=(model, lm_list,))
        t.start()
        lm_list = []

    #img = draw_class_on_image(label, img)
    serial_port.write(label.encode())

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
