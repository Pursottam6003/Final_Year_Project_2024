# DEPENDENCIES ------>
import pickle
import cv2
import mediapipe as mp
import numpy as np


model_dict = pickle.load(open("./data_small.p", "rb"))
model = model_dict["model"]
capture = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9
)


# labels_dict = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

labels_dict = {
    0: "aA",
    1: "bB",
    2: "cC",
    3: "dD",
    4: "eE",
    5: "fF",
    6: "gG",
    7: "hH",
    8: "iI",
    9: "jJ",
    10: "kK",
    11: "lL",
    12: "mM",
    13: "nN",
    14: "oO",
    15: "pP",
    16: "qQ",
    17: "rR",
    18: "sS",
    19: "tT",
    20: "uU",
    21: "vV",
    22: "wW",
    23: "xX",
    24: "yY",
    25: "zZ",
}
while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = capture.read()

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    ret, frame = capture.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
            x1 = int(min(x_) * W) - 30
            y1 = int(min(y_) * H) - 30
            x2 = int(max(x_) * W) + 30
            y2 = int(max(y_) * H) + 30
            prediction = model.predict([np.asarray(data_aux)])
            # what is its prediction accuracy
            print(prediction)
            # print the predition accuracy
            # print(prediction[0])
            predicted_sign = labels_dict[int(prediction[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(
                frame,
                predicted_sign,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            data_aux = []
            x_ = []
            y_ = []

    cv2.imshow("Handy", frame)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()
