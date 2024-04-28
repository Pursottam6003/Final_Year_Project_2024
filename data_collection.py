# DEPENDENCIES ------>
import os
import cv2


DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

num_classes = 16
dataset_size = 200

cap = cv2.VideoCapture(0)
for i in range(num_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(i))):
        os.makedirs(os.path.join(DATA_DIR, str(i+10)))
    print("Collecting data")
    done = False

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, "Press '1' ", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) == ord("1"):
            break

    count = 0
    while count < dataset_size:
        ret, frame = cap.read()
        cv2.imshow("Frame", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(i+10), "{}.jpg".format(count)), frame)
        count += 1
        print("Images saved: ", count)

cap.release()
cv2.destroyAllWindows()
