import cv2
import numpy as np
import serial
from collections import deque
import statistics
import time

# ser0 = serial.Serial('/dev/ttyUSB0', 115200)

cap = cv2.VideoCapture(0)
last_directions = deque(maxlen=5)
response = None
frame_count= 0
W = None
last_response = None
command_counts = {"a": 0, "s": 0, "d": 0, "z": 0, "x": 0}


def reset_command_counts():
    for key in command_counts:
        command_counts[key] = 0

def getContours(inImg, outImg):

    contours, hierarchy = cv2.findContours(inImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 1000:

            cv2.drawContours(outImg, cnt, -1, (255, 0, 255), 7)

            peri = cv2.arcLength(cnt, True)

            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 7:

                x, y, w, h = cv2.boundingRect(approx)

                cv2.rectangle(outImg, (x, y), (x + w, y + h), (0, 255, 0), 5)

                cv2.putText(outImg, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,

                            (0, 255, 0), 2)

                cv2.putText(outImg, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,

                            (0, 255, 0), 2)

                cv2.drawContours(outImg, [approx], -1, (0, 255, 0), 2)

                ellipse = cv2.fitEllipse(approx)

                cv2.ellipse(outImg, ellipse, (255, 255, 0), 2)
                center, axes, angle = ellipse
                angle_text = f"Angle: {int(angle)}"
                cv2.putText(outImg, angle_text, (int(center[0]), int(center[1])),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


                direction = None

                if 80 < angle < 100:

                    xval = list(approx[:, 0, 0])

                    xval_sorted = sorted(xval)

                    x_diff_max = xval_sorted[-1] - xval_sorted[-2]

                    x_diff_min = xval_sorted[1] - xval_sorted[0]

                    if x_diff_max < x_diff_min:

                        if xval_sorted[4] - xval_sorted[1] < 10:
                            direction = "left"
                    else:
                        if xval_sorted[5] - xval_sorted[2] < 10:
                            direction = "right"
                # elif angle <= 10 or angle >= 170:
                #     yval = list(approx[:, 0, 1])
                #     yval_sorted = sorted(yval)
                #     y_diff_max = yval_sorted[-1] - yval_sorted[-2]
                #     y_diff_min = yval_sorted[1] - yval_sorted[0]
                #     if y_diff_max < y_diff_min:
                #         if yval_sorted[4] - yval_sorted[1] < 10:
                #             direction = "up"
                #     else:
                #         if yval_sorted[5] - yval_sorted[2] < 10:
                #             direction = "down"
                if direction:
                    last_directions.append(direction)
                    if len(last_directions) == 5:
                        most_common_direction = statistics.mode(last_directions)
                        # print(last_directions)
                        return most_common_direction
    return None
# frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# kernel = np.ones((5, 5), np.uint8)
while True:
    success, frame = cap.read()
    frame = cv2.resize(frame , (640 , 480))
    frameOut = frame.copy()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameBlur = cv2.GaussianBlur(frameGray, (3, 3), 0)
    framethresh = cv2.adaptiveThreshold(frameBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    num_labels, labels, stats, centroids=cv2.connectedComponentsWithStats(framethresh)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if 50000 > area > 600:
            component = (labels == i).astype(np.uint8) * 255
            result = getContours(component, frameOut)
            # if (result == "up") and (command_counts["x"] < 2):
            #     response = "x"
            #     command_counts["x"] += 1
            #     last_directions.clear()
            #     continue
            # elif (result == "down") and (command_counts["z"] < 2):
            #     response = "z"
            #     command_counts["z"] += 1
            #     last_directions.clear()
            #     continue
            if (result == "right") and (command_counts["d"] < 2):
                response = "d"
                command_counts["d"] += 1
                last_directions.clear()
                continue
            elif (result == "left") and (command_counts["a"] < 2):
                response = "a"
                command_counts["a"] += 1
                last_directions.clear()
                continue
    if response and response != last_response:
        # if response == "w" and last_response != "w":
        #     for i in range(2):
        #         # print("las:",last_response)
        #         ser0.write(response.encode())
        #         print(response)
        #         time.sleep(0.5)

        if response != "w":
            for i in range(4):
                # print("las",last_response)
                # ser0.write(response.encode())
                print(response)
                time.sleep(0.5)
            f = "f"
            # ser0.write(f.encode())
            print(f)
            time.sleep(0.7)
        last_response = response
        reset_command_counts()
    if frame_count % 70 == 0:
        if response == None and W == None :
            f = "f"
            print(f)
            # ser0.write(f.encode())
            for i in range(2):
                W = "w"
                print(W)
                # ser0.write(W.encode())
    frame_count += 1
    cv2.imshow("Image processing", frameOut)
    # print(frame_count)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        response = "f"
        # ser0.write(response.encode())
        break

cap.release()
cv2.destroyAllWindows()