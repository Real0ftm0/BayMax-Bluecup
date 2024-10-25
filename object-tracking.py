import cv2
import numpy as np
import time
import serial

# ser0 = serial.Serial('/dev/ttyUSB0', 115200)
response = None

backSub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=80, detectShadows=False)
tracker = None
motion_contour = None

frame_count = 0
last_response = None
last_time = None

command_counts = {"a": 0, "s": 0, "d": 0, "z": 0, "x": 0}

def reset_command_counts():
    for key in command_counts:
        command_counts[key] = 0

def find_motion(frame, backSub):
    fgMask = backSub.apply(frame)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_motion_area = 0
    motion_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 <= area <= 60000:
            x, y, w, h = cv2.boundingRect(contour)
            if area > max_motion_area:
                max_motion_area = area
                motion_contour = (x, y, w, h)
    cv2.imshow('Tracking2', fgMask)
    return motion_contour

def track_object(tracker, frame, frame_center_x, frame_center_y, radius):
    global last_response, last_time
    success, bbox = tracker.update(frame)
    response = None

    if success:
        x, y, w, h = tuple(map(int, bbox))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        object_center_x = x + w // 2
        object_center_y = y + h // 2

        distance = np.sqrt((object_center_x - frame_center_x) ** 2 + (object_center_y - frame_center_y) ** 2)

        if object_center_y <= 60 and command_counts["x"] < 2:
            response = "x"
            command_counts["x"] += 1

        elif object_center_y >= frame.shape[0] - 60 and command_counts["z"] < 2:
            response = "z"
            command_counts["z"] += 1

        elif distance > radius:
            if object_center_x < frame_center_x and command_counts["a"] < 2:
                response = "a"
                command_counts["a"] += 1

            elif object_center_x > frame_center_x and command_counts["d"] < 2:
                response = "d"
                command_counts["d"] += 1
        else:
            if last_response != "w":
                response = "w"
                reset_command_counts()

    else:
        if last_response != "f":
            response = "f"
        tracker = None

    if response and response != last_response:
        if response == "f" and last_response != "f":
            for i in range(1):
                print(response)
                # ser0.write(response.encode())
                time.sleep(0.3)

        elif response != "f":
            for i in range(3):
                print(response)
                # ser0.write(response.encode())
                time.sleep(0.3)
        last_response = response

    return tracker

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2
    radius = max(frame_center_x, frame_center_y) // 2

    if tracker is None and frame_count % 5 == 0:
        reset_command_counts()
        motion_contour = find_motion(frame, backSub)
        if motion_contour:
            x, y, w, h = motion_contour
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (x, y, w, h))
            motion_contour = None
    elif tracker is not None:
        tracker = track_object(tracker, frame, frame_center_x, frame_center_y, radius)

    cv2.circle(frame, (frame_center_x, frame_center_y), radius, (255, 255, 255), 2)
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        response = "f"
        # ser0.write(response.encode())
        break

cap.release()
cv2.destroyAllWindows()