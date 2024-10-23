import time
import serial
import cv2
import numpy as np



action_command = {"w": 0, "a": 0, "d": 0, "f": 0}


def reset_command_counts():
    for key in action_command:
        action_command[key] = 0


def calculate_angle(line):
    x1, y1, x2, y2 = line[0]
    return np.arctan2((y2 - y1), (x2 - x1))


def line_length(line):
    x1, y1, x2, y2 = line[0]
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def distance_between_parallel_lines(line1, line2):
    # محاسبه شیب
    x1_1, y1_1, x2_1, y2_1 = line1[0]
    x1_2, y1_2, x2_2, y2_2 = line2[0]

    slope = (y2_2 - y1_2) / (x2_2 - x1_2) if (x2_2 - x1_2) != 0 else float('inf')
    c1 = y1_1 - slope * x1_1
    c2 = y1_2 - slope * x1_2

    # محاسبه فاصله
    distance = abs(c2 - c1) / np.sqrt(1 + slope ** 2)
    return distance


def draw_parallel_and_center(image):

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    high_pass_kernel = np.array([[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, -1]])

    # اعمال فیلتر بالاگذر
    high_pass_filtered = cv2.filter2D(blur, -1, high_pass_kernel)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(high_pass_filtered)[0]
    if lines is None:
        return image

    lines = np.int32(lines)
    angles = np.array([calculate_angle(line) for line in lines])

    parallel_threshold = np.pi / 36

    parallel_lines = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if np.any(abs(angles[i] - angles[j]) < parallel_threshold):
                parallel_lines.append((lines[i], lines[j]))

    max_length = 0
    longest_parallel_pair = None
    for line1, line2 in parallel_lines:
        length1 = line_length(line1)
        length2 = line_length(line2)
        total_length = length1 + length2
        if total_length > max_length:
            max_length = total_length
            longest_parallel_pair = (line1, line2)

    if longest_parallel_pair is not None:
        line1, line2 = longest_parallel_pair
        x1_1, y1_1, x2_1, y2_1 = line1[0]
        x1_2, y1_2, x2_2, y2_2 = line2[0]

        distance = distance_between_parallel_lines(line1, line2)

        for line in longest_parallel_pair:
            length = line_length(line)
            if length > 100:
                cv2.line(image, (x1_1, y1_1), (x2_1, y2_1), (0, 0, 255), 2)
                cv2.line(image, (x1_2, y1_2), (x2_2, y2_2), (0, 0, 255), 2)

        # محاسبه و رسم مرکز نقاط متناظر
        center_start_x = (x1_1 + x1_2) // 2
        center_start_y = (y1_1 + y1_2) // 2
        center_end_x = (x2_1 + x2_2) // 2
        center_end_y = (y2_1 + y2_2) // 2

        delta_x = center_end_x - center_start_x
        delta_y = center_end_y - center_start_y
        angle_pipe_center = np.arctan2(delta_y, delta_x)  # زاویه خط بین دو نقطه لوله

        # زاویه خط عمودی وسط تصویر
        angle_vertical_center = np.pi / 2  # خط عمودی در مرکز
        angle_difference = angle_pipe_center - angle_vertical_center
        angle_difference_degrees = np.degrees(angle_difference)  # تبدیل به درجه
        angle_difference_degrees = np.int32(angle_difference_degrees)
        if angle_difference_degrees < -90:
            angle_difference_degrees = angle_difference_degrees + 180
        elif angle_difference_degrees > 90:
            angle_difference_degrees = 180 - angle_difference_degrees
        if abs(angle_difference_degrees) < 10:
            angle_difference_degrees = 0
        # وقتی زاویه مثبته حرکت به راسته و وقتی منفیه حرکت به چپ
        if angle_difference_degrees < 0:
            print(angle_difference_degrees)
            for i in range(abs(angle_difference_degrees) // 10):
                response = "d"
                ser0.write(response.encode())
                print(response)
        elif angle_difference_degrees > 0:
            print(angle_difference_degrees)
            for i in range(angle_difference_degrees // 10):
                response = "a"
                ser0.write(response.encode())
                print(response)
        elif action_command["w"] < 2:
            response = "w"
            ser0.write(response.encode())
            print(angle_difference_degrees)
            print(response)
            action_command["w"] += 1

        else:
            response = "f"
            ser0.write(response.encode())
            time.sleep(1)
            reset_command_counts()
            print(response)

        # نمایش زاویه
        angle_text1 = f"Angle: {angle_difference_degrees:.2f} degrees"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, angle_text1, (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # رسم مرکز
        cv2.circle(image, (center_start_x, center_start_y), 5, (0, 255, 0), -1)
        cv2.circle(image, (center_end_x, center_end_y), 5, (0, 255, 0), -1)

        # رسم خط اتصال بین مراکز
        cv2.line(image, (center_start_x, center_start_y), (center_end_x, center_end_y), (0, 255, 0), 2)

    return image


cap = cv2.VideoCapture("data/Screencast from 2024-10-10 15-32-25.webm")
ser0 = serial.Serial('/dev/ttyUSB0', 115200)
for i in range(7):
    response = "f"
    ser0.write(response.encode())
    print(response)
    time.sleep(2)

for i in range(1):
    response = "w"
    # ser0.write(response.encode())
    print(response)

response = None

frame_counter = 0
process_every_n_frames = 5  # هر 5 فریم یکبار پردازش شود

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1

    if frame_counter % 15 == 0:
        frame = draw_parallel_and_center(frame)  # فریم پردازش نمی‌شود

    cv2.imshow("lines and center", frame)
    if cv2.waitKey(30) & 0XFF == ord("q"):
        response = "f"
        ser0.write(response.encode())
        break
cap.release()
cv2.destroyAllWindows()
