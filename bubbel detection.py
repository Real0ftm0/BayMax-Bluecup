import cv2
import numpy as np
import serial

ser0 = serial.Serial('/dev/ttyUSB0', 115200)

def compute_shapefactor(circles, binary):
    shapefactor_circles = []
    for (x, y, r) in circles:
        mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        circle_area = np.sum(mask == 255)
        intersection_area = np.sum(np.logical_and(mask == 255, binary == 255))
        if circle_area > 0:
            shapefactor = intersection_area / circle_area
            shapefactor_circles.append((x, y, r, shapefactor))
    return shapefactor_circles

def filter_close_circles(circles, min_count=3, max_distance=30):
    filtered_circles = []
    visited = [False] * len(circles)

    def distance(c1, c2):
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def group_circles(index):
        group = []
        queue = [index]
        while queue:
            i = queue.pop(0)
            if not visited[i]:
                visited[i] = True
                group.append(circles[i])
                for j in range(len(circles)):
                    if not visited[j] and distance(circles[i], circles[j]) < max_distance:
                        queue.append(j)
        return group

    for i in range(len(circles)):
        if not visited[i]:
            group = group_circles(i)
            if len(group) >= min_count:
                filtered_circles.append(group)

    return filtered_circles

cap = cv2.VideoCapture('C:\\Users\Jamal\Desktop\\New folder\yazd blue cup\data\pipeline-real.mp4')

ret, prev_frame = cap.read()
if not ret:
    print("Unable to read video.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, curr_frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, curr_gray)

    blurred = cv2.GaussianBlur(diff, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary', binary)

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    height, width = curr_frame.shape[:2]
    crop_height = int(height * 0.7)
    curr_frame = curr_frame[:crop_height, :]
    cleaned = binary[:crop_height, :]

    circles = cv2.HoughCircles(cleaned, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=50)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        shapefactor_circles = compute_shapefactor(circles, binary)

        filtered_circles_groups = filter_close_circles(circles)

        for group in filtered_circles_groups:
            for (x, y, r) in group:
                cv2.circle(curr_frame, (x, y), r, (0, 0, 255), 4)

    cv2.imshow('Detected Bubbles', curr_frame)

    prev_gray = curr_gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        response = "f"
        ser0.write(response.encode())
        break

cap.release()
cv2.destroyAllWindows()
