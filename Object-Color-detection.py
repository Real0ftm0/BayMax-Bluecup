import cv2
import numpy as np
import serial
import sys
import select

ser0 = serial.Serial('/dev/ttyUSB0', 115200)

color_ranges = {
    'blue': [(90, 50, 80), (130, 255, 255)],
    'yellow': [(20, 100, 100), (30, 255, 255)],
    'red_1': [(0, 100, 100), (10, 255, 255)],
    'red_2': [(170, 100, 100), (180, 255, 255)],
    'green': [(40, 50, 80), (110, 255, 255)],
    'orange': [(10, 100, 100), (20, 255, 255)],
    'white': [(0, 0, 200), (180, 20, 255)],
    'black': [(0, 0, 0), (180, 255, 38)],
    'brown': [(10, 100, 60), (20, 255, 200)],
    'purple': [(130, 50, 40), (160, 255, 255)],
    'pink': [(160, 50, 70), (210, 255, 255)]
}

def calculate_mean_color(contour_region_hsv):
    mask = cv2.cvtColor(contour_region_hsv, cv2.COLOR_BGR2GRAY)
    mean_color = cv2.mean(contour_region_hsv, mask=mask)[:3]
    return mean_color

# cap = cv2.VideoCapture('E:\BLUECUP\\vid\\test1\\2024-08-31_11.18.34.mp4')
cap = cv2.VideoCapture(0)

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    b, g, r = cv2.split(frame)

    denoised_img1 = cv2.bilateralFilter(r, 9, 20, 20)
    denoised_img2 = cv2.bilateralFilter(b, 9, 20, 20)
    denoised_img3 = cv2.bilateralFilter(g, 9, 20, 20)

    blurred1 = cv2.GaussianBlur(denoised_img1, (11, 11), 0)
    blurred2 = cv2.GaussianBlur(denoised_img2, (11, 11), 0)
    blurred3 = cv2.GaussianBlur(denoised_img3, (11, 11), 0)

    adaptive_gaussian_thresh1 = cv2.adaptiveThreshold(blurred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    adaptive_gaussian_thresh2 = cv2.adaptiveThreshold(blurred2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    adaptive_gaussian_thresh3 = cv2.adaptiveThreshold(blurred3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    combined_edges1 = cv2.bitwise_and(adaptive_gaussian_thresh2, adaptive_gaussian_thresh3)
    combined_edges2 = cv2.bitwise_and(combined_edges1, adaptive_gaussian_thresh1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_edges2)


    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if 20000 > area > 600:
            component = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                hull = cv2.convexHull(c)
                epsilon = 0.02 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)

                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

                contour_region_hsv = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)

                mean_color = calculate_mean_color(contour_region_hsv)
                best_color = "unknown"
                for color_name, (lower_hsv, upper_hsv) in color_ranges.items():
                    lower_hsv = np.array(lower_hsv)
                    upper_hsv = np.array(upper_hsv)
                    if np.all(lower_hsv <= mean_color) and np.all(mean_color <= upper_hsv):
                        best_color = color_name
                        break

                shape_name = "Unknown"
                if len(approx) == 3:
                    shape_name = "Triangle"
                elif len(approx) == 4:
                    aspect_ratio = float(w) / h
                    if 0.95 <= aspect_ratio <= 1.05:
                        shape_name = "Square"
                    else:
                        shape_name = "Rectangle"
                elif len(approx) == 5:
                    shape_name = "Pentagon"
                elif len(approx) == 6:
                    shape_name = "Hexagon"
                else:
                    shape_name = "Circle"

                if (shape_name != "Unknown") and (best_color != "unknown"):
                    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
                    cv2.putText(frame, f"{best_color}, {shape_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    # cv2.putText(frame, f"Area: {int(area)}" ,(x - 50, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Frame', frame)
    # if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
    #     response = sys.stdin.readline().strip()
        # ser0.write(response.encode())
    # cv2.imshow('tresh', contour_region_hsv)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        response = "f"
        ser0.write(response.encode())
        break

cap.release()
cv2.destroyAllWindows()
