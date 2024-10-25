import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
frame_count = 0
object_data = []
target_color = None
initial_area = None

def get_dominant_color(hsv_frame, mask):
    masked_frame = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)
    hist = cv2.calcHist([masked_frame], [0], mask, [180], [0, 180])
    dominant_color = np.argmax(hist)
    return dominant_color
    lower_blue = np.array([11, 47, 192])
    upper_blue = np.array([52, 165, 255])
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    denoised_img1 = cv2.bilateralFilter(h, 9, 20, 20)
    denoised_img2 = cv2.bilateralFilter(s, 9, 20, 20)
    denoised_img3 = cv2.bilateralFilter(v, 9, 20, 20)

    blurred1 = cv2.GaussianBlur(denoised_img1, (11, 11), 0)
    blurred2 = cv2.GaussianBlur(denoised_img2, (11, 11), 0)
    blurred3 = cv2.GaussianBlur(denoised_img3, (11, 11), 0)

    adaptive_gaussian_thresh1 = cv2.adaptiveThreshold(blurred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    adaptive_gaussian_thresh2 = cv2.adaptiveThreshold(blurred2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    adaptive_gaussian_thresh3 = cv2.adaptiveThreshold(blurred3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    combined_edges1 = cv2.bitwise_and(adaptive_gaussian_thresh2, adaptive_gaussian_thresh3)
    combined_edges2 = cv2.bitwise_and(combined_edges1, adaptive_gaussian_thresh1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_edges2)

    if frame_count < 30:
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if 30000 > area > 1000:
                component = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in contours:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
                    dominant_color = get_dominant_color(hsv, mask)
                    object_data.append((area, dominant_color))

    if frame_count == 30:
        color_counts = {}
        for area, color in object_data:
            if color in color_counts:
                color_counts[color] += 1
            else:
                color_counts[color] = 1
        target_color = max(color_counts, key=color_counts.get)

    if frame_count > 30 and target_color is not None:
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if 20000 > area > 600:
                component = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in contours:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
                    dominant_color = get_dominant_color(hsv, mask)

                    if dominant_color == target_color:
                        if initial_area is None:
                            initial_area = area
                        elif area < initial_area:
                            print("w")
                        elif area > initial_area:
                            print("s")

                        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

    frame_count += 1
    cv2.imshow('Frame', frame)
    cv2.imshow('mask', mask)


    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
