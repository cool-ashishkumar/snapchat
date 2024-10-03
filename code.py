import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from the webcam
video_capture = cv2.VideoCapture(0)

# Filter selection
filter_index = 0
filter_names = ['None', 'Grayscale', 'Blur', 'Edges', 'Sepia', 'Sketch', 'Invert Colors', 'Brightness Increase', 'Contrast Increase', 'HDR Effect', 'Cartoon Effect']

def none_filter(frame):
    return frame

def grayscale_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def blur_filter(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def edge_detection_filter(frame):
    return cv2.Canny(frame, 100, 200)

def sepia_filter(frame):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    return cv2.transform(frame, sepia_filter)

def sketch_filter(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv_gray_frame = cv2.bitwise_not(gray_frame)
    return cv2.divide(gray_frame, 255 - inv_gray_frame, scale=256.0)

def invert_colors_filter(frame):
    return cv2.bitwise_not(frame)

def increase_brightness_filter(frame, value=30):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def increase_contrast_filter(frame, alpha=1.5, beta=0):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def hdr_effect_filter(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    hdr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    hdr = cv2.filter2D(hdr, -1, kernel)
    return hdr

def cartoon_effect_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_filter(frame, filter_index):
    if filter_index == 1:
        return grayscale_filter(frame)
    elif filter_index == 2:
        return blur_filter(frame)
    elif filter_index == 3:
        return edge_detection_filter(frame)
    elif filter_index == 4:
        return sepia_filter(frame)
    elif filter_index == 10:
        return sketch_filter(frame)
    elif filter_index == 6:
        return invert_colors_filter(frame)
    elif filter_index == 7:
        return increase_brightness_filter(frame)
    elif filter_index == 8:
        return increase_contrast_filter(frame)
    elif filter_index == 9:
        return hdr_effect_filter(frame)
    elif filter_index == 5:
        return cartoon_effect_filter(frame)
    else:
        return none_filter(frame)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Detect faces in the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Apply the selected filter
    filtered_frame = apply_filter(frame, filter_index)

    # Display the resulting frame
    cv2.imshow('Video with Filter', filtered_frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Change filter on key press (0-10 keys)
    if key == ord('1'):
        filter_index = 1
    elif key == ord('2'):
        filter_index = 2
    elif key == ord('3'):
        filter_index = 3
    elif key == ord('4'):
        filter_index = 4
    elif key == ord('5'):
        filter_index = 5
    elif key == ord('6'):
        filter_index = 6
    elif key == ord('7'):
        filter_index = 7
    elif key == ord('8'):
        filter_index = 8
    elif key == ord('9'):
        filter_index = 9
    elif key == ord('0'):
        filter_index = 0
    elif key == ord('c'):
        filter_index = 10
    elif key == ord('m'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


# Press 0 for no filter.
# Press 1 for grayscale.
# Press 2 for blur.
# Press 3 for edge detection.
# Press 4 for sepia.
# Press 5 for sketch.
# Press 6 for invert colors.
# Press 7 for brightness increase.
# Press 8 for contrast increase.
# Press 9 for HDR effect.
# Press c for cartoon effect.