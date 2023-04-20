import cv2
import numpy as np

# Load model DNN untuk deteksi wajah
model = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Load Cascade Classifier untuk deteksi mata
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load Cascade Classifier untuk deteksi senyum
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Menyalakan kamera laptop
cap = cv2.VideoCapture(0)

while True:
    # Mendapatkan frame dari kamera
    ret, frame = cap.read()

    # Mendeteksi wajah pada frame
    resized_frame = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(
        resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    # Menggambar kotak pada wajah, mata, dan senyum yang terdeteksi
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Mendeteksi mata pada wajah yang terdeteksi
            roi_gray = frame[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey),
                              (ex + ew, ey + eh), (0, 255, 0), 2)

            # Mendeteksi senyum pada wajah yang terdeteksi
            smile = smile_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(roi_color, (sx, sy),
                              (sx + sw, sy + sh), (0, 0, 255), 2)

    # Menampilkan frame
    cv2.imshow('Camera', frame)

    # Menghentikan looping dengan menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan kamera dan menutup jendela
cap.release()
cv2.destroyAllWindows()
