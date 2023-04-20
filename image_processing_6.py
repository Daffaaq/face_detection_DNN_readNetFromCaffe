import cv2
import numpy as np

# Load model DNN untuk deteksi wajah
model = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Menyalakan kamera laptop
cap = cv2.VideoCapture(0)

while True:
    # Mendapatkan frame dari kamera
    ret, frame = cap.read()

    # Mengubah ukuran frame menjadi 300x300 piksel
    resized_frame = cv2.resize(frame, (300, 300))

    # Mengubah tipe data frame menjadi blob
    blob = cv2.dnn.blobFromImage(
        resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Memasukkan blob ke dalam model untuk mendeteksi wajah
    model.setInput(blob)
    detections = model.forward()

    # Menggambar kotak pada wajah yang terdeteksi
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype("int")
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)

    # Menampilkan frame
    cv2.imshow('Camera', frame)

    # Menghentikan looping dengan menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan kamera dan menutup jendela
cap.release()
cv2.destroyAllWindows()
