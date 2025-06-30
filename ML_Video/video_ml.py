import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        eye_img = roi_color[ey:ey+eh, ex:ex+ew]
        cv2.imshow('Eye', eye_img)


from sklearn.cluster import KMeans
import numpy as np

def dominant_color(img, k=3):
    img = cv2.resize(img, (50, 50))  # speed up
    data = img.reshape((-1, 3)).astype(np.float32)
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(data)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    dominant = colors[np.bincount(labels).argmax()]
    return tuple(dominant)


bgr = dominant_color(eye_img)
print("Eye color (BGR):", bgr)
