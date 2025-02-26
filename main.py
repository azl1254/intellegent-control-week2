import cv2
import joblib
import numpy as np
import pandas as pd

# Load dataset warna
df = pd.read_csv('colors_cleaned.csv')
X = df[['B', 'G', 'R']].values
y = df['color'].values

# Muat model SVM dan scaler
svm = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ambil ukuran frame
    height, width, _ = frame.shape
    
    # Koordinat tengah untuk dua bounding box
    cx1, cy1 = width // 3, height // 2  # Kiri tengah
    cx2, cy2 = 2 * width // 3, height // 2  # Kanan tengah

    # Ukuran bounding box
    box_size = 50
    x1a, y1a, x1b, y1b = cx1 - box_size // 2, cy1 - box_size // 2, cx1 + box_size // 2, cy1 + box_size // 2
    x2a, y2a, x2b, y2b = cx2 - box_size // 2, cy2 - box_size // 2, cx2 + box_size // 2, cy2 + box_size // 2
    
    # Ambil warna rata-rata dalam kedua bounding box
    roi1 = frame[y1a:y1b, x1a:x1b]
    roi2 = frame[y2a:y2b, x2a:x2b]
    
    mean_color1 = roi1.mean(axis=(0, 1))  # Rata-rata warna pertama (B, G, R)
    mean_color2 = roi2.mean(axis=(0, 1))  # Rata-rata warna kedua (B, G, R)
    
    pixel_center1 = np.array(mean_color1, dtype=np.uint8).reshape(1, -1)
    pixel_center2 = np.array(mean_color2, dtype=np.uint8).reshape(1, -1)

    # Normalisasi sebelum prediksi
    pixel_center_scaled1 = scaler.transform(pixel_center1)
    pixel_center_scaled2 = scaler.transform(pixel_center2)

    # Prediksi warna dengan SVM
    color_pred1 = svm.predict(pixel_center_scaled1)[0]
    color_pred2 = svm.predict(pixel_center_scaled2)[0]
    
    # Hitung akurasi prediksi dengan jarak ke warna yang paling mendekati
    distances1 = np.linalg.norm(X - pixel_center1, axis=1)
    distances2 = np.linalg.norm(X - pixel_center2, axis=1)
    
    min_distance1 = np.min(distances1)
    min_distance2 = np.min(distances2)
    
    accuracy1 = max(0, 100 - (min_distance1 / np.max(distances1)) * 100)  # Rentang 0-100%
    accuracy2 = max(0, 100 - (min_distance2 / np.max(distances2)) * 100)  # Rentang 0-100%

    # Gambar bounding box
    cv2.rectangle(frame, (x1a, y1a), (x1b, y1b), (0, 255, 0), 2)
    cv2.rectangle(frame, (x2a, y2a), (x2b, y2b), (255, 0, 0), 2)
    
    # Tampilkan warna dan akurasi di layar
    text1 = f'Color: {color_pred1} ({accuracy1:.2f}%)'
    text2 = f'Color: {color_pred2} ({accuracy2:.2f}%)'
    
    cv2.putText(frame, text1, (x1a, y1a - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, text2, (x2a, y2a - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Tampilkan frame
    cv2.imshow('frame', frame)
    
    # Print informasi ke terminal
    print(f'Detected Color 1: {color_pred1}, Accuracy: {accuracy1:.2f}%')
    print(f'Detected Color 2: {color_pred2}, Accuracy: {accuracy2:.2f}%')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
