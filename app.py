import cv2
from yolov7  import YOLOv7  # ต้องแก้ไขเป็นการ import จากโมดูลที่คุณติดตั้ง

# สร้างอินสแตนซ์ของโมเดล YOLOv7
yolo = YOLOv7()

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)  # 0 หมายถึงกล้องหลัก

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ทำนายวัตถุในเฟรม
    predictions = yolo.predict(frame)

    # แสดงผลลัพธ์ที่ได้จากการทำนายบนเฟรม
    # โดยวาดกรอบสี่เหลี่ยมรอบวัตถุและแสดงชื่อคลาส
    frame = yolo.draw_predictions(frame, predictions)

    cv2.imshow('YOLOv7 Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
