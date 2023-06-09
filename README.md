# Emotion Detector Test Webapplication

Emotion Detector เป็นเว็บแอปพลิเคชันที่ใช้งานด้วย Streamlit และ WebRTC ในการตรวจจับอารมณ์ของใบหน้าผ่านกล้องเว็บแคม โดยใช้โมเดลการเรียนรู้เชิงลึกที่ได้รับการฝึกสอน (model.h5)
เว็บแอปพลิเคชันนี้จัดทำขึ้นเพื่อเป็นการทดลองในการทำโปรเจคมหาวิทยาลัยเท่านั้น

## ความต้องการ

- Python 3.7+
- Streamlit
- streamlit-webrtc
- OpenCV
- Tensorflow
- Keras

## วิธีติดตั้ง

1. ติดตั้ง Python 3.7+ หากยังไม่มี
2. ใช้คำสั่งต่อไปนี้เพื่อติดตั้งไลบรารีที่จำเป็น:

```bash
pip install streamlit streamlit-webrtc opencv-python tensorflow keras
```

## วิธีใช้งาน

1. ดาวน์โหลดโค้ดตัวอย่างนี้
2. ดาวน์โหลดไฟล์ haarcascade_frontalface_default.xml และ model.h5 จากแหล่งข้อมูลที่เชื่อถือได้ และนำมาเก็บในโฟลเดอร์เดียวกับโค้ด
3. ใน terminal, ย้ายไปยังโฟลเดอร์ที่มีไฟล์ app.py และรันคำสั่ง:

```bash
streamlit run app.py
```

4. เปิดเว็บเบราว์เซอร์ของคุณและไปที่ URL ที่ Streamlit แสดงให้ (โดยปกติจะเป็น http://localhost:8501)
5. หากเว็บแอปถามสิทธิ์การเข้าถึงกล้อง กรุณาอนุญาตให้เข้าถึง
6. ดูผลลัพธ์ของการตรวจจับอารมณ์จากกล้องเว็บแคมของคุณ


สามารถลองใช้งานได้ง่ายผ่าน https://noctiszia-test-emotion-recognition-web-2-app-zeejly.streamlit.app/