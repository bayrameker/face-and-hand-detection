# Gerekli modülleri içeri aktaralım
import cv2
import mediapipe as mp
import time

# Burada web kamerasından görüntü yakalıyoruz
cap = cv2.VideoCapture(0)

# Yüz ve Göz kademeleri
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


# Mp hand yani eklem ve parmak birleşim alanlarını tanımlıyoruz.
mpHands = mp.solutions.hands

# Mp Hand parametrelerini tanımlıyoruz
hands = mpHands.Hands()

# Mp method tanımlıyoruz çizmek için
mpDraw = mp.solutions.drawing_utils

# Kare hızı değişkenlerimiz
pTime = 0
cTime = 0

while True:
    # Video kameramızı çerçeve olarak yani frame olarak okuyoruz.
    success, img = cap.read()

    # okudumğumuz anlık çerçeveyi RGB olarak çeviriyoruz
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # okudumğumuz anlık çerçeveyi gray olarak çeviriyoruz
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sonuçlarımızı aldık
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    # Yüz ve göz kordinatlarımızı belirliyoruz
    face_coordinates = face_cascade.detectMultiScale(gray_frame)
    eye_coordinates = eye_cascade.detectMultiScale(gray_frame)

    # Kare olarak yüz ve göz çerçevelerini çizdiriyoruz
    for (fx, fy, fw, fh) in face_coordinates:
        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    for (ex, ey, ew, eh) in eye_coordinates:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

    # Birden fazla el olup olmadığını kontrol ediyoruz
    # Belirlediğimiz noktaların bağlantılarını çizdiriyoruz
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # El izleme verileri için Landmark ID'yi yazdırın
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                # LandMark Merkezini Matematiksel Olarak Belirleyin
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx,cy)
                # İlk dönüm noktasının (avuç içi) görüntülenip görüntülenmediğini belirleyin, ardından ayırt edin
                if id == 0:
                    cv2.circle(img, (cx,cy), 20, (255,0,255), cv2.FILLED)
                # Başparmağın (id=4) görüntülenip görüntülenmediğini belirleyin
                if id == 4 :
                    cv2.circle(img, (cx, cy), 10, (255, 127, 0), cv2.FILLED)
                # Parmak uçlarının görüntülenip görüntülenmediğini belirleyin
                if id == 8 or id == 12 or id == 16 or id == 20 :
                    cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Şu an
    cTime = time.time()

    # Saniyedeki Kare Sayısını (FPS) Hesaplıyoruz
    fps = 1/(cTime-pTime)

    # zamanı güncelliyoruz
    pTime = cTime

    # Fps Göster
    cv2.putText(img, "FPS ", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    cv2.putText(img,str(int(fps)), (10,80), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

    # Frame Göster
    cv2.imshow("El ve Yüz Kontrolü", img)

    # Her 1 mili saniyede frame güncelle
    key=cv2.waitKey(1)

    # Programdan çıkmak için q / Q tuşuna basın
    if key==81 or key==113:
        break

print("Kodumuz Tamamlandı")