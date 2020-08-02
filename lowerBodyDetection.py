import cv2

face_classifier = cv2.CascadeClassifier('haarFiles/haarcascade_lowerbody.xml')

cap = cv2.VideoCapture(0)
img = cv2.imread("t5.jpg")
count = 0
while cap:
    # ret, img = cap.read()
    img = cv2.resize(img, (600, 700))
    lbody = face_classifier.detectMultiScale(img)
    print(lbody)
    for x, y, w, h in lbody:
        end_cord_x = x + w
        end_cord_y = y + h
        color = (0, 0, 255)
        stroke = 2
        cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)

    cv2.imshow('img', img)
    count += 1
    if cv2.waitKey(1) == 13 or 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
