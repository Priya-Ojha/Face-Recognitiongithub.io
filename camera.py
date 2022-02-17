import cv2
cap=cv2.VideoCapture(0)
while True:
ret, frame = cap. read()
if ret == false:
 continue
 cv2. imshow("video frame", frame)
 key_pressed = cv2.waitkey(1) & 0xFF
 if key_pressed == ord('q'):
 break
 cap.release()
 cv2.destroyAllWindows()