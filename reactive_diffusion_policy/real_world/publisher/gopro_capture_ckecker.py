import cv2

cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Failed to open camera")
else:
    ret, frame = cap.read()
    print("capture result:", ret)
    if ret:
        print("Frame shape:", frame.shape)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
        cv2.imwrite("test_frame.jpg", frame)
    else:
        print("Failed to read frame")
cap.release()
