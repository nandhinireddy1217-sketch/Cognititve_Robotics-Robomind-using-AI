import cv2
def capture_camera():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        return None

    success, frame = camera.read()
    camera.release()

    if not success:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame