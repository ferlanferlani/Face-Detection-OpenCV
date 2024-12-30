import cv2
import numpy as np

cascPath = "src/eye.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

def apply_funny_effects(frame, faces):
    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        
        # Enlarge the eyes
        eyes = eyeCascade.detectMultiScale(roi_color)
        for (ex, ey, ew, eh) in eyes:
            if ew > 0 and eh > 0:
                eye = roi_color[ey:ey+eh, ex:ex+ew]
                eye = cv2.resize(eye, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                ew, eh = eye.shape[:2]
                # Ensure the resized eye fits back into the original region
                eye = eye[:min(eh, roi_color.shape[0] - ey), :min(ew, roi_color.shape[1] - ex)]
                roi_color[ey:ey+eye.shape[0], ex:ex+eye.shape[1]] = eye

        # Enlarge the mouth area
        mouth_y = y + int(h * 0.7)
        mouth_h = int(h * 0.3)
        if mouth_y + mouth_h <= roi_color.shape[0]:
            mouth = roi_color[mouth_y:mouth_y+mouth_h, :]
            if mouth.size > 0:
                mouth = cv2.resize(mouth, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                mh, mw = mouth.shape[:2]
                # Ensure the resized mouth fits back into the original region
                mouth = mouth[:min(mh, roi_color.shape[0] - mouth_y), :min(mw, roi_color.shape[1])]
                roi_color[mouth_y:mouth_y+mouth.shape[0], :mouth.shape[1]] = mouth

while True:
    # Capture frame-by-frame
    rect, frame = video_capture.read()    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Apply funny effects to the faces
    apply_funny_effects(frame, faces)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()