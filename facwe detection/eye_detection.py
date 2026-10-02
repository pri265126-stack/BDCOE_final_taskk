import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

looking_away_events = 0
looking_away_start = None
event_counted = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]

        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        nose = face_landmarks[1]

        eye_center = (left_eye.x + right_eye.x) / 2
        difference = eye_center - nose.x

        if difference < -0.015:
            gaze = "Looking Left"
        elif difference > 0.015:
            gaze = "Looking Right"
        else:
            gaze = "Looking Center"

        # Looking-away timer
        if gaze == "Looking Center":

            looking_away_start = None
            event_counted = False

        else:

            if looking_away_start is None:
                looking_away_start = time.time()

            away_time = time.time() - looking_away_start

            if away_time >= 1.5 and not event_counted:
                looking_away_events += 1
                event_counted = True

        cv2.putText(
            frame,
            gaze,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Looking Away Events: {looking_away_events}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "No Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    cv2.imshow("SmartRecruit Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()