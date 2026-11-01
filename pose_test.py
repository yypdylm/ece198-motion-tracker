import cv2
import time
import mediapipe as mp

# 加载 Google MediaPipe Full 模型
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="pose_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1
)

# 人体骨骼连接
connections = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Camera unavailable")

start = time.monotonic()

with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.monotonic() - start) * 1000)
        result = detector.detect_for_video(image, timestamp)

        for pose in result.pose_landmarks:
            for a, b in connections:
                p1 = (int(pose[a].x * w), int(pose[a].y * h))
                p2 = (int(pose[b].x * w), int(pose[b].y * h))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            for point in pose:
                x = int(point.x * w)
                y = int(point.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        cv2.imshow("ECE198 AI Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
