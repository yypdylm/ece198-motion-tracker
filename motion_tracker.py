cat > motion_tracker.py <<'PY'
import argparse
import cv2
import time
import math
import mediapipe as mp
import serial
from pathlib import Path

# ==========================
# Configuration
# ==========================
BAUD_RATE = 115200
SEND_RATE = 20
VISIBILITY_THRESHOLD = 0.6
CAMERA_ID = 0

LEFT = (23, 25, 27)
RIGHT = (24, 26, 28)

MODEL_PATH = Path(__file__).with_name("pose_landmarker.task")


# ==========================
# Knee angle calculation
# ==========================
def knee_angle(pose, ids, width, height):
    points = [pose[i] for i in ids]

    if any(p.visibility < VISIBILITY_THRESHOLD for p in points):
        return -1.0

    if any(not (0 <= p.x <= 1 and 0 <= p.y <= 1) for p in points):
        return -1.0

    hip, knee, ankle = points

    hx, hy = hip.x * width, hip.y * height
    kx, ky = knee.x * width, knee.y * height
    ax, ay = ankle.x * width, ankle.y * height

    ux, uy = hx - kx, hy - ky
    vx, vy = ax - kx, ay - ky

    if min(math.hypot(ux, uy), math.hypot(vx, vy)) < 3:
        return -1.0

    dot = ux * vx + uy * vy
    cross = ux * vy - uy * vx

    return math.degrees(math.atan2(abs(cross), dot))


def format_angle(angle):
    return "-1" if angle < 0 else f"{angle:.1f}"


# ==========================
# Serial configuration
# ==========================
parser = argparse.ArgumentParser()
parser.add_argument("--port", default=None)
parser.add_argument("--camera", type=int, default=CAMERA_ID)
args = parser.parse_args()

ser = None

if args.port:
    ser = serial.Serial(
        args.port,
        BAUD_RATE,
        timeout=0,
        write_timeout=0.2
    )
    time.sleep(2)
    print(f"Serial connected: {args.port}")
else:
    print("PREVIEW MODE - Serial transmission disabled")


# ==========================
# MediaPipe initialization
# ==========================
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1
)

cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    if ser:
        ser.close()
    raise RuntimeError("Camera unavailable")

start = time.monotonic()
next_send = start
last_timestamp = -1

print("ECE198 Motion Tracker Started")


# ==========================
# Main loop
# ==========================
try:
    with mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    ) as detector:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Camera frame unavailable")
                break

            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = max(
                last_timestamp + 1,
                int((time.monotonic() - start) * 1000)
            )
            last_timestamp = timestamp

            result = detector.detect_for_video(image, timestamp)

            left_angle = -1.0
            right_angle = -1.0

            if result.pose_landmarks:
                pose = result.pose_landmarks[0]

                left_angle = knee_angle(
                    pose, LEFT, width, height
                )

                right_angle = knee_angle(
                    pose, RIGHT, width, height
                )

                # Draw valid leg landmarks
                for ids, angle in [
                    (LEFT, left_angle),
                    (RIGHT, right_angle)
                ]:
                    if angle < 0:
                        continue

                    for a, b in zip(ids, ids[1:]):
                        p1 = (
                            int(pose[a].x * width),
                            int(pose[a].y * height)
                        )
                        p2 = (
                            int(pose[b].x * width),
                            int(pose[b].y * height)
                        )
                        cv2.line(frame, p1, p2, (0, 255, 0), 3)

            # ==========================
            # USB Serial transmission
            # ==========================
            now = time.monotonic()

            if now >= next_send:
                message = (
                    f"K1,{format_angle(left_angle)},"
                    f"{format_angle(right_angle)}\n"
                )

                if ser:
                    ser.write(message.encode("ascii"))
                    print("TX:", message.strip())
                else:
                    print("PREVIEW:", message.strip())

                next_send = now + 1.0 / SEND_RATE

            # ==========================
            # Camera preview
            # ==========================
            cv2.putText(
                frame,
                f"LEFT: {format_angle(left_angle)} deg",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2
            )

            cv2.putText(
                frame,
                f"RIGHT: {format_angle(right_angle)} deg",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2
            )

            cv2.imshow("ECE198 Motion Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()

    if ser:
        ser.close()

    print("Motion Tracker Stopped")
PY