import cv2
import time
import math
import mediapipe as mp

def knee_angle(hip, knee, ankle, w, h):
    # 转换成像素坐标
    hx, hy = hip.x * w, hip.y * h
    kx, ky = knee.x * w, knee.y * h
    ax, ay = ankle.x * w, ankle.y * h

    # 以膝盖为起点的两个向量
    ux, uy = hx - kx, hy - ky
    vx, vy = ax - kx, ay - ky

    dot = ux * vx + uy * vy
    cross = ux * vy - uy * vx

    angle = math.degrees(math.atan2(abs(cross), dot))
    return angle

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="pose_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1
)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("Camera unavailable")

start = time.monotonic()
last_print = 0
peak_right = None

smooth_angle = None
reps = 0
stage = "WAIT_BEND"
confirm = 0
last_valid = time.monotonic()

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

        output = []

        if result.pose_landmarks:
            pose = result.pose_landmarks[0]

            for name, ids in [
                ("RIGHT", (24, 26, 28))
            ]:
                hip, knee, ankle = [pose[i] for i in ids]

                # 关节不可见时不输出角度
                if min(hip.visibility, knee.visibility,
                       ankle.visibility) < 0.6:
                    continue

                angle = knee_angle(hip, knee, ankle, w, h)

                last_valid = time.monotonic()

                if smooth_angle is None:
                    smooth_angle = angle
                else:
                    smooth_angle += 0.30 * (angle - smooth_angle)

                peak_right = angle if peak_right is None else max(peak_right, angle)

                target = (
                    smooth_angle <= 105 if stage != "EXTEND"
                    else smooth_angle >= 160
                )

                confirm = confirm + 1 if target else 0

                if confirm >= 4:
                    confirm = 0
                    if stage == "WAIT_BEND":
                        stage = "EXTEND"
                    elif stage == "EXTEND":
                        stage = "RETURN"
                    elif stage == "RETURN":
                        reps += 1
                        stage = "EXTEND"
                        print(f"*** REP COMPLETED! TOTAL={reps} ***")

                output.append(
                    f"RAW={angle:.1f} "
                    f"SMOOTH={smooth_angle:.1f} "
                    f"STAGE={stage} REPS={reps}"
                )

                # 在膝盖附近显示角度
                x = int(knee.x * w)
                y = int(knee.y * h)

                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)

                cv2.putText(
                    frame,
                    f"{name}: {angle:.1f} deg",
                    (max(10, x - 40), max(30, y - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                for a, b in [(hip, knee), (knee, ankle)]:
                    p1 = (int(a.x * w), int(a.y * h))
                    p2 = (int(b.x * w), int(b.y * h))
                    cv2.line(frame, p1, p2, (0, 255, 0), 3)

        if not output:
            confirm = 0
            if time.monotonic() - last_valid > 0.5:
                smooth_angle = None
                stage = "WAIT_BEND"

        cv2.putText(
            frame, f"REPS: {reps}   {stage}",
            (20, 85), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2
        )

        # 每 0.3 秒向 Terminal 输出一次
        now = time.monotonic()
        if now - last_print >= 0.3:
            print(" ".join(output) if output else "NO_KNEE")
            last_print = now

        if peak_right is not None:
            cv2.putText(
                frame, f"PEAK: {peak_right:.1f} deg",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2
            )

        cv2.imshow("ECE198 Knee Angle", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            peak_right = None
            print("PEAK RESET")

cap.release()
cv2.destroyAllWindows()
