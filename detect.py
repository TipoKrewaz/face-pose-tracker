import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


mp_face   = mp.solutions.face_mesh
mp_pose   = mp.solutions.pose
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# iris появляется когда refine_landmarks=True
LEFT_IRIS_C  = 468
RIGHT_IRIS_C = 473
LEFT_CORNERS  = (33,  133)
RIGHT_CORNERS = (362, 263)

EAR_CLOSED = 0.21

PROC_W = 640
PROC_H = 480

POSE_NAMES = ["lite", "full", "heavy"]


def ear(pts, idx):
    p = [pts[i] for i in idx]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    hz = np.linalg.norm(p[0] - p[3])
    if hz < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * hz)


def gaze_x(pts, corners, iris_idx):
    outer, inner = pts[corners[0]], pts[corners[1]]
    iris = pts[iris_idx]
    axis = inner - outer
    axis_len2 = float(axis @ axis) or 1.0
    t = float((iris - (outer + inner) * 0.5) @ axis) / axis_len2
    return float(np.clip(2.0 * t, -1.5, 1.5))


def make_pose(complexity):
    return mp_pose.Pose(
        model_complexity=complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def run(cam_idx=0):
    cam = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cam.isOpened():
        sys.exit("камера не открылась")

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  PROC_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, PROC_H)
    cam.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cam.set(cv2.CAP_PROP_FPS,          30)

    face = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("loading pose models: lite, full, heavy ...", flush=True)
    poses = [make_pose(c) for c in (0, 1, 2)]
    print("ready", flush=True)

    pose_idx  = 1
    show_mesh = False

    ear_win = deque(maxlen=5)
    blinks  = 0
    was_closed = False

    t_mark  = time.time()
    n_since = 0
    fps     = 0.0

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            p_res = poses[pose_idx].process(rgb)
            f_res = face.process(rgb)
            rgb.flags.writeable = True

            if p_res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    p_res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            eyes_line = ""
            gaze_line = ""
            if f_res.multi_face_landmarks:
                fm  = f_res.multi_face_landmarks[0]
                pts = np.array([[p.x * w, p.y * h] for p in fm.landmark], dtype=np.float32)

                if show_mesh:
                    mp_draw.draw_landmarks(
                        frame, fm, mp_face.FACEMESH_TESSELATION, None,
                        mp_styles.get_default_face_mesh_tesselation_style(),
                    )
                mp_draw.draw_landmarks(
                    frame, fm, mp_face.FACEMESH_CONTOURS, None,
                    mp_styles.get_default_face_mesh_contours_style(),
                )
                mp_draw.draw_landmarks(
                    frame, fm, mp_face.FACEMESH_IRISES, None,
                    mp_styles.get_default_face_mesh_iris_connections_style(),
                )

                le = ear(pts, LEFT_EYE)
                re = ear(pts, RIGHT_EYE)
                ear_win.append((le + re) * 0.5)
                ear_smooth = float(np.mean(ear_win))
                closed = ear_smooth < EAR_CLOSED
                if was_closed and not closed:
                    blinks += 1
                was_closed = closed

                gx = 0.5 * (gaze_x(pts, LEFT_CORNERS,  LEFT_IRIS_C) +
                            gaze_x(pts, RIGHT_CORNERS, RIGHT_IRIS_C))
                if   gx < -0.15: look = "влево"
                elif gx >  0.15: look = "вправо"
                else:            look = "прямо"

                eyes_line = f"EAR {ear_smooth:.2f}  {'закрыты' if closed else 'открыты'}  blinks={blinks}"
                gaze_line = f"взгляд: {look} ({gx:+.2f})"

                for iris_c in (LEFT_IRIS_C, RIGHT_IRIS_C):
                    ix, iy = pts[iris_c]
                    cv2.circle(frame, (int(ix), int(iy)), 3, (0, 255, 255), -1, cv2.LINE_AA)

            # fps
            n_since += 1
            dt = time.time() - t_mark
            if dt >= 0.5:
                fps = n_since / dt
                t_mark = time.time()
                n_since = 0

            hint = (f"{fps:.1f} fps  pose={POSE_NAMES[pose_idx]}  mesh={'on' if show_mesh else 'off'}   "
                    f"[1/2/3 pose | space mesh | 0 reset | Esc quit]")
            lines = [s for s in (hint, eyes_line, gaze_line) if s]
            y = h - 10
            for s in reversed(lines):
                (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (6, y - th - 8), (12 + tw, y + 4), (0, 0, 0), -1)
                cv2.putText(frame, s, (10, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
                y -= th + 12

            cv2.imshow("face+pose", frame)

            # буквы дублирую русскими кодами на случай ру-раскладки
            k = cv2.waitKey(1) & 0xFFFF
            if k == 27 or k == ord('q') or k == 0x0439:
                break
            if k == ord('0') or k == ord('r') or k == 0x043A:
                blinks = 0
            if k == ord('1'):
                pose_idx = 0
            if k == ord('2'):
                pose_idx = 1
            if k == ord('3'):
                pose_idx = 2
            if k == ord(' ') or k == ord('m') or k == 0x044C:
                show_mesh = not show_mesh
    finally:
        cam.release()
        cv2.destroyAllWindows()
        face.close()
        for p in poses:
            p.close()


if __name__ == "__main__":
    run()
