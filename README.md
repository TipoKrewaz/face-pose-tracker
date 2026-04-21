# face-pose-tracker

Realtime face mesh, body skeleton and basic eye behavior from a webcam.
MediaPipe does the inference, OpenCV handles capture and rendering.

## install

```bash
pip install -r requirements.txt
```

Python 3.10 – 3.12. On Windows, camera access must be enabled in
Privacy settings → Camera → *Let desktop apps access your camera*.

## run

```bash
python detect.py
```

## keys

The window has to be focused.

- `1` / `2` / `3` — pose model: lite / full / heavy
- `space` — toggle dense face mesh
- `0` — reset blink counter
- `Esc` / `q` — quit

## notes

- `EAR_CLOSED = 0.21` — if blinks flicker under bad lighting, drop it to `0.18`.
- If the camera index `0` picks up OBS virtual cam / IR camera, try `cv2.VideoCapture(1, cv2.CAP_DSHOW)`.
- Face mesh tracks one face. For multi-person skeletons you'd want
  YOLOv8-pose instead of MediaPipe.
- Gaze estimation is coarse (left / right / center) — for on-screen
  coordinates it would need per-user calibration.
