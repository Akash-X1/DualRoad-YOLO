# LIVE DETECTION - TWO SEPARATE WINDOWS (POTHOLE + CRACK)

import cv2
import numpy as np
from ultralytics import YOLO

CAMERA_INDEX = 1  # 0 = laptop webcam, 1 = phone via Phone Link

POTHOLE_MODEL_PATH = r"C:\Users\akaka\Desktop\msc thesis\final 2\final 3\final 4\runs\detect\Pothole_Final4_Runs\phase2_finetune_all\weights\best.pt"

POTHOLE_IMGSZ = 640

CRACK_MODEL_PATH = r"C:\Users\akaka\Desktop\msc thesis\final 2\final 3\final 4\final5\final6\final7\crack\crack3\runs\detect\Crack3_Runs\crack3\weights\best.pt"

CRACK_IMGSZ = 800


def compute_risk(is_pothole, box_area, frame_area, confidence):

    area_ratio = box_area / frame_area

    base_weight = 4.0 if is_pothole else 2.0

    proximity_factor = min(area_ratio * 25.0, 6.0)

    raw_score = (base_weight + proximity_factor) * confidence

    return int(np.clip(np.round(raw_score), 1, 10))


def risk_color(risk):

    if risk <= 3:

        return (0, 255, 0)

    elif risk <= 6:

        return (0, 255, 255)

    else:

        return (0, 0, 255)


def draw_detections(model, frame, imgsz, label, is_pothole):

    """Runs one model on a COPY of the frame and draws its own boxes.

    Returns the annotated copy - the original frame is never touched,

    so the other model always sees a clean, unmarked frame too."""

    working_frame = frame.copy()

    h, w, _ = working_frame.shape

    frame_area = float(h * w)

    results = model.predict(
        frame,
        imgsz=imgsz,
        conf=0.35,
        verbose=False
    )

    for result in results:

        if result.boxes is None:

            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])

            box_area = float((x2 - x1) * (y2 - y1))

            risk = compute_risk(
                is_pothole,
                box_area,
                frame_area,
                conf
            )

            color = risk_color(risk)

            cv2.rectangle(
                working_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                working_frame,
                f"{label} Risk:{risk}/10",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return working_frame


print("Loading Pothole model (Final 4)...")

pothole_model = YOLO(POTHOLE_MODEL_PATH)

print("Loading Crack model (Crack 3)...")

crack_model = YOLO(CRACK_MODEL_PATH)

print("Both models loaded.")

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():

    print(
        "Could not open camera. Check CAMERA_INDEX, "
        "or that Phone Link is connected."
    )

else:

    print(
        "Camera started. Click either window and "
        "press 'q' to stop both."
    )

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        # Same original frame goes into BOTH models - each draws on its
        # own separate copy, so neither model's boxes ever appear on
        # the other model's window

        pothole_frame = draw_detections(
            pothole_model,
            frame,
            POTHOLE_IMGSZ,
            "Pothole",
            is_pothole=True
        )

        crack_frame = draw_detections(
            crack_model,
            frame,
            CRACK_IMGSZ,
            "Crack",
            is_pothole=False
        )

        cv2.imshow(
            "Pothole Detection (Final 4) - press q to quit",
            pothole_frame
        )

        cv2.imshow(
            "Crack Detection (Crack 3) - press q to quit",
            crack_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()

    print("Camera stopped.")