import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
RESULT_DIR = os.path.join(STATIC_DIR, "results")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# DUAL MODEL MODES

MODEL_MODES = {

    "original": {

        "label": "Original (Uncompressed)",

        "pothole_path": r"C:\Users\akaka\Desktop\msc thesis\final 2\final 3\final 4\runs\detect\Pothole_Final4_Runs\phase2_finetune_all\weights\best.pt",

        "crack_path": r"C:\Users\akaka\Desktop\msc thesis\final 2\final 3\final 4\final5\final6\final7\crack\crack3\runs\detect\Crack3_Runs\crack3\weights\best.pt",

        "device": None,

    },

    "compressed": {

        "label": "Compressed (FP16 ONNX)",

        "pothole_path": r"C:\Users\akaka\Desktop\msc thesis\final 2\final 3\final 4\runs\detect\Pothole_Final4_Runs\phase2_finetune_all\weights\best.onnx",

        "crack_path": r"C:\Users\akaka\Desktop\msc thesis\final 2\final 3\final 4\final5\final6\final7\crack\crack3\runs\detect\Crack3_Runs\crack3\weights\best.onnx",

        "device": "cpu",

    },

}

POTHOLE_IMGSZ = 640

CRACK_IMGSZ = 800

loaded_models = {}


def get_models(mode):

    """Load the selected pothole/crack model pair once, then reuse it."""

    if mode not in MODEL_MODES:

        raise ValueError(f"Unknown model mode: {mode}")

    if mode not in loaded_models:

        entry = MODEL_MODES[mode]

        for path in (entry["pothole_path"], entry["crack_path"]):

            if not os.path.exists(path):

                raise FileNotFoundError(f"Model file not found: {path}")

        print(f"Loading {entry['label']} models...")

        loaded_models[mode] = {

            "pothole": YOLO(entry["pothole_path"]),

            "crack": YOLO(entry["crack_path"]),

        }

        print(f"{entry['label']} models loaded.")

    return loaded_models[mode]


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

    return (0, 0, 255)


def run_one_model(model, image, imgsz, label, is_pothole, device=None):

    """Run one specialist model on its own copy of the original image."""

    working_image = image.copy()

    h, w, _ = working_image.shape

    frame_area = float(h * w)

    predict_args = {

        "source": image,

        "imgsz": imgsz,

        "conf": 0.35,

        "verbose": False,

    }

    if device is not None:

        predict_args["device"] = device

    results = model.predict(**predict_args)

    count = 0

    for result in results:

        if result.boxes is None:

            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])

            count += 1

            box_area = float((x2 - x1) * (y2 - y1))

            risk = compute_risk(is_pothole, box_area, frame_area, conf)

            color = risk_color(risk)

            cv2.rectangle(working_image, (x1, y1), (x2, y2), color, 2)

            cv2.putText(

                working_image,

                f"{label} Risk:{risk}/10",

                (x1, y1 - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                color,

                2,

            )

    return working_image, count


def get_selected_mode():

    mode = request.form.get("model_mode", "original")

    if mode not in MODEL_MODES:

        return None

    return mode


@app.route("/", methods=["GET"])
def index():

    return render_template("dual_index.html")


@app.route("/predict_image", methods=["POST"])
def predict_image():

    if "file" not in request.files:

        return jsonify({"error": "No file uploaded"}), 400

    mode = get_selected_mode()

    if mode is None:

        return jsonify({"error": "Unknown model mode"}), 400

    file = request.files["file"]

    if file.filename == "":

        return jsonify({"error": "No file selected"}), 400

    safe_name = secure_filename(file.filename)

    upload_path = os.path.join(UPLOAD_DIR, safe_name)

    file.save(upload_path)

    image = cv2.imread(upload_path)

    if image is None:

        return jsonify({"error": "Failed to read image."}), 400

    try:

        models = get_models(mode)

    except (FileNotFoundError, ValueError) as exc:

        return jsonify({"error": str(exc)}), 500

    device = MODEL_MODES[mode]["device"]

    pothole_image, pothole_count = run_one_model(

        models["pothole"], image, POTHOLE_IMGSZ, "Pothole", True, device

    )

    crack_image, crack_count = run_one_model(

        models["crack"], image, CRACK_IMGSZ, "Crack", False, device

    )

    base_name = os.path.splitext(safe_name)[0]

    pothole_result_name = f"pothole_{mode}_{base_name}.jpg"

    crack_result_name = f"crack_{mode}_{base_name}.jpg"

    pothole_result_path = os.path.join(RESULT_DIR, pothole_result_name)

    crack_result_path = os.path.join(RESULT_DIR, crack_result_name)

    pothole_saved = cv2.imwrite(pothole_result_path, pothole_image)

    crack_saved = cv2.imwrite(crack_result_path, crack_image)

    if not pothole_saved or not crack_saved:

        return jsonify({
            "error": "Detection completed, but the result image could not be saved."
        }), 500

    return jsonify({

        "model_mode": mode,

        "model_label": MODEL_MODES[mode]["label"],

        "pothole_count": pothole_count,

        "crack_count": crack_count,

        "pothole_image": url_for(
            "static",
            filename=f"results/{pothole_result_name}"
        ),

        "crack_image": url_for(
            "static",
            filename=f"results/{crack_result_name}"
        ),

    })


@app.route("/predict_video", methods=["POST"])
def predict_video():

    if "file" not in request.files:

        return jsonify({"error": "No file uploaded"}), 400

    mode = get_selected_mode()

    if mode is None:

        return jsonify({"error": "Unknown model mode"}), 400

    file = request.files["file"]

    if file.filename == "":

        return jsonify({"error": "No file selected"}), 400

    safe_name = secure_filename(file.filename)

    upload_path = os.path.join(UPLOAD_DIR, safe_name)

    file.save(upload_path)

    cap = cv2.VideoCapture(upload_path)

    if not cap.isOpened():

        return jsonify({"error": "Failed to open video"}), 400

    try:

        models = get_models(mode)

    except (FileNotFoundError, ValueError) as exc:

        cap.release()

        return jsonify({"error": str(exc)}), 500

    device = MODEL_MODES[mode]["device"]

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    base_name = os.path.splitext(safe_name)[0]

    pothole_video_name = f"pothole_{mode}_{base_name}.mp4"

    crack_video_name = f"crack_{mode}_{base_name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    pothole_out = cv2.VideoWriter(

        os.path.join(RESULT_DIR, pothole_video_name),
        fourcc,
        fps,
        (w, h)

    )

    crack_out = cv2.VideoWriter(

        os.path.join(RESULT_DIR, crack_video_name),
        fourcc,
        fps,
        (w, h)

    )

    pothole_total = 0

    crack_total = 0

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        pothole_frame, p_count = run_one_model(

            models["pothole"],
            frame,
            POTHOLE_IMGSZ,
            "Pothole",
            True,
            device

        )

        crack_frame, c_count = run_one_model(

            models["crack"],
            frame,
            CRACK_IMGSZ,
            "Crack",
            False,
            device

        )

        pothole_total += p_count

        crack_total += c_count

        pothole_out.write(pothole_frame)

        crack_out.write(crack_frame)

    cap.release()

    pothole_out.release()

    crack_out.release()

    return jsonify({

        "model_mode": mode,

        "model_label": MODEL_MODES[mode]["label"],

        "pothole_count": pothole_total,

        "crack_count": crack_total,

        "pothole_video": url_for(
            "static",
            filename=f"results/{pothole_video_name}"
        ),

        "crack_video": url_for(
            "static",
            filename=f"results/{crack_video_name}"
        ),

    })


if __name__ == "__main__":

    app.run(debug=True, port=5000)