from flask import Flask, jsonify, request, send_from_directory
from ultralytics import YOLO
from PIL import Image
import io
import os
import time
import threading

# ============================================================
# SAFEPATH AI
# MAXIMUM SPEED VISION ENGINE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

app = Flask(__name__)

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")

IMAGE_SIZE = 320
CONFIDENCE = 0.35
IOU = 0.45
MAX_OBJECTS = 25

# CPU threads
CPU_THREADS = max(1, (os.cpu_count() or 4) - 1)

# ============================================================
# MODEL
# ============================================================

model = None
model_lock = threading.Lock()

print()
print("=" * 62)
print("                    SAFEPATH AI")
print("              MAXIMUM SPEED ENGINE")
print("=" * 62)
print()

try:

    model = YOLO(MODEL_PATH)

    print("AI MODEL  : ONLINE")
    print("MODEL     : YOLO11n")
    print("IMAGE     :", IMAGE_SIZE)
    print("CONF      :", CONFIDENCE)
    print("CPU THREADS:", CPU_THREADS)

except Exception as error:

    print("AI MODEL : OFFLINE")
    print("ERROR    :", error)

print()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# STATUS
# ============================================================

@app.route("/api/status")
def status():

    return jsonify({

        "success": True,

        "system": "SafePath AI",

        "ai": "online" if model else "offline",

        "mode": "maximum-speed",

        "model": "YOLO11n",

        "image_size": IMAGE_SIZE

    })


# ============================================================
# FAST DETECTION
# ============================================================

@app.route("/api/detect", methods=["POST"])
def detect():

    if model is None:

        return jsonify({

            "success": False,

            "error": "AI model is offline"

        }), 500


    if "image" not in request.files:

        return jsonify({

            "success": False,

            "error": "No image received"

        }), 400


    start = time.perf_counter()


    try:

        # ----------------------------------------------------
        # READ IMAGE DIRECTLY INTO MEMORY
        # NO DISK WRITE
        # ----------------------------------------------------

        image_bytes = request.files["image"].read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        # ----------------------------------------------------
        # YOLO INFERENCE
        # ----------------------------------------------------

        with model_lock:

            results = model.predict(

                source=image,

                imgsz=IMAGE_SIZE,

                conf=CONFIDENCE,

                iou=IOU,

                max_det=MAX_OBJECTS,

                verbose=False,

                device="cpu",

                half=False,

                augment=False,

                agnostic_nms=False

            )


        detections = []


        for result in results:

            if result.boxes is None:

                continue


            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )


                name = str(
                    model.names[class_id]
                )


                coordinates = box.xyxy[
                    0
                ].tolist()


                x1 = int(coordinates[0])
                y1 = int(coordinates[1])
                x2 = int(coordinates[2])
                y2 = int(coordinates[3])


                detections.append({

                    "class": name,

                    "confidence": round(
                        confidence,
                        3
                    ),

                    "box": {

                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2

                    }

                })


        # ----------------------------------------------------
        # POTHOLE SEARCH
        # ----------------------------------------------------

        potholes = []


        for detection in detections:

            name = (
                detection["class"]
                .lower()
                .replace("_", " ")
                .replace("-", " ")
            )


            if (
                "pothole" in name
                or "road damage" in name
                or "road_damage" in name
            ):

                potholes.append(
                    detection
                )


        elapsed = (
            time.perf_counter()
            - start
        )


        processing_ms = round(
            elapsed * 1000,
            1
        )


        # ----------------------------------------------------
        # OBJECT COUNTS
        # ----------------------------------------------------

        object_counts = {}


        for detection in detections:

            name = detection["class"]

            object_counts[name] = (
                object_counts.get(
                    name,
                    0
                ) + 1
            )


        return jsonify({

            "success": True,

            "detections":
                detections,

            "potholes":
                potholes,

            "pothole_count":
                len(potholes),

            "object_counts":
                object_counts,

            "object_count":
                len(detections),

            "processing_ms":
                processing_ms,

            "risk":
                "HIGH"
                if potholes
                else "LOW"

        })


    except Exception as error:

        print(
            "Detection error:",
            error
        )


        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "SafePath AI running at:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        threaded=True

    )