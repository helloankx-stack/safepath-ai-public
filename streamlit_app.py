import os
import time
import threading

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SafePath AI",
    page_icon="🛣️",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f1e8;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 20px;
    }

    .title-box {
        background: #fffdf8;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #ddd6c9;
        margin-bottom: 18px;
    }

    .title-box h1 {
        color: #263633;
        margin: 0;
    }

    .title-box p {
        color: #737b77;
        margin-top: 5px;
    }

    .safe-box {
        background: #edf7f3;
        color: #176b62;
        padding: 16px;
        border-radius: 12px;
        font-weight: bold;
    }

    .danger-box {
        background: #fff0ed;
        color: #a63225;
        padding: 16px;
        border-radius: 12px;
        font-weight: bold;
        border: 1px solid #efc6be;
    }

    .warning-box {
        background: #fff8df;
        color: #876b00;
        padding: 16px;
        border-radius: 12px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="title-box">
        <h1>🛣️ SafePath AI</h1>
        <p>Hidden Problem Detection System</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL PATH
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DIR = os.path.join(
    BASE_DIR,
    "backend"
)

POTHOLE_MODEL_PATH = os.path.join(
    BACKEND_DIR,
    "pothole.pt"
)

GENERAL_MODEL_PATH = os.path.join(
    BACKEND_DIR,
    "yolo11n.pt"
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    pothole = None
    general = None

    if YOLO is None:
        return None, None

    if os.path.exists(POTHOLE_MODEL_PATH):

        try:
            pothole = YOLO(
                POTHOLE_MODEL_PATH
            )
        except Exception:
            pothole = None

    if os.path.exists(GENERAL_MODEL_PATH):

        try:
            general = YOLO(
                GENERAL_MODEL_PATH
            )
        except Exception:
            general = None

    return pothole, general


pothole_model, general_model = load_models()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("SafePath Control")

    monitoring = st.checkbox(
        "🧠 AI Monitoring",
        value=True
    )

    alerts_enabled = st.checkbox(
        "🔔 Hazard Alert",
        value=True
    )

    object_detection = st.checkbox(
        "👤 Object Detection",
        value=False
    )

    st.divider()

    st.subheader("AI Performance")

    target_fps = st.slider(
        "AI FPS",
        1.0,
        4.0,
        2.0,
        0.5
    )

    confidence = st.slider(
        "Pothole Confidence",
        0.20,
        0.80,
        0.40,
        0.05
    )

    st.divider()

    st.write("📱 Mobile Optimized")
    st.write("⚡ Low CPU Mode")
    st.write("🎥 Smooth Camera")
    st.write("🕳️ Pothole First")


# =========================================================
# LIVE PROCESSOR
# =========================================================

class SafePathProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()

        self.latest_frame = None

        self.pothole_boxes = []

        self.object_boxes = []

        self.inference_time = 0.0

        self.current_fps = 0.0

        self.last_confidence = 0.0

        self.error_message = ""

        self.stop_event = threading.Event()

        self.monitoring = True

        self.object_detection = False

        self.target_fps = 2.0

        self.confidence = 0.40

        self.worker = threading.Thread(
            target=self.ai_loop,
            daemon=True
        )

        self.worker.start()


    # =====================================================
    # CAMERA FRAME
    # =====================================================

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        # Store ONLY the newest frame.
        with self.lock:
            self.latest_frame = image.copy()

            potholes = list(
                self.pothole_boxes
            )

            objects = list(
                self.object_boxes
            )


        # =================================================
        # OBJECT BOXES
        # =================================================

        for item in objects:

            x1, y1, x2, y2, name, conf = item

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (45, 110, 95),
                2
            )

            cv2.putText(
                image,
                name + " " + str(round(conf * 100)) + "%",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (45, 110, 95),
                2
            )


        # =================================================
        # POTHOLE BOXES
        # =================================================

        for item in potholes:

            x1, y1, x2, y2, name, conf = item

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (40, 60, 210),
                3
            )

            cv2.putText(
                image,
                "POTHOLE " + str(round(conf * 100)) + "%",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (40, 60, 210),
                2
            )


        # =================================================
        # WARNING
        # =================================================

        if len(potholes) > 0:

            cv2.rectangle(
                image,
                (0, 0),
                (image.shape[1], 52),
                (40, 60, 210),
                -1
            )

            cv2.putText(
                image,
                "WARNING: POTHOLE DETECTED",
                (15, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2
            )


        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


    # =====================================================
    # AI LOOP
    # =====================================================

    def ai_loop(self):

        last_run = 0.0

        while not self.stop_event.is_set():

            if not self.monitoring:

                time.sleep(0.10)

                continue


            fps = max(
                1.0,
                self.target_fps
            )

            interval = 1.0 / fps

            current_time = time.perf_counter()

            if current_time - last_run < interval:

                time.sleep(0.01)

                continue


            with self.lock:

                if self.latest_frame is None:

                    frame = None

                else:

                    frame = self.latest_frame.copy()


            if frame is None:

                time.sleep(0.02)

                continue


            last_run = current_time

            start = time.perf_counter()


            try:

                # -----------------------------------------
                # RESIZE
                # -----------------------------------------

                original_height = frame.shape[0]

                original_width = frame.shape[1]

                max_dimension = max(
                    original_height,
                    original_width
                )

                scale = (
                    256.0 /
                    max_dimension
                )


                if scale < 1.0:

                    small = cv2.resize(
                        frame,
                        (
                            int(
                                original_width *
                                scale
                            ),
                            int(
                                original_height *
                                scale
                            )
                        )
                    )

                else:

                    small = frame


                rgb = cv2.cvtColor(
                    small,
                    cv2.COLOR_BGR2RGB
                )


                scale_x = (
                    original_width /
                    small.shape[1]
                )

                scale_y = (
                    original_height /
                    small.shape[0]
                )


                new_potholes = []

                new_objects = []


                # =========================================
                # POTHOLE MODEL
                # =========================================

                if pothole_model is not None:

                    results = pothole_model.predict(
                        source=rgb,
                        imgsz=256,
                        conf=self.confidence,
                        max_det=5,
                        device="cpu",
                        verbose=False,
                        augment=False
                    )


                    for result in results:

                        if result.boxes is None:
                            continue


                        for box in result.boxes:

                            confidence_value = float(
                                box.conf[0]
                            )

                            class_id = int(
                                box.cls[0]
                            )

                            name = str(
                                result.names.get(
                                    class_id,
                                    "pothole"
                                )
                            )

                            coordinates = (
                                box.xyxy[0]
                                .tolist()
                            )

                            x1 = int(
                                coordinates[0] *
                                scale_x
                            )

                            y1 = int(
                                coordinates[1] *
                                scale_y
                            )

                            x2 = int(
                                coordinates[2] *
                                scale_x
                            )

                            y2 = int(
                                coordinates[3] *
                                scale_y
                            )

                            new_potholes.append(
                                (
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    name,
                                    confidence_value
                                )
                            )


                # =========================================
                # GENERAL OBJECT MODEL
                # =========================================

                if (
                    self.object_detection
                    and
                    general_model is not None
                ):

                    results = general_model.predict(
                        source=rgb,
                        imgsz=256,
                        conf=0.45,
                        max_det=8,
                        device="cpu",
                        verbose=False,
                        augment=False
                    )


                    for result in results:

                        if result.boxes is None:
                            continue


                        for box in result.boxes:

                            confidence_value = float(
                                box.conf[0]
                            )

                            class_id = int(
                                box.cls[0]
                            )

                            name = str(
                                result.names.get(
                                    class_id,
                                    "object"
                                )
                            )

                            coordinates = (
                                box.xyxy[0]
                                .tolist()
                            )

                            x1 = int(
                                coordinates[0] *
                                scale_x
                            )

                            y1 = int(
                                coordinates[1] *
                                scale_y
                            )

                            x2 = int(
                                coordinates[2] *
                                scale_x
                            )

                            y2 = int(
                                coordinates[3] *
                                scale_y
                            )

                            new_objects.append(
                                (
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    name,
                                    confidence_value
                                )
                            )


                # =========================================
                # PERFORMANCE
                # =========================================

                elapsed = (
                    time.perf_counter() -
                    start
                ) * 1000.0


                calculated_fps = (
                    1000.0 / elapsed
                    if elapsed > 0
                    else 0.0
                )


                with self.lock:

                    self.pothole_boxes = (
                        new_potholes
                    )

                    self.object_boxes = (
                        new_objects
                    )

                    self.inference_time = (
                        elapsed
                    )

                    self.current_fps = min(
                        calculated_fps,
                        self.target_fps
                    )

                    self.last_confidence = max(
                        (
                            item[5]
                            for item in new_potholes
                        ),
                        default=0.0
                    )

                    self.error_message = ""


            except Exception as error:

                with self.lock:

                    self.error_message = str(
                        error
                    )[:200]


    # =====================================================
    # STOP
    # =====================================================

    def stop(self):

        self.stop_event.set()


# =========================================================
# CAMERA AREA
# =========================================================

left, right = st.columns(
    [2.2, 1]
)


with left:

    st.subheader(
        "🎥 Live Road Camera"
    )


    context = webrtc_streamer(

        key="safepath_camera",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=(
            SafePathProcessor
        ),

        media_stream_constraints={

            "video": {

                "width": {
                    "ideal": 640,
                    "max": 640
                },

                "height": {
                    "ideal": 480,
                    "max": 480
                },

                "frameRate": {
                    "ideal": 15,
                    "max": 15
                }

            },

            "audio": False
        },

        rtc_configuration={

            "iceServers": [

                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }

            ]

        }
    )


    if context.video_processor:

        processor = (
            context.video_processor
        )

        processor.monitoring = (
            monitoring
        )

        processor.object_detection = (
            object_detection
        )

        processor.target_fps = (
            target_fps
        )

        processor.confidence = (
            confidence
        )


        with processor.lock:

            potholes = list(
                processor.pothole_boxes
            )

            objects = list(
                processor.object_boxes
            )

            inference_time = (
                processor.inference_time
            )

            current_fps = (
                processor.current_fps
            )

            last_confidence = (
                processor.last_confidence
            )

            error_message = (
                processor.error_message
            )

    else:

        potholes = []

        objects = []

        inference_time = 0.0

        current_fps = 0.0

        last_confidence = 0.0

        error_message = ""


    if error_message:

        st.error(
            "AI Error: " + error_message
        )


# =========================================================
# SAFETY PANEL
# =========================================================

with right:

    st.subheader(
        "🚨 Road Safety"
    )


    if potholes:

        confidence_text = (
            str(
                round(
                    last_confidence * 100
                )
            )
            + "%"
        )


        if alerts_enabled:

            st.markdown(
                f"""
                <div class="danger-box">
                    🚨 POTHOLE DETECTED
                    <br><br>
                    Confidence: {confidence_text}
                    <br>
                    Slow down and avoid the hazard.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="warning-box">
                    ⚠️ POTHOLE DETECTED
                    <br><br>
                    Confidence: {confidence_text}
                </div>
                """,
                unsafe_allow_html=True
            )


    elif pothole_model is None:

        st.markdown(
            """
            <div class="warning-box">
                🟡 POTHOLE MODEL REQUIRED
                <br><br>
                Add:
                <br>
                backend/pothole.pt
            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.markdown(
            """
            <div class="safe-box">
                🟢 ROAD CLEAR
                <br><br>
                No pothole detected
                in the latest AI scan.
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# TELEMETRY
# =========================================================

st.markdown(
    "### 📊 Live Telemetry"
)

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "AI Status",
    "ONLINE"
    if monitoring
    else
    "PAUSED"
)


c2.metric(
    "Objects",
    str(len(objects))
)


c3.metric(
    "Potholes",
    str(len(potholes))
)


if potholes:

    risk = "HIGH"

else:

    risk = "LOW"


c4.metric(
    "Risk",
    risk
)


# =========================================================
# PERFORMANCE
# =========================================================

st.markdown(
    "### ⚡ Performance"
)

p1, p2, p3 = st.columns(3)


p1.metric(
    "AI FPS",
    str(
        round(
            current_fps,
            1
        )
    )
)


p2.metric(
    "Inference",
    str(
        round(
            inference_time
        )
    )
    + " ms"
)


p3.metric(
    "Target",
    str(
        target_fps
    )
    + " FPS"
)


# =========================================================
# DETECTION STATUS
# =========================================================

st.markdown(
    "### 🔍 Detection System"
)

d1, d2, d3 = st.columns(3)


with d1:

    st.info(
        "👤 Person Detection\n\n"
        +
        (
            "ACTIVE"
            if object_detection
            else
            "OFF"
        )
    )


with d2:

    st.info(
        "🚗 Vehicle Detection\n\n"
        +
        (
            "ACTIVE"
            if object_detection
            else
            "OFF"
        )
    )


with d3:

    st.info(
        "🕳️ Pothole AI\n\n"
        +
        (
            "ACTIVE"
            if pothole_model is not None
            else
            "MODEL REQUIRED"
        )
    )


# =========================================================
# LOCATION
# =========================================================

st.markdown(
    "### 📍 Location"
)

st.info(
    "GPS integration will be connected in the next stage."
)


st.caption(
    "SafePath AI • Hidden Problem Detection System"
)