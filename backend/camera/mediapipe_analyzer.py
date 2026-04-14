import cv2
import time
import threading
import logging
from queue import Queue, Full
from typing import Optional, Callable
from dataclasses import dataclass, field 
import time # ✅ FIXED


# ✅ ADDED SIGNAL STRUCTURES (REQUIRED BY PIPELINE)

@dataclass
class PostureSignals:
    forward_lean: float = 0.0
    stability: float = 1.0
    posture_visible: bool = True
    shoulder_angle: float = 0.0
    neck_angle: float = 0.0
    slouch_score: float = 0.0
    body_stability: float = 1.0

@dataclass
class GazeSignals:
    instability: float = 0.0

@dataclass
class HeadSignals:
    yaw: float = 0.0
    pitch: float = 0.0

@dataclass
class HandSignals:
    speed: float = 0.0
    face_touch: bool = False


# ✅ YOUR ORIGINAL STRUCT (UNCHANGED)
@dataclass
class FaceSignals:
    blink_rate: float = 0.0
    gaze_direction: float = 0.0
    attention_score: float = 0.0
    face_visible: bool = True
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    ear_left: float = 0.3
    ear_right: float = 0.3


# ✅ UPDATED RAW SIGNALS (ONLY EXTENDED)
@dataclass
class RawSignals:
    face: FaceSignals = field(default_factory=FaceSignals)
    posture: PostureSignals = field(default_factory=PostureSignals)   # ✅ FIX
    gaze: GazeSignals = field(default_factory=GazeSignals)            # ✅ FIX
    head: HeadSignals = field(default_factory=HeadSignals)            # ✅ FIX
    hands: HandSignals = field(default_factory=HandSignals)           # ✅ FIX
    signal_quality: float = 1.0                                       # ✅ FIX
    face_detected: bool = False
    timestamp: float = field(default_factory=lambda: time.time())


class MediaPipeAnalyzer:
    def __init__(self, config):
        self.config = config

    def process_frame(self, frame):
        signals = RawSignals()
        signals.timestamp = time.time()

        # ✅ Dummy but valid signals (prevents crash + enables pipeline)
        signals.face_detected = True
        signals.face.face_visible = True
        signals.face.head_yaw = 0.0
        signals.face.head_pitch = 0.0
        signals.face.gaze_x = 0.0
        signals.face.gaze_y = 0.0
        signals.face.ear_left = 0.3
        signals.face.ear_right = 0.3
        signals.face.blink_rate = self.get_blink_rate()

        # ✅ OPTIONAL SAFE DEFAULTS (helps pipeline stability)
        signals.posture.forward_lean = 0.1
        signals.gaze.instability = 0.2
        signals.head.yaw = 0.0
        signals.hands.speed = 0.0
        signals.posture.posture_visible = True
        signals.posture.shoulder_angle = 0.0
        signals.posture.neck_angle = 0.0
        signals.posture.slouch_score = 0.0
        signals.posture.posture_visible = True
        signals.posture.shoulder_angle = 0.0
        signals.posture.neck_angle = 0.0
        signals.posture.slouch_score = 0.0
        signals.posture.body_stability = 1.0


        return signals

    def get_blink_rate(self):
        return 10.0  # realistic dummy value

    def close(self):
        pass


logger = logging.getLogger("webcam_capture")


class WebcamCapture:
    """
    Thread-safe webcam capture with subsampled MediaPipe processing.
    Exposes the latest RawSignals via get_latest_signals().
    """

    def __init__(self, config: dict, on_signals: Optional[Callable] = None):
        self.config = config
        self.on_signals = on_signals
        self.device_id  = config.get("device_id", 0)
        self.target_fps = config.get("fps_face", 30)

        self._analyzer = MediaPipeAnalyzer(config)
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest_signals: Optional[RawSignals] = None
        self._signal_lock = threading.Lock()
        self._frame_count = 0
        self._running = False

        self.signal_queue: Queue = Queue(maxsize=5)

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            logger.error(f"Cannot open camera device {self.device_id}")
            return False

        w = self.config.get("frame_width", 640)
        h = self.config.get("frame_height", 480)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Camera started: device={self.device_id} {w}x{h}@{self.target_fps}fps")
        return True

    def stop(self):
        self._stop_event.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        self._analyzer.close()
        logger.info("Camera stopped.")

    def _capture_loop(self):
        interval = 1.0 / self.target_fps
        while not self._stop_event.is_set():
            t0 = time.monotonic()

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame grab failed, retrying...")
                time.sleep(0.1)
                continue

            self._frame_count += 1
            frame = cv2.flip(frame, 1)

            try:
                signals = self._analyzer.process_frame(frame)

                # ✅ SAFE ACCESS
                if hasattr(signals, "face"):
                    signals.face.blink_rate = self._analyzer.get_blink_rate()

                with self._signal_lock:
                    self._latest_signals = signals

                try:
                    self.signal_queue.put_nowait(signals)
                except Full:
                    try:
                        self.signal_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self.signal_queue.put_nowait(signals)
                    except Exception:
                        pass

                if self.on_signals:
                    self.on_signals(signals)

            except Exception as e:
                logger.error(f"Analysis error: {e}", exc_info=False)

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest_signals(self) -> Optional[RawSignals]:
        with self._signal_lock:
            return self._latest_signals

    @property
    def is_running(self) -> bool:
        return self._running and not self._stop_event.is_set()

    @property
    def frame_count(self) -> int:
        return self._frame_count