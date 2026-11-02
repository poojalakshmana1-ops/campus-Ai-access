"""
Campus AI Access - AI & CCTV Monitoring Module Prototype
Smart Campus Management System

Prototype Module: AI Person Detection, Occupancy Counting, Crowd Alert,
and Restricted Area Intrusion Monitoring.

Technology Stack:
- Python 3
- OpenCV (cv2)
- Ultralytics YOLO (Pretrained lightweight model: yolov8n.pt)

Keyboard Controls:
- Press 'Q' or 'q' to safely exit the application.
"""

import sys
import os
import time
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION SETTINGS
# Easy to configure at the top of the program without changing detection logic.
# ==============================================================================

# Path to the pretrained lightweight YOLO model
MODEL_PATH = "yolov8n.pt"

# Input source:
# - Set to 0 (integer) for default laptop webcam.
# - Set to a string path (e.g., "sample_video.mp4") for a local video file.
SOURCE = 0

# Maximum allowed people before triggering a CROWD ALERT
MAX_PEOPLE = 5

# Confidence threshold for person detection (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.50

# Predefined rectangular restricted area coordinates: (x1, y1, x2, y2)
# Ensure coordinates fit within the frame resolution (e.g. 640x480 or 768x432)
RESTRICTED_AREA = (320, 120, 620, 390)

# Window title
WINDOW_TITLE = "Campus AI Access - CCTV Monitoring Prototype"

# COCO Class ID for person is 0
PERSON_CLASS_ID = 0


# ==============================================================================
# CORE MODULE FUNCTIONS
# ==============================================================================

def load_model(model_path: str):
    """
    Load the pretrained YOLO object detection model.
    Handles loading failure gracefully.
    """
    print(f"[INFO] Loading YOLO model from '{model_path}'...")
    if not os.path.exists(model_path):
        # If not found directly, check if it exists relative to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path

    try:
        model = YOLO(model_path)
        print(f"[SUCCESS] Pretrained YOLO model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model from '{model_path}': {e}")
        return None


def initialize_video_source(source):
    """
    Initialize webcam or video file capture source.
    Returns (cap, is_connected).
    """
    print(f"[INFO] Connecting to video source: {source}...")
    
    # If source is a string representing an integer (e.g. "0"), convert it
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    try:
        cap = cv2.VideoCapture(source)
        if cap is not None and cap.isOpened():
            # Test reading a single frame to ensure actual feed availability
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                # Rewind video file back to the beginning if reading a file
                if not isinstance(source, int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print(f"[SUCCESS] Video source opened successfully. CAMERA: CONNECTED")
                return cap, True
            else:
                print(f"[WARNING] Source opened but could not read initial frame.")
                cap.release()
                return None, False
        else:
            print(f"[WARNING] Video source could not be opened. CAMERA: DISCONNECTED")
            return None, False
    except Exception as e:
        print(f"[ERROR] Exception while opening video source: {e}")
        return None, False


def detect_people(model, frame, confidence_threshold: float = 0.5):
    """
    Run YOLO inference on the frame and filter for person objects (COCO class 0).
    For each detected person:
    - Extracts bounding box (x1, y1, x2, y2)
    - Extracts confidence score
    - Computes bounding box center point (center_x, center_y)

    Returns:
        List of dicts: [{'bbox': (x1, y1, x2, y2), 'conf': float, 'center': (cx, cy)}]
    """
    detections = []
    if model is None or frame is None:
        return detections

    try:
        # Run YOLO inference in silent mode (verbose=False)
        results = model(frame, verbose=False)
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # Only process COCO class 0 ('person') above threshold
            if cls_id == PERSON_CLASS_ID and conf >= confidence_threshold:
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

                # Calculate center point of the person bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "center": (center_x, center_y)
                })

    except Exception as e:
        print(f"[ERROR] Exception during person detection: {e}")

    return detections


def check_restricted_area(detections, restricted_area: tuple):
    """
    Check each detected person's center point against the rectangular restricted area.
    Restricted area coordinates: (rx1, ry1, rx2, ry2)

    Returns:
        (restricted_detections, normal_detections)
    """
    rx1, ry1, rx2, ry2 = restricted_area
    restricted_detections = []
    normal_detections = []

    for det in detections:
        cx, cy = det["center"]
        # Check if the center point is inside the restricted zone
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            restricted_detections.append(det)
        else:
            normal_detections.append(det)

    return restricted_detections, normal_detections


def generate_restricted_area_event(restricted_count: int, timestamp: str = None):
    """
    Generate an event when one or more persons are detected inside the restricted area.
    Modular design for future backend/logging integration.
    Prints to console and returns the event payload dictionary.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_message = f"[{timestamp}] [EVENT] Restricted area entry detected! (People in zone: {restricted_count})"
    print(log_message)

    return {
        "event_type": "RESTRICTED_AREA_ENTRY",
        "timestamp": timestamp,
        "people_in_restricted_area": restricted_count,
        "message": log_message
    }


def draw_information(frame,
                     detections,
                     restricted_detections,
                     normal_detections,
                     restricted_area: tuple,
                     max_people: int,
                     is_connected: bool,
                     latest_event_str: str = None):
    """
    Render all visual monitoring elements onto the video frame:
    1. Camera Connection Status
    2. Restricted Area Zone & Status
    3. Person Bounding Boxes, Labels, Confidence Scores & Center Points
    4. Total People Count
    5. Crowd Status & Crowd Alert (when applicable)
    6. Number of People in Restricted Area
    7. Latest Security Event Banner
    """
    h, w, _ = frame.shape
    rx1, ry1, rx2, ry2 = restricted_area

    # Clamp restricted area coordinates to frame dimensions
    rx1 = max(0, min(rx1, w - 1))
    rx2 = max(0, min(rx2, w - 1))
    ry1 = max(0, min(ry1, h - 1))
    ry2 = max(0, min(ry2, h - 1))

    total_people = len(detections)
    restricted_count = len(restricted_detections)
    is_crowd_alert = total_people > max_people
    is_restricted_alert = restricted_count > 0

    # --------------------------------------------------------------------------
    # 1. DRAW RESTRICTED AREA ZONE
    # --------------------------------------------------------------------------
    # Create semi-transparent overlay for restricted zone
    zone_overlay = frame.copy()
    if is_restricted_alert:
        zone_color = (0, 0, 220)        # Bright Red when breached
        zone_border_color = (0, 0, 255)
    else:
        zone_color = (255, 180, 0)      # Amber / Cyan-Yellow when clear
        zone_border_color = (255, 200, 0)

    cv2.rectangle(zone_overlay, (rx1, ry1), (rx2, ry2), zone_color, -1)
    # Blend overlay with 18% opacity
    cv2.addWeighted(zone_overlay, 0.18, frame, 0.82, 0, frame)

    # Draw clear outline and corner accents
    border_thickness = 3 if is_restricted_alert else 2
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), zone_border_color, border_thickness)

    # Restricted Area Header Tag
    tag_text = " RESTRICTED AREA "
    tag_size, _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    tag_w, tag_h = tag_size[0] + 8, tag_size[1] + 8
    cv2.rectangle(frame, (rx1, ry1 - tag_h), (rx1 + tag_w, ry1), zone_border_color, -1)
    cv2.putText(frame, tag_text.strip(), (rx1 + 4, ry1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # 2. DRAW PERSON BOUNDING BOXES, LABELS, CONFIDENCE & CENTER POINTS
    # --------------------------------------------------------------------------
    # A) Normal people outside restricted area (Green)
    for det in normal_detections:
        bx1, by1, bx2, by2 = det["bbox"]
        cx, cy = det["center"]
        conf = det["conf"]
        box_color = (0, 230, 0)  # Bright Green

        # Draw bounding box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)

        # Label: Person + Confidence
        label = f"Person {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (bx1, by1 - lh - 6), (bx1 + lw + 6, by1), box_color, -1)
        cv2.putText(frame, label, (bx1 + 3, by1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw center point
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # Yellow-cyan center dot
        cv2.circle(frame, (cx, cy), 7, box_color, 1)

    # B) People inside restricted area (Red)
    for det in restricted_detections:
        bx1, by1, bx2, by2 = det["bbox"]
        cx, cy = det["center"]
        conf = det["conf"]
        alert_box_color = (0, 0, 255)  # Bright Red

        # Draw prominent red bounding box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), alert_box_color, 3)

        # Label: Person [RESTRICTED] + Confidence
        label = f"Person [RESTRICTED] {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (bx1, by1 - lh - 8), (bx1 + lw + 6, by1), alert_box_color, -1)
        cv2.putText(frame, label, (bx1 + 3, by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Highlight center point with target marker
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)

    # --------------------------------------------------------------------------
    # 3. TOP HUD STATUS PANEL
    # --------------------------------------------------------------------------
    panel_height = 135
    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, panel_height), (20, 24, 28), -1)
    cv2.addWeighted(hud_overlay, 0.78, frame, 0.22, 0, frame)
    cv2.line(frame, (0, panel_height), (w, panel_height), (80, 85, 95), 1)

    # Header / System Title
    cv2.putText(frame, "CAMPUS AI ACCESS | CCTV AI MONITOR", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    # Line 1: Camera Status & People Count
    # Camera Connection Status
    if is_connected:
        cam_text = "CAMERA: CONNECTED"
        cam_color = (0, 255, 0)
    else:
        cam_text = "CAMERA: DISCONNECTED"
        cam_color = (0, 0, 255)

    cv2.putText(frame, cam_text, (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cam_color, 2, cv2.LINE_AA)

    # Total People Count
    count_text = f"People Count: {total_people}"
    cv2.putText(frame, count_text, (260, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Line 2: Crowd Status
    if is_crowd_alert:
        crowd_status_text = f"Crowd Status: CROWD ALERT (Max Allowed: {max_people})"
        crowd_color = (0, 0, 255)
    else:
        crowd_status_text = f"Crowd Status: NORMAL (Max: {max_people})"
        crowd_color = (0, 255, 0)

    cv2.putText(frame, crowd_status_text, (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, crowd_color, 2, cv2.LINE_AA)

    # Line 3: Restricted Area Status & Count
    if is_restricted_alert:
        rest_status_text = f"Restricted Area: ALERT ({restricted_count} in zone)"
        rest_color = (0, 0, 255)
    else:
        rest_status_text = "Restricted Area: CLEAR"
        rest_color = (0, 255, 0)

    cv2.putText(frame, rest_status_text, (15, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, rest_color, 2, cv2.LINE_AA)

    # Number of People in Restricted Area
    in_zone_text = f"People in Restricted Area: {restricted_count}"
    cv2.putText(frame, in_zone_text, (370, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Keypress hint
    cv2.putText(frame, "Press 'Q' to Exit", (w - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # 4. LARGE ALERT BADGES (WHEN ACTIVE)
    # --------------------------------------------------------------------------
    badge_x = w - 310
    badge_y = 48

    if is_crowd_alert:
        # Draw prominent CROWD ALERT badge
        cv2.rectangle(frame, (badge_x, badge_y), (w - 15, badge_y + 36), (0, 0, 220), -1)
        cv2.rectangle(frame, (badge_x, badge_y), (w - 15, badge_y + 36), (255, 255, 255), 2)
        cv2.putText(frame, "CROWD ALERT", (badge_x + 15, badge_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        badge_y += 44

    if is_restricted_alert:
        # Draw prominent RESTRICTED AREA ALERT badge
        cv2.rectangle(frame, (badge_x - 30, badge_y), (w - 15, badge_y + 36), (0, 0, 220), -1)
        cv2.rectangle(frame, (badge_x - 30, badge_y), (w - 15, badge_y + 36), (255, 255, 255), 2)
        cv2.putText(frame, "RESTRICTED AREA ALERT", (badge_x - 20, badge_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # 5. BOTTOM EVENT LOG BANNER
    # --------------------------------------------------------------------------
    if latest_event_str:
        footer_h = 30
        footer_overlay = frame.copy()
        cv2.rectangle(footer_overlay, (0, h - footer_h), (w, h), (10, 12, 14), -1)
        cv2.addWeighted(footer_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, latest_event_str, (12, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    return frame


def render_disconnected_frame(source_name):
    """
    Generate an informational video frame when the camera/video source cannot be opened.
    """
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 30)

    # Warning Box
    cv2.rectangle(canvas, (40, 40), (600, 440), (45, 50, 60), -1)
    cv2.rectangle(canvas, (40, 40), (600, 440), (0, 0, 220), 2)

    cv2.putText(canvas, "CAMPUS AI ACCESS - CCTV MONITOR", (60, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

    # Camera status
    cv2.putText(canvas, "CAMERA: DISCONNECTED", (60, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)

    error_msg1 = f"Unable to open video source: {source_name}"
    error_msg2 = "If using webcam (SOURCE=0), ensure camera is plugged in"
    error_msg3 = "and not used by another program."
    error_msg4 = "To use a local video, set SOURCE = 'sample_video.mp4'"

    cv2.putText(canvas, error_msg1, (60, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, error_msg2, (60, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, error_msg3, (60, 265),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, error_msg4, (60, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Press 'Q' to Exit Application", (60, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 2, cv2.LINE_AA)

    return canvas


# ==============================================================================
# MAIN APPLICATION LOOP
# ==============================================================================

def main():
    """
    Main execution loop for Campus AI Access CCTV monitoring.
    """
    print("=" * 60)
    print("  CAMPUS AI ACCESS - SMART CCTV MONITORING PROTOTYPE")
    print("=" * 60)

    # Allow optional command-line source override:
    # e.g., python detection.py sample_video.mp4  OR  python detection.py 0
    active_source = SOURCE
    if len(sys.argv) > 1:
        arg_src = sys.argv[1]
        active_source = int(arg_src) if arg_src.isdigit() else arg_src
        print(f"[CONFIG] Overriding SOURCE from command line: {active_source}")
    else:
        print(f"[CONFIG] Active SOURCE: {active_source}")

    print(f"[CONFIG] MODEL_PATH: {MODEL_PATH}")
    print(f"[CONFIG] MAX_PEOPLE limit: {MAX_PEOPLE}")
    print(f"[CONFIG] CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}")
    print(f"[CONFIG] RESTRICTED_AREA: {RESTRICTED_AREA}")
    print("-" * 60)

    # 1. Load Pretrained YOLO Model
    model = load_model(MODEL_PATH)
    if model is None:
        print("[FATAL] Exiting because the YOLO model could not be loaded.")
        sys.exit(1)

    # 2. Initialize Video Source
    cap, is_connected = initialize_video_source(active_source)

    if not is_connected or cap is None:
        print("\n[CAMERA STATUS] CAMERA: DISCONNECTED")
        print("[ERROR] Could not connect to the specified camera or video file.")
        print("[HELP] If you are running on a machine without a webcam, you can run:")
        print("       python detection.py sample_video.mp4")
        print("-" * 60)

        # Show graceful disconnected screen in OpenCV window
        err_frame = render_disconnected_frame(str(active_source))
        cv2.imshow(WINDOW_TITLE, err_frame)
        print("[INFO] Showing status window. Press 'Q' or close window to exit.")
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                break
            try:
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("[INFO] Application closed cleanly.")
        return

    # 3. Process Video Stream
    print(f"[INFO] Monitoring window opened. Press 'Q' to exit.")
    last_event_time = 0.0
    latest_event_str = None
    prev_restricted_count = 0

    try:
        while True:
            ret, frame = cap.read()

            # Handle end of video stream or failed frame read
            if not ret or frame is None:
                # If reading from a video file, loop back to the beginning
                if not isinstance(active_source, int):
                    print("[INFO] Video file reached end. Looping from beginning...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[WARNING] Frame capture returned empty. Camera stream interrupted.")
                    break

            # Run Person Detection
            detections = detect_people(model, frame, CONFIDENCE_THRESHOLD)

            # Check Restricted Area
            restricted_dets, normal_dets = check_restricted_area(detections, RESTRICTED_AREA)
            current_restricted_count = len(restricted_dets)

            # Event Generation Logic:
            # Trigger event when restricted count > 0, debounced so it does not spam console
            now = time.time()
            if current_restricted_count > 0:
                # Trigger if state changed or at least 2.5 seconds have elapsed since last console log
                if current_restricted_count != prev_restricted_count or (now - last_event_time) > 2.5:
                    event_data = generate_restricted_area_event(current_restricted_count)
                    last_event_time = now
                    latest_event_str = f"LATEST EVENT: {event_data['message']}"

            prev_restricted_count = current_restricted_count

            # Draw Visual Information & HUD
            display_frame = draw_information(
                frame=frame,
                detections=detections,
                restricted_detections=restricted_dets,
                normal_detections=normal_dets,
                restricted_area=RESTRICTED_AREA,
                max_people=MAX_PEOPLE,
                is_connected=True,
                latest_event_str=latest_event_str
            )

            # Show frame in OpenCV window
            cv2.imshow(WINDOW_TITLE, display_frame)

            # Check keyboard input: 'Q' or 'q' or ESC (27) to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                print("[INFO] 'Q' pressed by user. Safely shutting down CCTV monitor...")
                break

            # Check if user closed the window using the window close button
            try:
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed by user. Safely shutting down CCTV monitor...")
                    break
            except Exception:
                pass

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in monitoring loop: {e}")
    finally:
        # Cleanly release camera and destroy windows
        if cap is not None:
            cap.release()
            print("[INFO] Video capture resource released.")
        cv2.destroyAllWindows()
        print("[INFO] OpenCV windows closed cleanly. Goodbye!")


if __name__ == "__main__":
    main()
