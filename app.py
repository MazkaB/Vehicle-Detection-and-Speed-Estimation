import os
import cv2
import math
import logging
import numpy as np
import threading
import queue
import torch  # for optional GPU usage

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import supervision as sv
from vidgear.gears import CamGear, WriteGear
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# Custom imports
from detection.homography import compute_homography_and_scale, transform_point
from detection.kalman_filter import KalmanFilter1D
from detection.vehicle_counter import VehicleCounter
from detection.yolo_detector import YOLODetector

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

app = Flask(__name__)
CORS(app)

DATABASE_URL = "sqlite:///database.db"
engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()

class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(Text)
    source_type = Column(String)  # "file" or "stream"
    source_path = Column(Text)    # path to local file or HLS URL

class Stats(Base):
    __tablename__ = "stats"
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer)
    max_speed = Column(Float, default=0.0)
    min_speed = Column(Float, default=0.0)
    total_count = Column(Integer, default=0)
    count_car = Column(Integer, default=0)
    count_truck = Column(Integer, default=0)
    count_motorcycle = Column(Integer, default=0)
    count_bus = Column(Integer, default=0)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

MODEL_PATH = os.path.join("models", "yolo11m.pt")

yolo_detector = YOLODetector(MODEL_PATH)


@app.route("/api/videos", methods=["GET"])
def get_videos():
    """Return a list of all videos."""
    videos = session.query(Video).all()
    data = []
    for v in videos:
        data.append({
            "id": v.id,
            "name": v.name,
            "description": v.description,
            "source_type": v.source_type,
            "source_path": v.source_path
        })
    return jsonify(data)

@app.route("/api/videos", methods=["POST"])
def upload_video():
    name = request.form.get("name", "")
    description = request.form.get("description", "")
    hls_link = request.form.get("hls_link", None)
    file = request.files.get("file", None)

    if file:
        filename = file.filename
        save_path = os.path.join("static", "uploads", filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)

        new_video = Video(
            name=name,
            description=description,
            source_type="file",
            source_path=save_path
        )
    elif hls_link:
        new_video = Video(
            name=name,
            description=description,
            source_type="stream",
            source_path=hls_link
        )
    else:
        return jsonify({"error": "No file or HLS link provided"}), 400

    session.add(new_video)
    session.commit()

    # Initialize stats row
    stats = Stats(
        video_id=new_video.id,
        max_speed=0.0,
        min_speed=9999.0,  # We'll treat 9999.0 as 'not updated' yet
        total_count=0
    )
    session.add(stats)
    session.commit()

    return jsonify({"message": "Video added successfully", "video_id": new_video.id})

@app.route("/api/videos/<int:video_id>/stats", methods=["GET"])
def get_video_stats(video_id):
    """Return stats for a given video by ID."""
    stats = session.query(Stats).filter_by(video_id=video_id).first()
    if not stats:
        return jsonify({"error": "No stats found"}), 404

    # If min_speed is still 9999.0, treat as 0
    min_speed = stats.min_speed if stats.min_speed != 9999.0 else 0.0

    data = {
        "max_speed": stats.max_speed,
        "min_speed": min_speed,
        "total_count": stats.total_count,
        "count_car": stats.count_car,
        "count_truck": stats.count_truck,
        "count_motorcycle": stats.count_motorcycle,
        "count_bus": stats.count_bus,
    }
    return jsonify(data)

@app.route("/api/videos/<int:video_id>", methods=["DELETE"])
def delete_video(video_id):
    """Delete a video from the database, including its Stats entry."""
    video = session.query(Video).filter_by(id=video_id).first()
    if not video:
        return jsonify({"error": "Video not found"}), 404

    stats = session.query(Stats).filter_by(video_id=video_id).first()
    if stats:
        session.delete(stats)

    session.delete(video)
    session.commit()
    return jsonify({"message": f"Video ID={video_id} deleted"}), 200



HLS_BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "hls_output"))
os.makedirs(HLS_BASE_DIR, exist_ok=True)

QUEUE_MAXSIZE = 200

def frame_reader(cap, frame_queue):
    """ Continuously read frames from cap and push them into frame_queue. """
    while True:
        frame = cap.read()
        if frame is None:
            # End of video or stream
            break
        try:
            frame_queue.put(frame, timeout=1)
        except queue.Full:
            logging.warning("Frame queue is full; dropping frame")
    # Signal the end of frames
    frame_queue.put(None)

def detection_writer(video_id, frame_queue):
    """
    Pull frames from frame_queue, run YOLO + annotation,
    write them to HLS segments.
    """
    # 1) Get DB + video info
    video = session.query(Video).filter_by(id=video_id).first()
    if not video:
        logging.error(f"No video found with ID={video_id}")
        return

    stats = session.query(Stats).filter_by(video_id=video_id).first()
    if not stats:
        stats = Stats(video_id=video_id)
        session.add(stats)
        session.commit()

    source = video.source_path
    cap_test = CamGear(source=source, stream_mode=(video.source_type == "stream")).start()
    first_frame = cap_test.read()
    cap_test.stop()
    if first_frame is None:
        logging.error("Could not read first_frame for homography.")
        return

    # Homography
    H, scale = compute_homography_and_scale(first_frame)
    fps = 30.0
    dt = 1.0 / fps

    tracker = sv.ByteTrack()
    tracker.frame_rate = fps
    box_annotator = sv.BoxAnnotator(color=sv.Color(255, 0, 0))
    vehicle_counter = VehicleCounter()
    object_info = {}

    # Prepare HLS output
    video_hls_dir = os.path.join(HLS_BASE_DIR, f"video_{video_id}")
    os.makedirs(video_hls_dir, exist_ok=True)
    playlist_path = os.path.join(video_hls_dir, "stream.m3u8")

    output_params = {
        "-f": "hls",
        "-hls_time": "2",
        "-hls_list_size": "0",
        "-hls_segment_filename": os.path.join(video_hls_dir, "segment_%03d.ts"),
        "-hls_flags": "delete_segments+append_list",
        "-hls_playlist_type": "event",
        "-c:v": "libx264",
        "-preset": "veryfast",
        "-g": "60",
        "-vf": "scale=1280:720",
        "-pix_fmt": "yuv420p",
        "-r": str(fps)
    }

    writer = WriteGear(
        output=playlist_path,
        compression_mode=True,
        logging=True,
        **output_params
    )

    frame_idx = 0

    while True:
        frame = frame_queue.get()
        if frame is None:
            # End of stream
            break

        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yolo_results = yolo_detector.predict(frame_rgb)
        detections = sv.Detections.from_ultralytics(yolo_results)
        class_names = [yolo_detector.model.names[cls_id] for cls_id in detections.class_id]
        detections.data["class_name"] = np.array(class_names)

        tracked = tracker.update_with_detections(detections)

        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=tracked
        )

        for i, box in enumerate(tracked.xyxy):
            x1, y1, x2, y2 = box
            tracker_id = tracked.tracker_id[i]
            cls_name = tracked.data["class_name"][i]

            vehicle_counter.increment_class_count(tracker_id, cls_name)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            h, w = annotated_frame.shape[:2]
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))

            world_pt = transform_point((cx, cy), H, scale)
            if tracker_id not in object_info:
                object_info[tracker_id] = {
                    "prev_position": world_pt,
                    "kalman": KalmanFilter1D(initial_value=0.0)
                }
                speed_smoothed = 0.0
            else:
                prev = object_info[tracker_id]["prev_position"]
                kf = object_info[tracker_id]["kalman"]
                dx = world_pt[0] - prev[0]
                dy = world_pt[1] - prev[1]
                displacement = math.sqrt(dx*dx + dy*dy)
                speed_ms = displacement / dt
                speed_smoothed = kf.update(speed_ms)
                object_info[tracker_id]["prev_position"] = world_pt

            speed_kmh = speed_smoothed * 3.6
            label = f"{cls_name} ID:{tracker_id}, {speed_kmh:.1f} km/h"
            cv2.putText(
                annotated_frame, label, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            if speed_kmh > stats.max_speed:
                stats.max_speed = speed_kmh
            if speed_kmh < stats.min_speed:
                stats.min_speed = speed_kmh

        stats.total_count = vehicle_counter.total_count
        stats.count_car = vehicle_counter.class_count.get("car", 0)
        stats.count_truck = vehicle_counter.class_count.get("truck", 0)
        stats.count_motorcycle = vehicle_counter.class_count.get("motorcycle", 0)
        stats.count_bus = vehicle_counter.class_count.get("bus", 0)

        if frame_idx % 30 == 0:
            session.commit()

        writer.write(annotated_frame)

    session.commit()
    writer.close()


def generate_hls_segments(video_id):
    # 1) open the source
    video = session.query(Video).filter_by(id=video_id).first()
    if not video:
        raise ValueError(f"No video with id={video_id}")
    source = video.source_path
    cap = CamGear(source=source, stream_mode=(video.source_type == "stream")).start()

    # 2) build a queue
    frame_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    t_reader = threading.Thread(target=frame_reader, args=(cap, frame_queue))
    t_writer = threading.Thread(target=detection_writer, args=(video_id, frame_queue))

    # 5) start both
    t_reader.start()
    t_writer.start()

    t_reader.join()
    t_writer.join()

    # 7) release capture
    cap.stop()

def generate_hls_segments_background(video_id):
    try:
        generate_hls_segments(video_id)
    except Exception as e:
        logging.exception(f"Background HLS generation error for video {video_id}: {e}")

stream_threads = {}

@app.route("/api/videos/<int:video_id>/start_stream", methods=["GET"])
def start_stream(video_id):
    if video_id in stream_threads and stream_threads[video_id].is_alive():
        return jsonify({
            "message": f"Stream already running for video {video_id}",
            "m3u8_url": f"/hls/video_{video_id}/stream.m3u8"
        })

    t = threading.Thread(target=generate_hls_segments_background, args=(video_id,))
    t.start()
    stream_threads[video_id] = t

    return jsonify({
        "message": f"Started HLS generation in background for video {video_id}",
        "m3u8_url": f"/hls/video_{video_id}/stream.m3u8"
    })

@app.route("/hls/<path:filename>", methods=["GET"])
def serve_hls_file(filename):
    """Serve .m3u8 and .ts segments from disk."""
    return send_from_directory(HLS_BASE_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
