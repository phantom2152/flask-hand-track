from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
from engineio.payload import Payload
from database import DrawingDatabase
from gemini_helper import GeminiHelper
from hand_tracker import HandTracker
import threading

# Increase max payload size for WebSocket
Payload.max_decode_packets = 500

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=600)
app.config['GEMINI_API_KEY'] = "AIzaSyAkWn9ijAQyJCC2mSn_pmi3fsko6IYrMRs"
# Global variables
camera = None
hand_tracker = HandTracker()
video_thread = None
thread_lock = threading.Lock()
stop_thread = False


def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def video_stream():
    global stop_thread
    camera = get_camera()

    while not stop_thread:
        success, frame = camera.read()
        if not success:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Process frame with hand tracker
        try:
            frame_data = hand_tracker.process_and_encode_frame(frame)
            socketio.emit('video_frame', frame_data)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        socketio.sleep(0.04)  # 25 FPS - slightly slower but more stable


def stop_camera():
    global camera, stop_thread
    stop_thread = True
    if camera is not None:
        camera.release()
        camera = None


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    stop_camera()


@socketio.on('start-video')
def handle_start_video():
    global video_thread, stop_thread
    print('Starting video stream')

    with thread_lock:
        if video_thread is None:
            stop_thread = False
            video_thread = socketio.start_background_task(video_stream)


@socketio.on('stop-video')
def handle_stop_video():
    print('Stopping video stream')
    stop_camera()


@socketio.on('drawing_data')
def handle_drawing_data(data):
    try:
        # Validate input data
        if not data or 'image' not in data:
            raise ValueError("No image data received")

        image_data = data.get('image')
        if not image_data:
            raise ValueError("Empty image data received")

        # Initialize database and Gemini
        db = DrawingDatabase()
        gemini = GeminiHelper(app.config.get('GEMINI_API_KEY'))

        # Analyze with Gemini if configured
        analysis = None
        if app.config.get('GEMINI_API_KEY'):
            try:
                analysis = gemini.analyze_image(image_data)
            except Exception as e:
                print(f"Gemini analysis failed: {str(e)}")
                # Continue with saving even if analysis fails

        # Save to database
        try:
            drawing_id = db.save_drawing(image_data, analysis)
        except Exception as e:
            raise Exception(f"Failed to save drawing to database: {str(e)}")

        # Return success response
        socketio.emit('drawing_saved', {
            'status': 'success',
            'drawing_id': drawing_id,
            'analysis': analysis
        })

    except ValueError as ve:
        print(f"Validation error: {str(ve)}")
        socketio.emit('drawing_saved', {
            'status': 'error',
            'message': str(ve)
        })
    except Exception as e:
        print(f"Error handling drawing data: {str(e)}")
        socketio.emit('drawing_saved', {
            'status': 'error',
            'message': 'Failed to process drawing'
        })
    finally:
        if 'db' in locals():
            db.close()


if __name__ == '__main__':
    try:
        socketio.run(app, debug=True, allow_unsafe_werkzeug=True, port=5001)
    finally:
        stop_camera()
