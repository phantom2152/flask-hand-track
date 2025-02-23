from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
from engineio.payload import Payload
from database import DrawingDatabase
from gemini_helper import GeminiHelper
from hand_tracker import HandTracker
import threading
import re

# Increase max payload size for WebSocket
Payload.max_decode_packets = 500

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['GEMINI_API_KEY'] = "AIzaSyAkWn9ijAQyJCC2mSn_pmi3fsko6IYrMRs"
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=600)

# Initialize hand tracker
hand_tracker = HandTracker()


@app.route('/')
def index():
    return render_template('index.html')


def process_base64_image(base64_string):
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_string)

    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


@socketio.on('process_frame')
def handle_frame(data):
    try:
        # Get frame data
        frame = process_base64_image(data['frame'])
        if frame is None:
            raise ValueError("Invalid frame data")

        # Process with hand tracker
        hand_data = hand_tracker.process_and_encode_frame(frame)

        # Send processed data back to client
        socketio.emit('frame_processed', {
            'hand_data': hand_data
        }, room=request.sid)

    except Exception as e:
        print(f"Error processing frame: {e}")
        socketio.emit('frame_processed', {
            'error': str(e)
        }, room=request.sid)


@socketio.on('get_drawings')
def handle_get_drawings():
    try:
        db = DrawingDatabase()
        drawings = db.get_all_drawings()
        socketio.emit('drawings_list', drawings, room=request.sid)
    except Exception as e:
        print(f"Error fetching drawings: {e}")
        socketio.emit('drawings_list', [], room=request.sid)
    finally:
        if 'db' in locals():
            db.close()


@socketio.on('save_drawing')
def handle_save_drawing(data):
    try:
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
        }, room=request.sid)

    except ValueError as ve:
        print(f"Validation error: {str(ve)}")
        socketio.emit('drawing_saved', {
            'status': 'error',
            'message': str(ve)
        }, room=request.sid)
    except Exception as e:
        print(f"Error handling drawing data: {str(e)}")
        socketio.emit('drawing_saved', {
            'status': 'error',
            'message': 'Failed to process drawing'
        }, room=request.sid)
    finally:
        if 'db' in locals():
            db.close()


if __name__ == '__main__':
    try:
        socketio.run(app, debug=True, allow_unsafe_werkzeug=True, port=5001)
    finally:
        print("Server shutting down")
