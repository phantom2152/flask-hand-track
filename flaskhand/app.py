"""
Flask application for webcam-based drawing with hand tracking and image analysis.
"""
from flask import Flask, render_template, request
from flask_socketio import SocketIO
import cv2
import os
import numpy as np
import base64
from engineio.payload import Payload
from database import DrawingDatabase
from gemini_helper import GeminiHelper
from hand_tracker import HandTracker
from dotenv import load_dotenv

load_dotenv()


# Increase max payload size for WebSocket
Payload.max_decode_packets = 500

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['GEMINI_API_KEY'] = os.getenv('GEMINI_API')
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=600)

# Initialize hand tracker
hand_tracker = HandTracker()


def process_base64_image(base64_string, flip_horizontal=True):
    """Process and optionally flip a base64 encoded image."""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Flip the image horizontally if needed
    if flip_horizontal and image is not None:
        image = cv2.flip(image, 1)

    return image


def encode_image_to_base64(image):
    """Convert an OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/drawings')
def drawings_list():
    db = DrawingDatabase()
    try:
        drawings = db.get_all_drawings()
        return render_template('drawings.html', drawings=drawings)
    finally:
        db.close()


@app.route('/drawings/<int:drawing_id>')
def drawing_detail(drawing_id):
    db = DrawingDatabase()
    try:
        drawing = db.get_drawing(drawing_id)
        if drawing is None:
            return "Drawing not found", 404
        return render_template('drawing_detail.html', drawing=drawing)
    finally:
        db.close()


@socketio.on('process_frame')
def handle_frame(data):
    try:
        # Get frame data - don't flip for live preview
        frame = process_base64_image(data['frame'], flip_horizontal=False)
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


@socketio.on('save_drawing')
def handle_save_drawing(data):
    try:
        if not data or 'image' not in data:
            raise ValueError("No image data received")

        image_data = data.get('image')
        if not image_data:
            raise ValueError("Empty image data received")

        # Process the image and flip it horizontally
        image = process_base64_image(image_data, flip_horizontal=True)
        if image is None:
            raise ValueError("Failed to process image")

        # Convert back to base64
        image_data = encode_image_to_base64(image)

        # Initialize components
        db = DrawingDatabase()
        gemini = GeminiHelper(app.config.get('GEMINI_API_KEY'))

        try:
            # Get analysis if API key is configured
            analysis = None
            if app.config.get('GEMINI_API_KEY'):
                try:
                    analysis = gemini.analyze_image(image_data)
                except Exception as e:
                    print(f"Gemini analysis failed: {str(e)}")

            # Save to database
            drawing_id = db.save_drawing(image_data, analysis)

            # Return success response
            socketio.emit('drawing_saved', {
                'status': 'success',
                'drawing_id': drawing_id,
                'image_data': image_data,
                'analysis': analysis
            }, room=request.sid)

        finally:
            db.close()

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


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, port=5001)
