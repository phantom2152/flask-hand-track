import streamlit as st
import cv2
import numpy as np
from hand_tracker import HandTracker
from database import DrawingDatabase
from gemini_helper import GeminiHelper
import base64
from io import BytesIO
from PIL import Image


def initialize_state():
    if 'canvas' not in st.session_state:
        st.session_state.canvas = None
    if 'drawing' not in st.session_state:
        st.session_state.drawing = False
    if 'prev_point' not in st.session_state:
        st.session_state.prev_point = None
    if 'save_triggered' not in st.session_state:
        st.session_state.save_triggered = False
    if 'view_history_triggered' not in st.session_state:
        st.session_state.view_history_triggered = False
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = True
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None


def clear_canvas():
    if st.session_state.canvas is not None:
        st.session_state.canvas = np.zeros_like(st.session_state.canvas)
    st.session_state.camera_active = True
    st.session_state.current_analysis = None
    st.session_state.view_history_triggered = False


def save_drawing_callback():
    st.session_state.save_triggered = True
    st.session_state.camera_active = False


def view_history_callback():
    st.session_state.view_history_triggered = True
    st.session_state.camera_active = False


def display_drawing_history(frame_placeholder, db):
    st.header("Drawing History")
    drawings = db.get_all_drawings()

    if not drawings:
        st.info("No drawings saved yet!")
        return

    st.write("Click 'Clear Canvas' to resume drawing")
    for idx, drawing in enumerate(drawings):
        with st.expander(f"Drawing {idx + 1} - {drawing[3]}", expanded=False):
            st.image(
                f"data:image/png;base64,{drawing[1]}",
                use_container_width=True
            )
            if drawing[2]:  # If Gemini analysis exists
                st.markdown(f"**AI Analysis:**\n{drawing[2]}")
            else:
                st.info("No AI analysis available (API key was not provided)")


def main():
    st.title("Hand Gesture Drawing App")

    # Initialize components
    initialize_state()
    hand_tracker = HandTracker()
    db = DrawingDatabase()

    # Sidebar controls
    st.sidebar.header("Drawing Controls")
    min_distance = st.sidebar.slider(
        "Minimum Pinch Distance",
        0, 100, 20,
        key="min_distance_slider"
    )
    max_distance = st.sidebar.slider(
        "Maximum Pinch Distance",
        min_distance, 200, 100,
        key="max_distance_slider"
    )
    line_thickness = st.sidebar.slider(
        "Line Thickness",
        1, 20, 5,
        key="line_thickness_slider"
    )
    color = st.sidebar.color_picker(
        "Drawing Color",
        "#FF0000",
        key="color_picker"
    )

    # Convert color from hex to RGB
    color_rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # Gemini API Setup
    st.sidebar.header("AI Analysis")
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        key="gemini_api_key",
        help="Enter your Gemini API key to enable AI analysis of your drawings"
    )
    gemini_helper = GeminiHelper(api_key if api_key else None)

    # Canvas operations
    col1, col2, col3 = st.sidebar.columns(3)
    clear_btn = col1.button(
        "Clear Canvas", key=f"clear_canvas_btn", on_click=clear_canvas)
    save_btn = col2.button(
        "Save Drawing", key=f"save_drawing_btn", on_click=save_drawing_callback)
    view_btn = col3.button(
        "View History", key=f"view_history_btn", on_click=view_history_callback)

    # Main content area
    frame_placeholder = st.empty()
    analysis_placeholder = st.empty()

    # Handle different view states
    if st.session_state.view_history_triggered:
        # Display history view
        display_drawing_history(frame_placeholder, db)
    else:
        # Camera and drawing view
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while True:
                if st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access webcam")
                        break

                    frame = cv2.flip(frame, 1)

                    # Initialize canvas if needed
                    if st.session_state.canvas is None:
                        st.session_state.canvas = np.zeros(
                            frame.shape, dtype=np.uint8)

                    results = hand_tracker.process_frame(frame)
                    thumb_pos, index_pos = hand_tracker.get_finger_positions(
                        results, frame.shape)

                    if thumb_pos and index_pos:
                        distance = hand_tracker.calculate_distance(
                            thumb_pos, index_pos)

                        if distance < min_distance:
                            if not st.session_state.drawing:
                                st.session_state.drawing = True
                                st.session_state.prev_point = index_pos
                            else:
                                cv2.line(
                                    st.session_state.canvas,
                                    st.session_state.prev_point,
                                    index_pos,
                                    color_rgb,
                                    line_thickness
                                )
                                st.session_state.prev_point = index_pos

                            cv2.circle(frame, thumb_pos, 15, (0, 255, 0), -1)
                            cv2.circle(frame, index_pos, 15, (0, 255, 0), -1)

                        elif distance > max_distance:
                            st.session_state.drawing = False
                            st.session_state.prev_point = None

                    frame = hand_tracker.draw_landmarks(frame, results)
                    combined_image = cv2.addWeighted(
                        frame, 0.7, st.session_state.canvas, 0.8, 0)
                    frame_placeholder.image(
                        combined_image, channels="BGR", use_container_width=True)

                # Handle save action
                if st.session_state.save_triggered:
                    try:
                        canvas_pil = Image.fromarray(cv2.cvtColor(
                            st.session_state.canvas, cv2.COLOR_BGR2RGB))
                        buffered = BytesIO()
                        canvas_pil.save(buffered, format="PNG")
                        img_str = base64.b64encode(
                            buffered.getvalue()).decode()

                        # Show frozen frame
                        frame_placeholder.image(cv2.cvtColor(
                            st.session_state.canvas, cv2.COLOR_BGR2RGB))

                        # Process with Gemini if available
                        gemini_analysis = None
                        if api_key:
                            with st.spinner('Analyzing drawing with Gemini AI...'):
                                gemini_analysis = gemini_helper.analyze_image(
                                    f"data:image/png;base64,{img_str}")
                                analysis_placeholder.markdown(
                                    f"**AI Analysis:**\n{gemini_analysis}")
                        else:
                            analysis_placeholder.warning(
                                "No Gemini API key provided. Add your API key in the sidebar for AI analysis.")

                        # Save to database
                        db.save_drawing(img_str, gemini_analysis)
                        st.sidebar.success("Drawing saved successfully!")
                        st.session_state.save_triggered = False

                    except Exception as e:
                        st.sidebar.error(f"Error saving drawing: {str(e)}")
                        st.session_state.save_triggered = False
                        st.session_state.camera_active = True

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            cap.release()


if __name__ == "__main__":
    main()
