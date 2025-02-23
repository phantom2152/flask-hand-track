import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def process_frame(self, frame):
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame and detect hands
        results = self.hands.process(frame_rgb)
        return results

    def get_finger_positions(self, results, frame_shape):
        if not results.multi_hand_landmarks:
            return None, None

        thumb_tip = results.multi_hand_landmarks[0].landmark[4]
        index_tip = results.multi_hand_landmarks[0].landmark[8]

        # Convert normalized coordinates to pixel coordinates
        thumb_pos = (
            int(thumb_tip.x * frame_shape[1]),
            int(thumb_tip.y * frame_shape[0])
        )
        index_pos = (
            int(index_tip.x * frame_shape[1]),
            int(index_tip.y * frame_shape[0])
        )

        return thumb_pos, index_pos

    def process_and_encode_frame(self, frame):
        """Process a frame and return hand tracking data"""
        try:
            # Process the frame
            results = self.process_frame(frame)
            frame_shape = frame.shape

            # Get finger positions
            thumb_pos, index_pos = self.get_finger_positions(
                results, frame_shape)

            # Return processed data
            return {
                'has_hand': thumb_pos is not None and index_pos is not None,
                'thumb_pos': thumb_pos,
                'index_pos': index_pos
            }
        except Exception as e:
            print(f"Error processing frame in hand tracker: {e}")
            return {
                'has_hand': False,
                'thumb_pos': None,
                'index_pos': None
            }
