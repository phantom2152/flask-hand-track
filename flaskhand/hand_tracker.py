import cv2
import mediapipe as mp
import numpy as np
import base64


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results, frame

    def get_finger_positions(self, results, frame_shape):
        if not results.multi_hand_landmarks:
            return None, None

        thumb_tip = results.multi_hand_landmarks[0].landmark[4]
        index_tip = results.multi_hand_landmarks[0].landmark[8]

        thumb_pos = (
            int(thumb_tip.x * frame_shape[1]),
            int(thumb_tip.y * frame_shape[0])
        )
        index_pos = (
            int(index_tip.x * frame_shape[1]),
            int(index_tip.y * frame_shape[0])
        )

        return thumb_pos, index_pos

    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    def frame_to_base64(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def process_and_encode_frame(self, frame):
        results, processed_frame = self.process_frame(frame)
        frame_shape = frame.shape

        # Get finger positions
        thumb_pos, index_pos = self.get_finger_positions(results, frame_shape)

        # Draw landmarks
        processed_frame = self.draw_landmarks(processed_frame, results)

        # Convert to base64
        encoded_frame = self.frame_to_base64(processed_frame)

        return {
            'frame': encoded_frame,
            'thumb_pos': thumb_pos,
            'index_pos': index_pos,
            'has_hand': thumb_pos is not None and index_pos is not None
        }
