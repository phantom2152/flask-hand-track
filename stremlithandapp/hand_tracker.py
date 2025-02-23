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
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results

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
