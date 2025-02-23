import sqlite3
from datetime import datetime
import json


class DrawingDatabase:
    def __init__(self):
        self.conn = sqlite3.connect('drawings.db', check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS drawings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_data TEXT,
            gemini_analysis TEXT,
            timestamp DATETIME
        )
        ''')
        self.conn.commit()

    def save_drawing(self, image_data, gemini_analysis=None):
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO drawings (image_data, gemini_analysis, timestamp) VALUES (?, ?, ?)',
            (image_data, json.dumps(gemini_analysis)
             if gemini_analysis else None, datetime.now())
        )
        self.conn.commit()

    def get_all_drawings(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM drawings ORDER BY timestamp DESC')
        drawings = cursor.fetchall()

        # Parse JSON gemini_analysis back to string
        parsed_drawings = []
        for drawing in drawings:
            gemini_analysis = json.loads(drawing[2]) if drawing[2] else None
            parsed_drawings.append(
                (drawing[0], drawing[1], gemini_analysis, drawing[3]))
        return parsed_drawings

    def close(self):
        self.conn.close()
