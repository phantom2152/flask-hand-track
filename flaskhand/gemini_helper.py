import google.generativeai as genai
from PIL import Image
import io
import base64


class GeminiHelper:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None
        if api_key:
            self.setup_model()

    def setup_model(self):
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        except Exception as e:
            print(f"Error setting up Gemini model: {e}")
            self.model = None

    def analyze_image(self, image_data):
        if not self.model:
            return "Gemini API key not configured"

        try:
            # Handle both base64 string and data URL
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            # Convert base64 to PIL Image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            prompt = """
            Analyze this hand-drawn image and provide:
            1. A description of what's drawn
            2. Identify any geometric shapes, patterns, or symbols
            3. If there are any mathematical expressions, solve them
            4. Provide any insights about the drawing style or technique
            Keep the analysis concise but informative.
            """

            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return f"Error analyzing image: {str(e)}"
