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
            # Convert base64 to PIL Image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))

            prompt = """
            Analyze this hand-drawn image and:
            1. Identify any geometric shapes present
            2. Detect any mathematical expressions or equations
            3. Describe any patterns or symbols
            4. Provide a brief interpretation of what's drawn
            """

            print(f"Analyzing image with prompt: {prompt}")
            response = self.model.generate_content([prompt, image])
            print(f"Received response: {response.text}")
            return response.text
        except Exception as e:
            print(f"Received Error: {e}")
            return f"Error analyzing image: {str(e)}"
