import moondream as md
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MOONDREAM_API_KEY")

class CloudMoondream:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CloudMoondream, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized:
            return
        
        if not API_KEY:
            print("AI: WARNING - MOONDREAM_API_KEY not found in .env file.")
            return

        print("AI: Initializing Moondream Cloud API...")
        self.model = md.vl(api_key=API_KEY)
        self.initialized = True
        print("AI: Moondream Cloud API initialized successfully.")

    def describe(self, image_bytes, prompt):
        if not self.initialized:
            self.initialize()
            
        try:
            image = Image.open(io.BytesIO(image_bytes))
            answer = self.model.query(image, prompt)
            return answer["answer"]
        except Exception as e:
            return f"Error in Moondream Cloud processing: {str(e)}"

vlm_assistant = CloudMoondream()

def describe_object(image_bytes, prompt):
    return vlm_assistant.describe(image_bytes, prompt)
