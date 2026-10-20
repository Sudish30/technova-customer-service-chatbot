# Quick one-message check that the Gemini key + model work
from config import get_model
m = get_model()
print("MODEL REPLY:", m.invoke("Say OK").content)
