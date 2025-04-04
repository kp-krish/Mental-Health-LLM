# Basic usage with interactive chat
from medical_chatbot import MedicalChatbot

chatbot = MedicalChatbot()
# chatbot.chat()

# Or use it programmatically
response = chatbot.generate_response("I have Headache because of stress. Give me a short response.")
print(response)