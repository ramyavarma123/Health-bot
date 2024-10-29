import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from typing import List, Dict, Optional, Tuple
import gradio as gr
from g4f.client import Client
import json
import datetime
from dataclasses import dataclass
from pathlib import Path
import logging
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CUSTOM_CSS = """
.gradio-container {
    background-color: #f0f4f8;
    font-family: 'Arial', sans-serif;
}

.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 8px;
    max-width: 80%;
}

.user-message {
    background-color: #e3f2fd;
    margin-left: auto;
}

.bot-message {
    background-color: #f5f5f5;
    margin-right: auto;
}

.message-timestamp {
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.25rem;
}

button.primary-btn {
    background-color: #2196f3;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s;
}

button.primary-btn:hover {
    background-color: #1976d2;
}
"""

@dataclass
class ChatMessage:
    """Data class to store chat messages"""
    role: str
    content: str
    timestamp: str

class MedicalChatBot:
    """Advanced Medical Chatbot with conversation management"""
    
    MAX_HISTORY: int = 30
    
    def __init__(self) -> None:
        self.client = Client()
        self.conversation_history: deque = deque(maxlen=self.MAX_HISTORY)
        self.decision_tree = None
        self.symptom_columns = []
        self.load_data_and_train_model()
        self.initialize_system()

    def load_data_and_train_model(self):
        """Load datasets and train the Decision Tree model"""
        # Load datasets
        train_data = pd.read_csv("Training.csv")
        test_data = pd.read_csv("Testing.csv")

        # Separate features and target from training data
        X_train = train_data.drop(columns=['prognosis'])
        y_train = train_data['prognosis']

        # Train the Decision Tree model
        self.decision_tree = DecisionTreeClassifier(random_state=0)
        self.decision_tree.fit(X_train, y_train)

        # Store the symptom columns for predictions
        self.symptom_columns = X_train.columns.tolist()

    def initialize_system(self) -> None:
        """Initialize system with medical context"""
        self.system_prompt = """
        You are an AI Medical Assistant. Please:
        1. Ask relevant diagnostic questions
        2. Provide medical advice based on symptoms
        3. Recommend when to seek immediate medical attention
        4. Always maintain professional medical ethics
        # 5. Clearly state you are an AI and not a replacement for real doctors
        """

    def generate_response(self, user_input: str) -> str:
        """Generate response using chain of thought reasoning"""
        try:
            # Check for symptoms in user input
            symptoms = self.extract_symptoms(user_input)
            prognosis_response = ""
            if symptoms:
                prognosis_response = self.predict_prognosis(symptoms)
            
            # Create chat messages for response
            messages = [
                {"role": "system", "content": self.system_prompt},
                *[{"role": m.role, "content": m.content} for m in self.conversation_history],
                {"role": "user", "content": user_input}
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )

            # Combine AI response with prognosis if available
            full_response = response.choices[0].message.content
            if prognosis_response:
                full_response += f"\n\nBased on your symptoms, the likely condition is: {prognosis_response}."
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

    def _update_history(self, user_input: str, bot_response: str) -> None:
        """Update conversation history with new messages"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.conversation_history.append(
            ChatMessage(role="user", content=user_input, timestamp=timestamp)
        )
        self.conversation_history.append(
            ChatMessage(role="assistant", content=bot_response, timestamp=timestamp)
        )

    def predict_prognosis(self, symptoms: List[str]) -> str:
        """Predict prognosis based on symptoms"""
        input_vector = np.zeros(len(self.symptom_columns))
        for symptom in symptoms:
            if symptom in self.symptom_columns:
                input_vector[self.symptom_columns.index(symptom)] = 1

        prediction = self.decision_tree.predict([input_vector])[0]
        return prediction

    def extract_symptoms(self, user_input: str) -> List[str]:
        """Extract symptoms mentioned in user input"""
        return [symptom for symptom in self.symptom_columns if symptom.lower() in user_input.lower()]

def create_demo() -> gr.Blocks:
    """Create Gradio interface with professional styling"""
    chatbot = MedicalChatBot(type="messages")

    with gr.Blocks(css=CUSTOM_CSS) as demo:
        # chatbot = gr.Chatbot(type="messages")
        gr.Markdown("# 🏥 HealthMate AI")
        gr.Markdown("""Welcome! I'm an AI medical assistant designed to provide general medical information. Please note that I'm not a replacement for professional medical care.""")

        chatbot_component = gr.Chatbot(label="Conversation History", height=400, show_label=True)
        
        msg = gr.Textbox(label="Your Message", placeholder="Type your medical concern here...", lines=2)

        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear Chat", variant="secondary")

        def user_input(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
            bot_response = chatbot.generate_response(message)
            chatbot._update_history(message, bot_response)
            history.append((message, bot_response))
            return "", history

        def clear_history() -> List[Tuple[str, str]]:
            chatbot.conversation_history.clear()
            return []

        submit.click(user_input, inputs=[msg, chatbot_component], outputs=[msg, chatbot_component])
        clear.click(clear_history, outputs=[chatbot_component])
        msg.submit(user_input, inputs=[msg, chatbot_component], outputs=[msg, chatbot_component])

    return demo

if __name__ == "__main__":
    try:
        demo = create_demo()
        demo.launch(server_port=7861, share=True)
    except Exception as e:
        logger.error(f"Failed to launch interface: {str(e)}")
