import gradio as gr
import random
import time
import mlflow
import os
from pathlib import Path

# Set up MLflow
tracking_uri = "http://127.0.0.1:5000"  # Default MLflow server URI
mlflow.set_tracking_uri(tracking_uri)

# Create a new experiment for this project
experiment_name = "gradio-chatbot"
try:
    mlflow.create_experiment(experiment_name)
except:
    pass  # Experiment already exists
mlflow.set_experiment(experiment_name)

# Simple chatbot response function
@mlflow.trace
async def respond(message, chat_history):
    # Simple echo response with some variation
    responses = [
        f"You said: {message}",
        f"I heard: {message}",
        f"Interesting! You mentioned: {message}",
        f"Thanks for sharing: {message}",
        f"I'm a simple chatbot. You told me: {message}"
    ]
    
    # Simulate processing time
    time.sleep(1)
    
    # Add the user message and bot response to the chat history
    bot_message = random.choice(responses)
    chat_history.append((message, bot_message))
    
    return "", chat_history

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Simple Chatbot")
    
    # Chatbot interface
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Message")
    clear = gr.Button("Clear")
    
    # Set up the message handler with async function
    async def wrapped_respond(message, chat_history):
        return await respond(message, chat_history)
    
    msg.submit(wrapped_respond, [msg, chatbot], [msg, chatbot])
    
    # Clear button functionality
    def clear_chat():
        return None, []
    
    clear.click(clear_chat, None, [msg, chatbot], queue=False)

# Run the app
if __name__ == "__main__":
    print(f"MLflow tracking UI: {tracking_uri}")
    print("Starting Gradio app...")
    demo.launch(share=True)
