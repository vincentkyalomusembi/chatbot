# import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk
from tkinter import scrolledtext

# # Sample training data (user inputs & responses)
# questions = [
#     "hello", "hi", "how are you", "what's your name", "bye", "goodbye",
#     "hey", "see you later", "what's up", "how's it going", "nice to meet you",
#     "how do you do", "what's new", "what's happening", "what's going on"
# ]
# responses = [
#     "Hello!", "Hi there!", "I'm fine, thank you!", "My name is ChatBot.",
#     "Goodbye!", "See you later!", "Hey!", "See you later!", "Not much.",
#     "I'm doing well, thanks for asking!", "Nice to meet you!", "I'm doing great!",
#     "Nothing new.", "Not much, how about you?", "I'm your AI assistant"
# ]

# Load the dataset
df = pd.read_csv('Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special characters
    return text

# Apply text cleaning
df["question"] = df["instruction"].apply(clean_text)
df["response"] = df["response"].apply(clean_text)

# Vectorize text (convert text into numbers)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])  # Transform questions into numerical format
y = list(range(len(df)))  # Assign response indices

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X, y)

def extract_order_number(user_input):
    match = re.search(r"\b\d+\b", user_input)
    if match:
        return match.group()
    return ""
# Function to get chatbot response
def chatbot_response(user_input):
    order_number = extract_order_number(user_input)  # Clean user input
    user_input_vec = vectorizer.transform([user_input])  # Convert to numerical
    try:
        prediction = model.predict(user_input_vec)[0]  # Get predicted response index
        response = df["response"].iloc[prediction]

        #Check if response contains {{Order Number}}
        if "{{Order Number}}" in response and order_number:
            response = response.replace(r"{{Order Number}}", order_number)

        return response
    except:
        return "I'm sorry, I don't understand that."
# GUI Setup
def send_message():
    user_text = entry.get().strip()  # Get input text
    if not user_text:
        return  # Ignore empty input

    chat_area.insert(tk.END, "You: " + user_text + "\n")

    response = chatbot_response(user_text)
    chat_area.insert(tk.END, "Chatbot: " + response + "\n\n")

    entry.delete(0, tk.END)  # Clear input field

# Initialize Tkinter
root = tk.Tk()
root.title("AI Customer support Chatbot")
root.geometry("600x500")
root.configure(bg="#282c34")#dark back ground

# Chat display area
chat_area = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD)
chat_area.pack(pady=10, padx=10)
chat_area.configure(bg="#1e1e1e", fg="white")#dark theme with white

# Input field
entry = tk.Entry(root, width=60, font=("Arial", 12), bg="#ffffff")
entry.pack(pady=5, padx=5)

# Send button with styling
send_button = tk.Button(root, text="Send", command=send_message, font=("Arial", 12, "bold"), bg="#61afef", fg="white")
send_button.pack(pady=5)

# Run the GUI
root.mainloop()