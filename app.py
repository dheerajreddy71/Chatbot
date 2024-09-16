import streamlit as st
import spacy
from transformers import pipeline
from collections import defaultdict

# Load NLP models
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis")

# Dummy menu for the restaurant
MENU = {
    "pizza": {"ingredients": ["cheese", "tomato", "olive"], "price": 10},
    "burger": {"ingredients": ["lettuce", "tomato", "cheese"], "price": 8},
    "pasta": {"ingredients": ["basil", "parmesan", "tomato"], "price": 12}
}

# Storing user orders
orders = defaultdict(list)

# Function to detect intents from user input
def detect_intent(user_input):
    doc = nlp(user_input)
    
    # Intent Recognition based on keywords (this can be improved with machine learning models)
    if any(token.lemma_ in ['order', 'want', 'get'] for token in doc):
        return "ORDER_INTENT"
    elif any(token.lemma_ in ['menu', 'option', 'show'] for token in doc):
        return "MENU_INTENT"
    elif any(token.lemma_ in ['feedback', 'review'] for token in doc):
        return "FEEDBACK_INTENT"
    elif any(token.lemma_ in ['recommend', 'suggest'] for token in doc):
        return "RECOMMEND_INTENT"
    else:
        return "UNKNOWN"

# Function to process the order
def process_order(user_input):
    doc = nlp(user_input)
    ordered_items = []
    for token in doc:
        if token.lemma_ in MENU.keys():
            ordered_items.append(token.lemma_)
    
    if ordered_items:
        for item in ordered_items:
            orders["items"].append(item)
        return f"Added {', '.join(ordered_items)} to your order."
    else:
        return "Sorry, I couldn't find the items in your order."

# Function to show menu
def show_menu():
    menu_items = ", ".join(MENU.keys())
    return f"Our menu includes: {menu_items}."

# Function to recommend based on ingredients or preferences
def recommend_item(user_input):
    doc = nlp(user_input)
    preferences = [token.text for token in doc if token.text in ['cheese', 'tomato', 'vegetarian', 'spicy']]
    recommended = []
    
    for item, details in MENU.items():
        if any(pref in details['ingredients'] for pref in preferences):
            recommended.append(item)
    
    if recommended:
        return f"We recommend: {', '.join(recommended)} based on your preferences."
    else:
        return "Sorry, we couldn't find any recommendations based on your preferences."

# Function to analyze feedback
def analyze_feedback(feedback):
    result = sentiment_analyzer(feedback)
    return result[0]['label'], result[0]['score']

# Streamlit app interface
st.title("Restaurant Ordering Chatbot")

# Capture user input
user_input = st.text_input("Type your message here:")

# Determine user intent and respond
if user_input:
    intent = detect_intent(user_input)
    
    if intent == "ORDER_INTENT":
        response = process_order(user_input)
    elif intent == "MENU_INTENT":
        response = show_menu()
    elif intent == "RECOMMEND_INTENT":
        response = recommend_item(user_input)
    elif intent == "FEEDBACK_INTENT":
        feedback = st.text_area("Please provide your feedback")
        if st.button("Analyze Feedback"):
            sentiment, score = analyze_feedback(feedback)
            st.write(f"Feedback sentiment: {sentiment} (confidence: {score:.2f})")
            response = None  # Feedback analysis doesn't need a response
        else:
            response = "Please type your feedback."
    else:
        response = "Sorry, I didn't understand your request. Could you please rephrase?"
    
    if response:
        st.write(response)

# Display the user's current order
if orders["items"]:
    st.write("Your current order:", ", ".join(orders["items"]))
    st.write(f"Total price: ${sum(MENU[item]['price'] for item in orders['items'])}")
