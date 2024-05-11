import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

data = {
    "Email": [
        "Hey there, how are you doing?",
        "Congratulations! You've won a prize! Click here to claim it now!",
        "Reminder: Meeting at 2 PM today",
        "Exclusive offer for you: 50% off on all products! Limited time only!",
        "Hi, just checking in to see if you received my previous email.",
        "URGENT: Your account has been compromised. Click here to reset your password now!",
        "Don't miss out on our sale! Visit our website for exclusive deals.",
        "Hello, I'm writing to follow up on our previous conversation. Let me know if you need any further assistance.",
        "Your subscription has expired. Renew now and get 20% off!",
        "Hi there, hope you're having a great day! Just wanted to share some exciting news with you.",
        "Last chance to save 30% on your purchase! Click here to shop now!",
        "We miss you! Come back and shop with us to receive a special discount.",
        "Hey, are you free for a quick chat later today? Let me know your availability.",
        "URGENT: Action required! Your account has been suspended due to suspicious activity.",
        "Check out our latest collection of products. You won't be disappointed!",
        "Reminder: Tomorrow's meeting has been rescheduled to 3 PM.",
        "Your package has been delivered. Click here to track your order.",
        "Hi, can you please review the attached document and provide your feedback?",
        "Claim your free gift now! Limited quantities available.",
        "Thank you for your purchase! Here's a discount code for your next order.",
        "Final call! Don't miss your chance to save big on our Black Friday sale!",
        "Your work has been submitted.","Your resume has been shortlisted"
    ],
    "Spam": [
        "0", "1", "0", "1", "0", "1", "1", "0", "1", "0",
        "1", "1", "0", "1", "1", "0", "1", "0", "1", "0", "1","0","0"
    ]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)


# Split the data into features and target
X = df["Email"]
y = df["Spam"].astype(int)

# Build a simple Naive Bayes classifier pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X, y)

# Function to predict if the input text is spam or not
def predict_spam(input_text):
    prediction = model.predict([input_text])
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Email Spam Detection")
st.markdown(
    """
    <style>
    .title {
        text-align: center;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        padding: 10px 0;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Text input for the user to enter the email
input_text = st.text_area("Paste your email here:")

# Button to trigger prediction
if st.button("Predict"):
    if input_text:
        result = predict_spam(input_text)
        st.write(f"This email is: {result}")
    else:
        st.write("Please enter an email to classify.")

# Footer text
st.markdown("<div class='footer'>Created by Vishakha & Vidushi</div>", unsafe_allow_html=True)
