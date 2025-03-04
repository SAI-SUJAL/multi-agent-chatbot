import streamlit as st
import requests
API_URL = "https://legal-chatbot-india.streamlit.app"
st.title("India Legal Chatbot")
st.write("Enter your legal question below:")

user_query = st.text_input("Question:")
if st.button("Ask"):
    response = requests.post(f"{API_URL}/ask", json={"query": user_query})
    summary = response.json().get("summary")
    st.write("### Response:")
    st.write(summary)
