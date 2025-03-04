import streamlit as st
import requests

st.title("Legal Chatbot")
st.write("Enter your legal question below:")

user_query = st.text_input("Question:")
if st.button("Ask"):
    response = requests.post("http://127.0.0.1:8000/ask", json={"query": user_query})
    summary = response.json().get("summary")
    st.write("### Response:")
    st.write(summary)
