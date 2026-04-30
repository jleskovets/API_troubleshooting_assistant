import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="API Troubleshooting Assistant",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ API Troubleshooting Assistant")
st.write("Paste a customer API issue and get similar cases plus a draft reply.")

@st.cache_data
def load_cases():
    return pd.read_csv("data/troubleshooting_cases_english.csv", sep=";")

cases = load_cases()

customer_message = st.text_area(
    "Customer message",
    height=200,
    placeholder="Example: We receive 401 Invalid credentials when calling POST /auth/token..."
)

def simple_search(message, df):
    message = message.lower()
    results = []

    for _, row in df.iterrows():
        score = 0
        searchable_text = " ".join([
            str(row["api_area"]),
            str(row["endpoint"]),
            str(row["error_code"]),
            str(row["problem"]),
            str(row["root_cause"]),
            str(row["solution"]),
        ]).lower()

        for word in message.split():
            if word in searchable_text:
                score += 1

        if score > 0:
            results.append((score, row))

    return sorted(results, key=lambda x: x[0], reverse=True)[:3]

def generate_customer_reply(message, matches):
    cases_text = ""

    for score, case in matches:
        cases_text += f"""
Case ID: {case['id']}
API Area: {case['api_area']}
Endpoint: {case['endpoint']}
Error Code: {case['error_code']}
Problem: {case['problem']}
Possible Root Cause: {case['root_cause']}
Suggested Solution: {case['solution']}
Match Score: {score}
---
"""

    prompt = f"""
You are an API support assistant helping a system analyst respond to a customer.

Customer message:
{message}

Similar troubleshooting cases:
{cases_text}

Generate:
1. Short issue summary
2. Likely root cause
3. Recommended next steps
4. A professional customer email draft

Rules:
- Write in English
- Be polite and clear
- Do not claim that the issue is definitely solved
- If information is missing, ask the customer for specific details
- Keep the email concise
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

if st.button("Analyze issue"):
    if not customer_message.strip():
        st.warning("Please paste a customer message first.")
    else:
        matches = simple_search(customer_message, cases)

        st.subheader("Similar troubleshooting cases")

        if not matches:
            st.info("No similar cases found.")
        else:
            for score, case in matches:
                with st.container(border=True):
                    st.markdown(f"### Case #{case['id']} — {case['api_area']}")
                    st.write(f"**Endpoint:** {case['endpoint']}")
                    st.write(f"**Error code:** {case['error_code']}")
                    st.write(f"**Problem:** {case['problem']}")
                    st.write(f"**Possible root cause:** {case['root_cause']}")
                    st.write(f"**Suggested solution:** {case['solution']}")
                    st.caption(f"Match score: {score}")

            st.subheader("AI-generated response")

            with st.spinner("Generating customer reply..."):
                reply = generate_customer_reply(customer_message, matches)

            st.text_area(
                "Draft reply",
                value=reply,
                height=350
            )