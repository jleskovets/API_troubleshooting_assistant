import os
import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Setup
# -----------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Check your .env file.")

client = OpenAI(api_key=api_key)

st.set_page_config(
    page_title="API Troubleshooting Assistant",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ API Troubleshooting Assistant")
st.write("Paste a customer API issue and get similar cases plus a draft reply.")


# -----------------------------
# Load data
# -----------------------------

@st.cache_data
def load_cases():
    return pd.read_csv(
        "data/troubleshooting_cases_english.csv",
        sep=";"
    )


cases = load_cases()


# -----------------------------
# UI input
# -----------------------------

customer_message = st.text_area(
    "Customer message",
    height=200,
    placeholder="Example: We receive 401 Invalid credentials when calling POST /auth/token..."
)


# -----------------------------
# Search logic
# -----------------------------

def simple_search(message, df):

    message = message.lower()

    results = []

    for _, row in df.iterrows():

        score = 0

        # ---- High-weight fields ----

        endpoint = str(row["endpoint"]).lower()
        error_code = str(row["error_code"]).lower()

        if endpoint in message:
            score += 5

        if error_code in message:
            score += 5

        # ---- Medium-weight fields ----

        problem_text = str(row["problem"]).lower()

        for word in message.split():

            if word in problem_text:
                score += 2

        # ---- Low-weight fields ----

        full_text = " ".join([
            str(row["api_area"]),
            str(row["root_cause"]),
            str(row["solution"]),
        ]).lower()

        for word in message.split():

            if word in full_text:
                score += 1

        if score > 0:
            results.append((score, row))

    # сортируем по релевантности

    results = sorted(
        results,
        key=lambda x: x[0],
        reverse=True
    )

    # возвращаем только 1 лучший кейс

    return results[:1]


# -----------------------------
# AI logic
# -----------------------------

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

Return ONLY raw valid JSON. Do not use markdown code fences.
Use this exact structure:
{{
  "issue_summary": "...",
  "root_cause": "...",
  "next_steps": ["...", "...", "..."],
  "email_draft": "..."
}}

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


def parse_ai_json(reply):
    reply = reply.strip()

    if reply.startswith("```json"):
        reply = reply.replace("```json", "", 1).strip()

    if reply.startswith("```"):
        reply = reply.replace("```", "", 1).strip()

    if reply.endswith("```"):
        reply = reply[:-3].strip()

    return json.loads(reply)


# -----------------------------
# Main button logic
# -----------------------------

if st.button("Analyze issue"):

    if not customer_message.strip():
        st.warning("Please paste a customer message first.")

    else:
        matches = simple_search(customer_message, cases)

        st.subheader("Similar troubleshooting cases")

        if not matches:
            st.info("No similar cases found.")

        else:
            for score, case = matches[0]

            with st.container(border=True):

            st.markdown(
            f"### Case #{case['id']} — {case['api_area']}"
        )

        st.write(
            f"**Endpoint:** {case['endpoint']}"
        )

        st.write(
            f"**Error code:** {case['error_code']}"
        )

        st.write(
            f"**Problem:** {case['problem']}"
        )

        st.write(
            f"**Possible root cause:** {case['root_cause']}"
        )

        st.write(
            f"**Suggested solution:** {case['solution']}"
        )

        st.caption(
            f"Match score: {score}"
        )

            st.subheader("AI-generated response")

            with st.spinner("Generating customer reply..."):
                reply = generate_customer_reply(
                    customer_message,
                    matches
                )

            try:
                reply_data = parse_ai_json(reply)

                col1, col2 = st.columns(2)

                with col1:
                    with st.container(border=True):
                        st.markdown("### Issue summary")
                        st.write(reply_data["issue_summary"])

                with col2:
                    with st.container(border=True):
                        st.markdown("### Likely root cause")
                        st.write(reply_data["root_cause"])

                with st.container(border=True):
                    st.markdown("### Recommended next steps")

                    for step in reply_data["next_steps"]:
                        st.write(f"- {step}")

                with st.container(border=True):
                    st.markdown("### Email draft")

                    st.text_area(
                        "Generated email",
                        value=reply_data["email_draft"],
                        height=250,
                        label_visibility="collapsed"
                    )

            except json.JSONDecodeError:
                st.warning(
                    "AI response could not be parsed as structured JSON."
                )

                st.text_area(
                    "Raw AI response",
                    value=reply,
                    height=350
                )