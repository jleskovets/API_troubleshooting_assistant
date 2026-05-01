import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.vector_store import build_vector_store, semantic_search


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Check your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(
    page_title="API Troubleshooting Assistant",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ API Troubleshooting Assistant")
st.write("Paste a customer API issue and get the most relevant troubleshooting case plus a draft reply.")


@st.cache_data
def load_cases():
    return pd.read_csv(
        "data/troubleshooting_cases_english.csv",
        sep=";"
    )


@st.cache_resource
def initialize_semantic_index():
    count = build_vector_store()
    return count


with st.spinner("Preparing semantic search index..."):
    indexed_count = initialize_semantic_index()


cases = load_cases()

if "history" not in st.session_state:
    st.session_state.history = []


with st.sidebar:
    st.header("Knowledge base")
    st.success(f"Semantic index ready. Cases indexed: {indexed_count}")

    st.header("Request history")

    if not st.session_state.history:
        st.caption("No requests yet.")
    else:
        for item in reversed(st.session_state.history):
            with st.expander(item["title"]):
                st.write("**Customer message:**")
                st.write(item["customer_message"])

                st.write("**Matched case:**")
                st.write(item["matched_case"])

                st.write("**Email draft:**")
                st.write(item["email_draft"])

    if st.session_state.history:
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()


customer_message = st.text_area(
    "Customer message",
    height=200,
    placeholder="Example: We receive 401 Invalid credentials when calling POST /auth/token..."
)


def generate_customer_reply(message, match):
    case = match["case"]
    score = match["score"]

    cases_text = f"""
Case ID: {case['id']}
API Area: {case['api_area']}
Endpoint: {case['endpoint']}
Error Code: {case['error_code']}
Problem: {case['problem']}
Possible Root Cause: {case['root_cause']}
Suggested Solution: {case['solution']}
Semantic Match Score: {score}
"""

    prompt = f"""
You are an API support assistant helping a system analyst respond to a customer.

Customer message:
{message}

Most relevant troubleshooting case:
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


if st.button("Analyze issue"):

    if not customer_message.strip():
        st.warning("Please paste a customer message first.")

    else:
        semantic_matches = semantic_search(customer_message, top_k=1)

        st.subheader("Most relevant troubleshooting case")

        if not semantic_matches:
            st.info("No similar cases found.")

        else:
            match = semantic_matches[0]
            case = match["case"]
            score = match["score"]

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
                    f"Semantic match score: {score}"
                )

            st.subheader("AI-generated response")

            with st.spinner("Generating customer reply..."):
                reply = generate_customer_reply(
                    customer_message,
                    match
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

                st.session_state.history.append({
                    "title": f"{datetime.now().strftime('%H:%M')} — Case #{case['id']} / {case['api_area']}",
                    "customer_message": customer_message,
                    "matched_case": f"Case #{case['id']} — {case['problem']}",
                    "email_draft": reply_data["email_draft"]
                })

            except json.JSONDecodeError:
                st.warning(
                    "AI response could not be parsed as structured JSON."
                )

                st.text_area(
                    "Raw AI response",
                    value=reply,
                    height=350
                )