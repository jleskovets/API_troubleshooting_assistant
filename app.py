import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.vector_store import build_vector_store, semantic_search


# -----------------------------
# Setup
# -----------------------------

load_dotenv()

CSV_PATH = "data/troubleshooting_cases_english.csv"

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


# -----------------------------
# Data helpers
# -----------------------------

@st.cache_data
def load_cases():
    df = pd.read_csv(CSV_PATH, sep=";")

    for column in ["tags", "logs"]:
        if column not in df.columns:
            df[column] = ""

    return df


def save_cases(df):
    df.to_csv(CSV_PATH, sep=";", index=False)
    load_cases.clear()
    initialize_semantic_index.clear()


@st.cache_resource
def initialize_semantic_index():
    return build_vector_store(CSV_PATH)


def rebuild_index():
    load_cases.clear()
    initialize_semantic_index.clear()
    return build_vector_store(CSV_PATH)


with st.spinner("Preparing semantic search index..."):
    indexed_count = initialize_semantic_index()


if "history" not in st.session_state:
    st.session_state.history = []

if "page" not in st.session_state:
    st.session_state.page = "analyze"


# -----------------------------
# Sidebar navigation as buttons
# -----------------------------

with st.sidebar:
    st.header("Navigation")

    if st.button("🔍 Analyze issue", use_container_width=True):
        st.session_state.page = "analyze"
        st.rerun()

    if st.button("📚 Manage knowledge base", use_container_width=True):
        st.session_state.page = "kb"
        st.rerun()

    st.divider()
    st.success(f"Semantic index ready. Cases indexed: {indexed_count}")


# -----------------------------
# AI helpers
# -----------------------------

def parse_ai_json(reply):
    reply = reply.strip()

    if reply.startswith("```json"):
        reply = reply.replace("```json", "", 1).strip()

    if reply.startswith("```"):
        reply = reply.replace("```", "", 1).strip()

    if reply.endswith("```"):
        reply = reply[:-3].strip()

    return json.loads(reply)


def generate_customer_reply(message, match):
    case = match["case"]
    score = match["score"]

    logs_text = case.get("logs", "")
    tags_text = case.get("tags", "")

    cases_text = f"""
Case ID: {case['id']}
API Area: {case['api_area']}
Endpoint: {case['endpoint']}
Error Code: {case['error_code']}
Problem: {case['problem']}
Possible Root Cause: {case['root_cause']}
Suggested Solution: {case['solution']}
Tags: {tags_text}
Logs: {logs_text}
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


def suggest_tags(api_area, endpoint, error_code, problem, root_cause, solution):
    prompt = f"""
Generate 3 to 6 short tags for this API troubleshooting case.

Case:
API Area: {api_area}
Endpoint: {endpoint}
Error Code: {error_code}
Problem: {problem}
Root Cause: {root_cause}
Solution: {solution}

Return ONLY raw valid JSON:
{{
  "tags": ["tag1", "tag2", "tag3"]
}}

Rules:
- Tags must be lowercase
- Use short technical tags
- No markdown
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    data = parse_ai_json(response.output_text)
    return ", ".join(data["tags"])


def confidence_label(score):
    if score >= 0.75:
        return "High confidence"
    if score >= 0.45:
        return "Medium confidence"
    return "Low confidence"


# =============================
# Page 1 — Analyze issue
# =============================

if st.session_state.page == "analyze":

    st.write(
        "Paste a customer API issue and get the most relevant troubleshooting case plus a draft reply."
    )

    with st.sidebar:
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
            if st.button("Clear history", use_container_width=True):
                st.session_state.history = []
                st.rerun()

    customer_message = st.text_area(
        "Customer message",
        height=200,
        placeholder="Example: Authentication fails when requesting access token..."
    )

    if st.button("Analyze issue", type="primary"):

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

                    st.write(f"**Endpoint:** {case['endpoint']}")
                    st.write(f"**Error code:** {case['error_code']}")
                    st.write(f"**Problem:** {case['problem']}")
                    st.write(f"**Possible root cause:** {case['root_cause']}")
                    st.write(f"**Suggested solution:** {case['solution']}")

                    if case.get("tags"):
                        st.write(f"**Tags:** {case['tags']}")

                    if case.get("logs"):
                        with st.expander("Attached logs"):
                            st.code(case["logs"])

                    st.markdown("### Confidence")
                    st.progress(score)
                    st.caption(
                        f"{confidence_label(score)} — semantic match score: {score}"
                    )

                st.subheader("AI-generated response")

                with st.spinner("Generating customer reply..."):
                    reply = generate_customer_reply(customer_message, match)

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
                        "title": f"{datetime.now().strftime('%H:%M')} — Case #{case['id']}",
                        "customer_message": customer_message,
                        "matched_case": f"Case #{case['id']} — {case['problem']}",
                        "email_draft": reply_data["email_draft"]
                    })

                except json.JSONDecodeError:
                    st.warning("AI response could not be parsed as structured JSON.")
                    st.text_area("Raw AI response", value=reply, height=350)


# =============================
# Page 2 — Manage knowledge base
# =============================

if st.session_state.page == "kb":

    st.header("📚 Manage knowledge base")

    cases = load_cases()

    # -----------------------------
    # Add case
    # -----------------------------

    st.subheader("Add new troubleshooting case")

    with st.form("add_case_form"):

        api_area = st.text_input("API area")
        endpoint = st.text_input("Endpoint")
        error_code = st.text_input("Error code")
        problem = st.text_area("Problem description")
        root_cause = st.text_area("Root cause")
        solution = st.text_area("Solution")
        logs = st.text_area("Attach logs / error payload / request example")
        tags = st.text_input("Tags", placeholder="auth, token, invalid-credentials")

        col1, col2 = st.columns(2)

        with col1:
            suggest_tags_clicked = st.form_submit_button("🧠 Auto-suggest tags")

        with col2:
            add_case_clicked = st.form_submit_button("Add case")

        if suggest_tags_clicked:
            if not problem.strip():
                st.warning("Please fill at least Problem description first.")
            else:
                with st.spinner("Suggesting tags..."):
                    suggested = suggest_tags(
                        api_area,
                        endpoint,
                        error_code,
                        problem,
                        root_cause,
                        solution
                    )
                st.info(f"Suggested tags: {suggested}")

        if add_case_clicked:
            if not problem.strip() or not solution.strip():
                st.warning("Problem and Solution are required.")
            else:
                df = load_cases()

                new_id = int(df["id"].max()) + 1 if not df.empty else 1

                new_row = pd.DataFrame([{
                    "id": new_id,
                    "api_area": api_area,
                    "endpoint": endpoint,
                    "error_code": error_code,
                    "problem": problem,
                    "root_cause": root_cause,
                    "solution": solution,
                    "tags": tags,
                    "logs": logs
                }])

                df = pd.concat([df, new_row], ignore_index=True)

                save_cases(df)

                with st.spinner("Rebuilding semantic index..."):
                    rebuild_index()

                st.success("New case added and semantic index updated.")
                st.rerun()

    st.divider()

    # -----------------------------
    # Edit case
    # -----------------------------

    st.subheader("✏️ Edit case")

    cases = load_cases()

    if cases.empty:
        st.info("Knowledge base is empty.")
    else:
        case_options = {
            f"Case #{row['id']} — {row['api_area']} — {row['problem']}": row["id"]
            for _, row in cases.iterrows()
        }

        selected_label = st.selectbox(
            "Select case to edit",
            list(case_options.keys())
        )

        selected_id = case_options[selected_label]
        selected_case = cases[cases["id"] == selected_id].iloc[0]

        with st.form("edit_case_form"):

            edit_api_area = st.text_input(
                "API area",
                value=str(selected_case["api_area"])
            )

            edit_endpoint = st.text_input(
                "Endpoint",
                value=str(selected_case["endpoint"])
            )

            edit_error_code = st.text_input(
                "Error code",
                value=str(selected_case["error_code"])
            )

            edit_problem = st.text_area(
                "Problem description",
                value=str(selected_case["problem"])
            )

            edit_root_cause = st.text_area(
                "Root cause",
                value=str(selected_case["root_cause"])
            )

            edit_solution = st.text_area(
                "Solution",
                value=str(selected_case["solution"])
            )

            edit_tags = st.text_input(
                "Tags",
                value=str(selected_case["tags"]) if pd.notna(selected_case["tags"]) else ""
            )

            edit_logs = st.text_area(
                "Attached logs",
                value=str(selected_case["logs"]) if pd.notna(selected_case["logs"]) else ""
            )

            save_edit_clicked = st.form_submit_button("Save changes")

            if save_edit_clicked:

                cases.loc[cases["id"] == selected_id, "api_area"] = edit_api_area
                cases.loc[cases["id"] == selected_id, "endpoint"] = edit_endpoint
                cases.loc[cases["id"] == selected_id, "error_code"] = edit_error_code
                cases.loc[cases["id"] == selected_id, "problem"] = edit_problem
                cases.loc[cases["id"] == selected_id, "root_cause"] = edit_root_cause
                cases.loc[cases["id"] == selected_id, "solution"] = edit_solution
                cases.loc[cases["id"] == selected_id, "tags"] = edit_tags
                cases.loc[cases["id"] == selected_id, "logs"] = edit_logs

                save_cases(cases)

                with st.spinner("Rebuilding semantic index..."):
                    rebuild_index()

                st.success("Case updated and semantic index rebuilt.")
                st.rerun()

    st.divider()

    # -----------------------------
    # Delete case
    # -----------------------------

    st.subheader("🗑 Delete case")

    cases = load_cases()

    if cases.empty:
        st.info("No cases to delete.")
    else:
        delete_options = {
            f"Case #{row['id']} — {row['api_area']} — {row['problem']}": row["id"]
            for _, row in cases.iterrows()
        }

        delete_label = st.selectbox(
            "Select case to delete",
            list(delete_options.keys()),
            key="delete_case_select"
        )

        delete_id = delete_options[delete_label]

        confirm_delete = st.checkbox(
            "I understand this will permanently remove the case from the CSV file."
        )

        if st.button("Delete selected case", type="secondary"):

            if not confirm_delete:
                st.warning("Please confirm deletion first.")

            else:
                cases = cases[cases["id"] != delete_id]

                save_cases(cases)

                with st.spinner("Rebuilding semantic index..."):
                    rebuild_index()

                st.success("Case deleted and semantic index rebuilt.")
                st.rerun()

    st.divider()

    # -----------------------------
    # Current cases
    # -----------------------------

    st.subheader("Current cases")

    st.dataframe(
        load_cases(),
        use_container_width=True
    )