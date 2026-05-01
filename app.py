import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.vector_store import build_vector_store, semantic_search


load_dotenv()

CSV_PATH = "data/troubleshooting_cases_english.csv"

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found.")
    st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(
    page_title="API Troubleshooting Assistant",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ API Troubleshooting Assistant")


# =============================
# DATA
# =============================

@st.cache_data
def load_cases():
    df = pd.read_csv(
        CSV_PATH,
        sep=";",
        dtype={
            "id": int,
            "error_code": str
        }
    )

    for col in ["api_area", "endpoint", "error_code", "problem", "root_cause", "solution", "tags", "logs"]:
        if col not in df.columns:
            df[col] = ""

    df = df.fillna("")

    return df


def save_cases(df):
    df = df.copy()
    df["error_code"] = df["error_code"].astype(str)

    df.to_csv(
        CSV_PATH,
        sep=";",
        index=False
    )

    load_cases.clear()
    initialize_semantic_index.clear()


@st.cache_resource
def initialize_semantic_index():
    return build_vector_store(CSV_PATH)


def rebuild_index():
    load_cases.clear()
    initialize_semantic_index.clear()
    return build_vector_store(CSV_PATH)


with st.spinner("Preparing semantic index..."):
    indexed_count = initialize_semantic_index()


# =============================
# SESSION STATE
# =============================

if "page" not in st.session_state:
    st.session_state.page = "analyze"

if "history" not in st.session_state:
    st.session_state.history = []

if "suggested_tags" not in st.session_state:
    st.session_state.suggested_tags = ""


# =============================
# SIDEBAR NAVIGATION
# =============================

with st.sidebar:
    st.header("Navigation")

    if st.button("🔍 Analyze issue", use_container_width=True):
        st.session_state.page = "analyze"
        st.rerun()

    if st.button("📚 Manage knowledge base", use_container_width=True):
        st.session_state.page = "kb"
        st.rerun()

    st.divider()

    st.success(
        f"Semantic index ready ({indexed_count} cases)"
    )


# =============================
# HELPERS
# =============================

def parse_ai_json(reply):
    reply = reply.strip()

    if reply.startswith("```json"):
        reply = reply.replace("```json", "", 1).strip()

    if reply.startswith("```"):
        reply = reply.replace("```", "", 1).strip()

    if reply.endswith("```"):
        reply = reply[:-3].strip()

    return json.loads(reply)


def confidence_label(score):
    if score >= 0.75:
        return "High confidence"

    if score >= 0.45:
        return "Medium confidence"

    return "Low confidence"


def generate_customer_reply(message, match):
    case = match["case"]
    score = match["score"]

    prompt = f"""
You are an API support assistant helping a system analyst respond to a customer.

Customer message:
{message}

Most relevant troubleshooting case:
Case ID: {case.get("id", "")}
API Area: {case.get("api_area", "")}
Endpoint: {case.get("endpoint", "")}
Error Code: {case.get("error_code", "")}
Problem: {case.get("problem", "")}
Root cause: {case.get("root_cause", "")}
Solution: {case.get("solution", "")}
Tags: {case.get("tags", "")}
Logs: {case.get("logs", "")}
Semantic match score: {score}

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
- Tags must be short technical keywords
- Prefer API/support terminology
- No markdown code fences
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    data = parse_ai_json(response.output_text)

    return ", ".join(data["tags"])


def render_case_details(case):
    with st.container(border=True):
        st.markdown(f"### Case #{case['id']} — {case['api_area']}")

        st.write(f"**Endpoint:** {case['endpoint']}")
        st.write(f"**Error code:** {case['error_code']}")
        st.write(f"**Problem:** {case['problem']}")
        st.write(f"**Root cause:** {case['root_cause']}")
        st.write(f"**Solution:** {case['solution']}")

        if str(case.get("tags", "")).strip():
            st.write(f"**Tags:** {case['tags']}")

        if str(case.get("logs", "")).strip():
            with st.expander("Attached logs"):
                st.code(case["logs"])


# =============================
# PAGE — ANALYZE
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
            st.stop()

        matches = semantic_search(
            customer_message,
            top_k=1
        )

        if not matches:
            st.info("No cases found.")
            st.stop()

        match = matches[0]
        case = match["case"]
        score = match["score"]

        st.subheader("Most relevant troubleshooting case")

        render_case_details(case)

        st.markdown("### Confidence")
        st.progress(score)
        st.caption(
            f"{confidence_label(score)} — semantic match score: {score}"
        )

        st.subheader("AI-generated response")

        with st.spinner("Generating customer reply..."):
            reply = generate_customer_reply(
                customer_message,
                match
            )

        try:
            data = parse_ai_json(reply)

            col1, col2 = st.columns(2)

            with col1:
                with st.container(border=True):
                    st.markdown("### Issue summary")
                    st.write(data["issue_summary"])

            with col2:
                with st.container(border=True):
                    st.markdown("### Likely root cause")
                    st.write(data["root_cause"])

            with st.container(border=True):
                st.markdown("### Recommended next steps")
                for step in data["next_steps"]:
                    st.write(f"- {step}")

            with st.container(border=True):
                st.markdown("### Email draft")
                st.text_area(
                    "Generated email",
                    value=data["email_draft"],
                    height=250,
                    label_visibility="collapsed"
                )

            st.session_state.history.append({
                "title": f"{datetime.now().strftime('%H:%M')} — Case #{case['id']}",
                "customer_message": customer_message,
                "matched_case": f"Case #{case['id']} — {case['problem']}",
                "email_draft": data["email_draft"]
            })

        except json.JSONDecodeError:
            st.warning("AI response could not be parsed as structured JSON.")
            st.text_area(
                "Raw AI response",
                value=reply,
                height=350
            )


# =============================
# PAGE — KNOWLEDGE BASE
# =============================

if st.session_state.page == "kb":

    st.header("📚 Manage knowledge base")

    action = st.selectbox(
        "Select action",
        [
            "➕ Add case",
            "✏️ Edit case",
            "🗑 Delete case",
            "📋 View cases"
        ]
    )

    cases = load_cases()

    # -------------------------
    # ADD CASE
    # -------------------------

    if action == "➕ Add case":

        st.subheader("➕ Add new troubleshooting case")

        with st.form("add_form"):

            api_area = st.text_input("API area")
            endpoint = st.text_input("Endpoint")
            error_code = st.text_input("Error code")

            problem = st.text_area("Problem")
            root_cause = st.text_area("Root cause")
            solution = st.text_area("Solution")

            tags = st.text_input(
                "Tags",
                value=st.session_state.suggested_tags,
                placeholder="auth, token, invalid-credentials"
            )

            logs = st.text_area(
                "Logs / request example / error payload"
            )

            col1, col2 = st.columns(2)

            with col1:
                suggest_clicked = st.form_submit_button(
                    "🧠 Auto-suggest tags"
                )

            with col2:
                add_clicked = st.form_submit_button(
                    "Add case"
                )

            if suggest_clicked:

                if not problem.strip():
                    st.warning("Please fill at least Problem first.")
                else:
                    with st.spinner("Suggesting tags..."):
                        st.session_state.suggested_tags = suggest_tags(
                            api_area,
                            endpoint,
                            error_code,
                            problem,
                            root_cause,
                            solution
                        )

                    st.rerun()

            if add_clicked:

                if not problem.strip() or not solution.strip():
                    st.warning("Problem and Solution are required.")
                else:
                    new_id = (
                        int(cases["id"].max()) + 1
                        if not cases.empty
                        else 1
                    )

                    new_row = pd.DataFrame([{
                        "id": new_id,
                        "api_area": api_area,
                        "endpoint": endpoint,
                        "error_code": str(error_code),
                        "problem": problem,
                        "root_cause": root_cause,
                        "solution": solution,
                        "tags": tags,
                        "logs": logs
                    }])

                    updated_cases = pd.concat(
                        [cases, new_row],
                        ignore_index=True
                    )

                    save_cases(updated_cases)

                    with st.spinner("Rebuilding semantic index..."):
                        rebuild_index()

                    st.session_state.suggested_tags = ""

                    st.success("Case added and semantic index updated.")
                    st.rerun()

    # -------------------------
    # EDIT CASE
    # -------------------------

    if action == "✏️ Edit case":

        st.subheader("✏️ Edit troubleshooting case")

        if cases.empty:
            st.info("Knowledge base is empty.")
        else:
            options = {
                f"Case #{row['id']} — {row['api_area']} — {row['problem']}": row["id"]
                for _, row in cases.iterrows()
            }

            selected_label = st.selectbox(
                "Select case to edit",
                list(options.keys())
            )

            selected_id = options[selected_label]
            selected_case = cases[
                cases["id"] == selected_id
            ].iloc[0]

            st.markdown("### Current case details")
            render_case_details(selected_case)

            with st.form("edit_form"):

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
                    "Problem",
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
                    value=str(selected_case["tags"])
                )

                edit_logs = st.text_area(
                    "Logs / request example / error payload",
                    value=str(selected_case["logs"])
                )

                save_clicked = st.form_submit_button(
                    "Save changes"
                )

                if save_clicked:

                    cases.loc[
                        cases["id"] == selected_id,
                        "api_area"
                    ] = edit_api_area

                    cases.loc[
                        cases["id"] == selected_id,
                        "endpoint"
                    ] = edit_endpoint

                    cases.loc[
                        cases["id"] == selected_id,
                        "error_code"
                    ] = str(edit_error_code)

                    cases.loc[
                        cases["id"] == selected_id,
                        "problem"
                    ] = edit_problem

                    cases.loc[
                        cases["id"] == selected_id,
                        "root_cause"
                    ] = edit_root_cause

                    cases.loc[
                        cases["id"] == selected_id,
                        "solution"
                    ] = edit_solution

                    cases.loc[
                        cases["id"] == selected_id,
                        "tags"
                    ] = edit_tags

                    cases.loc[
                        cases["id"] == selected_id,
                        "logs"
                    ] = edit_logs

                    save_cases(cases)

                    with st.spinner("Rebuilding semantic index..."):
                        rebuild_index()

                    st.success("Case updated and semantic index rebuilt.")
                    st.rerun()

    # -------------------------
    # DELETE CASE
    # -------------------------

    if action == "🗑 Delete case":

        st.subheader("🗑 Delete troubleshooting case")

        if cases.empty:
            st.info("No cases to delete.")
        else:
            options = {
                f"Case #{row['id']} — {row['api_area']} — {row['problem']}": row["id"]
                for _, row in cases.iterrows()
            }

            selected_label = st.selectbox(
                "Select case to delete",
                list(options.keys())
            )

            selected_id = options[selected_label]
            selected_case = cases[
                cases["id"] == selected_id
            ].iloc[0]

            st.markdown("### Selected case details")
            render_case_details(selected_case)

            confirm_delete = st.checkbox(
                "I understand this will permanently remove the case from the CSV file."
            )

            if st.button("Delete selected case", type="secondary"):

                if not confirm_delete:
                    st.warning("Please confirm deletion first.")
                else:
                    updated_cases = cases[
                        cases["id"] != selected_id
                    ]

                    save_cases(updated_cases)

                    with st.spinner("Rebuilding semantic index..."):
                        rebuild_index()

                    st.success("Case deleted and semantic index rebuilt.")
                    st.rerun()

    # -------------------------
    # VIEW CASES
    # -------------------------

    if action == "📋 View cases":

        st.subheader("📋 View cases")

        st.write("Select a row in the table to view full case details.")

        event = st.dataframe(
            cases,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )

        selected_rows = event.selection.rows

        if selected_rows:
            selected_index = selected_rows[0]
            selected_case = cases.iloc[selected_index]

            st.markdown("### Selected case details")
            render_case_details(selected_case)