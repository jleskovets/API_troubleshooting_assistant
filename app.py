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
        dtype={"error_code": str}
    )

    for col in ["tags", "logs"]:
        if col not in df.columns:
            df[col] = ""

    return df


def save_cases(df):
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
# NAVIGATION BUTTONS
# =============================

if "page" not in st.session_state:
    st.session_state.page = "analyze"

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

    if reply.startswith("```"):
        reply = reply.replace("```json", "")
        reply = reply.replace("```", "")

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
Customer message:
{message}

Relevant case:
Problem: {case['problem']}
Root cause: {case['root_cause']}
Solution: {case['solution']}

Return JSON:
{{
"issue_summary": "...",
"root_cause": "...",
"next_steps": ["..."],
"email_draft": "..."
}}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


# =============================
# PAGE — ANALYZE
# =============================

if st.session_state.page == "analyze":

    customer_message = st.text_area(
        "Customer message",
        height=200
    )

    if st.button("Analyze issue", type="primary"):

        if not customer_message.strip():
            st.warning("Please paste message.")
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

        st.subheader("Most relevant case")

        with st.container(border=True):

            st.write(
                f"**Endpoint:** {case['endpoint']}"
            )

            st.write(
                f"**Error:** {case['error_code']}"
            )

            st.write(
                f"**Problem:** {case['problem']}"
            )

            st.write(
                f"**Solution:** {case['solution']}"
            )

            st.progress(score)

            st.caption(
                f"{confidence_label(score)} ({score})"
            )

        st.subheader("AI-generated response")

        with st.spinner("Generating..."):

            reply = generate_customer_reply(
                customer_message,
                match
            )

        data = parse_ai_json(reply)

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Issue summary")
            st.write(data["issue_summary"])

        with col2:

            st.markdown("### Root cause")
            st.write(data["root_cause"])

        st.markdown("### Next steps")

        for step in data["next_steps"]:
            st.write(f"- {step}")

        st.markdown("### Email draft")

        st.text_area(
            "",
            value=data["email_draft"],
            height=250
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
    # ADD
    # -------------------------

    if action == "➕ Add case":

        with st.form("add_form", clear_on_submit=True):

            api_area = st.text_input("API area")
            endpoint = st.text_input("Endpoint")
            error_code = st.text_input("Error code")

            problem = st.text_area("Problem")
            root_cause = st.text_area("Root cause")
            solution = st.text_area("Solution")

            tags = st.text_input("Tags")
            logs = st.text_area("Logs")

            submitted = st.form_submit_button("Add case")

            if submitted:

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

                cases = pd.concat(
                    [cases, new_row],
                    ignore_index=True
                )

                save_cases(cases)

                rebuild_index()

                st.success("Case added!")


    # -------------------------
    # EDIT
    # -------------------------

    if action == "✏️ Edit case":

        options = {
            f"Case #{r['id']} — {r['problem']}": r["id"]
            for _, r in cases.iterrows()
        }

        label = st.selectbox(
            "Select case",
            list(options.keys())
        )

        selected_id = options[label]

        row = cases[
            cases["id"] == selected_id
        ].iloc[0]

        with st.form("edit_form"):

            endpoint = st.text_input(
                "Endpoint",
                value=row["endpoint"]
            )

            error_code = st.text_input(
                "Error",
                value=str(row["error_code"])
            )

            problem = st.text_area(
                "Problem",
                value=row["problem"]
            )

            solution = st.text_area(
                "Solution",
                value=row["solution"]
            )

            submitted = st.form_submit_button(
                "Save changes"
            )

            if submitted:

                cases.loc[
                    cases["id"] == selected_id,
                    "endpoint"
                ] = endpoint

                cases.loc[
                    cases["id"] == selected_id,
                    "error_code"
                ] = str(error_code)

                cases.loc[
                    cases["id"] == selected_id,
                    "problem"
                ] = problem

                cases.loc[
                    cases["id"] == selected_id,
                    "solution"
                ] = solution

                save_cases(cases)

                rebuild_index()

                st.success("Updated!")


    # -------------------------
    # DELETE
    # -------------------------

    if action == "🗑 Delete case":

        options = {
            f"Case #{r['id']} — {r['problem']}": r["id"]
            for _, r in cases.iterrows()
        }

        label = st.selectbox(
            "Select case to delete",
            list(options.keys())
        )

        delete_id = options[label]

        if st.button("Delete"):

            cases = cases[
                cases["id"] != delete_id
            ]

            save_cases(cases)

            rebuild_index()

            st.success("Deleted!")


    # -------------------------
    # VIEW
    # -------------------------

    if action == "📋 View cases":

        st.dataframe(
            cases,
            use_container_width=True
        )