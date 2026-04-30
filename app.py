import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="API Troubleshooting Assistant",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ API Troubleshooting Assistant")
st.write("Paste a customer API issue and find similar troubleshooting cases.")

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

    results = sorted(results, key=lambda x: x[0], reverse=True)
    return results[:3]

if st.button("Analyze issue"):
    if not customer_message.strip():
        st.warning("Please paste a customer message first.")
    else:
        st.subheader("Similar troubleshooting cases")

        matches = simple_search(customer_message, cases)

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
                    