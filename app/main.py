import httpx
import streamlit as st

from src.constants.api_url import API_BASE_URL

st.set_page_config(page_title="Codif APE Classifier", layout="centered")

st.title("🔎 Codif APE Classifier")

classifier = st.selectbox(
    "Select Classification Method",
    [
        "flat-rag",
        "flat-embeddings",
        "hierarchical-rag",
        "hierarchical-embeddings",
    ],
)

st.markdown("### Classify a Single Activity")
query = st.text_input("Activity Description")

if st.button("Classify"):
    if not query.strip():
        st.warning("Please enter a valid activity description.")
    else:
        with st.spinner("Classifying..."):
            try:
                response = httpx.get(f"{API_BASE_URL}/{classifier}/classify", params={"query": query}, timeout=30)
                response.raise_for_status()
                result = response.json()
                st.success(f"✅ APE Code: `{result['code_ape']}`")
            except Exception as e:
                st.error(f"❌ Error: {e}")


st.markdown("---")
st.markdown("### Batch Classification")

batch_input = st.text_area("Paste one activity per line")
if st.button("Batch Classify"):
    lines = [line.strip() for line in batch_input.splitlines() if line.strip()]
    if not lines:
        st.warning("Please enter at least one activity.")
    else:
        with st.spinner("Classifying batch..."):
            try:
                response = httpx.post(f"{API_BASE_URL}/{classifier}/batch", json={"queries": lines}, timeout=60)
                response.raise_for_status()
                results = response.json()
                st.table(results)
            except Exception as e:
                st.error(f"❌ Error: {e}")
