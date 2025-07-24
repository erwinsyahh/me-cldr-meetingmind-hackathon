import os
import tempfile

import streamlit as st

from agent_launch import crew_launch, transcript_to_text
from scripts.utils import upload_to_gcs  # assumes this exists and works

st.set_page_config(page_title="MeetingMind Crew", layout="wide")
st.title("🤖📋 MeetingMind AI Assistant")

st.markdown(
    """
    Upload a meeting video or provide a GCS URI to automatically extract a transcript, summarize it, extract action items, and generate a professional meeting email.
    """
)

# --- Input Section ---
st.subheader("📥 Input Meeting Source")

gcs_uri = st.text_input(
    "GCS Video URI (optional)", placeholder="gs://your-bucket/video.mp4"
)

uploaded_file = st.file_uploader(
    "Or upload a local video", type=["mp4", "mov", "mkv"], label_visibility="visible"
)


start_button = st.button("🚀 Start Meeting Analysis")

# --- Processing ---
if start_button:

    if not gcs_uri and not uploaded_file:
        st.error("❌ Please provide either a GCS URI or upload a video file.")
        st.stop()

    with st.spinner("🔄 Processing input..."):

        # If local file uploaded, upload to GCS first
        if uploaded_file:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_file.write(uploaded_file.read())
            tmp_file.flush()

            bucket_name = "tes-cloudera"
            blob_name = f"video/{os.path.basename(tmp_file.name)}"

            st.info("Uploading video to GCS...")
            gcs_uri = upload_to_gcs(tmp_file.name, bucket_name, blob_name)
            st.success(f"✅ Uploaded to GCS: `{gcs_uri}`")

        # Step 1: Transcribe
        st.info("Transcribing video...")
        transcript = transcript_to_text(gcs_uri)

        st.text_area("📄 Transcript Preview", transcript, height=200)

        # Step 2: Launch Crew
        st.info("Running CrewAI agents...")
        result = crew_launch(transcript)

    # --- Output ---
    st.success("✅ Crew execution completed!")
