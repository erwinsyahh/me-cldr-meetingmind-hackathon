from crewai import Crew

from crew_agents.agents_and_task import (
    inquisitive_information_analyst,
    meeting_action_item_extractor,
    meeting_email_composer,
    meeting_orchestration_leader,
    meeting_summary_specialist,
    meeting_terminology_extractor,
    task1,
    task2,
)
from scripts.utils import transcribe_gcs_video_with_cache


def transcript_to_text(gcs_video_uri) -> str:
    full_transcript, _ = transcribe_gcs_video_with_cache(
        gcs_video_uri=gcs_video_uri,
        google_credentials_path="secrets/secret.json",
    )

    return full_transcript


def crew_launch(meeting_transcript: str):
    print("Instantiating MeetingMind Crew...")

    # === Step 1: Initialize Crew ===
    crew = Crew(
        agents=[
            meeting_summary_specialist,
            meeting_action_item_extractor,
            inquisitive_information_analyst,
            meeting_terminology_extractor,
            meeting_email_composer,
        ],
        tasks=[task1, task2],
        manager_agent=meeting_orchestration_leader,
        verbose=True,
    )

    inputs = {
        "meeting_transcript": meeting_transcript,
    }

    # === Step 3: Kick Off the Crew ===
    print("Launching Crew with inputs...")
    result = crew.kickoff(inputs=inputs)

    return result
