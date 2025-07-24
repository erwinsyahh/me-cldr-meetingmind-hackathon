import json
from typing import Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# Define parameters for tool input
class EmployeeToolParameters(BaseModel):
    input1: str = Field(
        ..., description="Employee ID or name used to look up employee details"
    )


# Define the CrewAI-compatible tool
class EmployeeProfileLookupTool(BaseTool):
    name: str = "employee_profile_lookup"
    description: str = (
        "Fetches employee profile data including their role in the project and current responsibilities. "
        "Takes an employee name or ID as input and returns a JSON string with their profile details."
    )
    args_schema: Optional[type] = EmployeeToolParameters

    def _run(self, input1: str) -> str:
        employee_id = input1.lower()

        # Dummy employee database
        employee_lookup = {
            "jack": {
                "employee_id": "jack",
                "name": "Jack",
                "role": "Moderator / Host",
                "email": "jack@whymeadows.com",
                "responsibilities": [
                    "Facilitating the panel discussion",
                    "Moderating breakout session summaries",
                    "Closing the event and acknowledging contributors",
                ],
            },
            "steve": {
                "employee_id": "steve",
                "name": "Steve Fiore",
                "role": "Senior Director, Customer Experiences",
                "email": "steve.fiore@teradata.com",
                "responsibilities": [
                    "Presenting on customer health metrics",
                    "Driving organizational alignment on success goals",
                    "Leading AI Innovation Days for customers",
                ],
            },
            "maddie": {
                "employee_id": "maddie",
                "name": "Maddie",
                "role": "Customer Experience Leader (B2C/D2C)",
                "email": "maddie@b2ccompany.com",
                "responsibilities": [
                    "Sharing perspectives on transactional customer metrics",
                    "Critiquing traditional metrics like NPS",
                    "Exploring retention in high-volume customer environments",
                ],
            },
            "michael": {
                "employee_id": "michael",
                "name": "Michael",
                "role": "Client Success & Support Lead",
                "email": "michael@startuptech.com",
                "responsibilities": [
                    "Tracking last logins and usage trends",
                    "Monitoring data feed continuity",
                    "Providing customer insight to cross-functional teams",
                ],
            },
            "rahil": {
                "employee_id": "rahil",
                "name": "Rahil",
                "role": "CX Metrics and AI Strategist",
                "email": "rahil@datatech.com",
                "responsibilities": [
                    "Developing customer health scorecards",
                    "Integrating AI into customer support metrics",
                    "Bringing cross-industry metric experience",
                ],
            },
            "alan": {
                "employee_id": "alan",
                "name": "Alan Rich",
                "role": "Founder & CEO",
                "email": "alan.rich@whymeadows.com",
                "responsibilities": [
                    "Organizing the Brain Trust event",
                    "Leading WhyMeadows’ strategic direction",
                    "Supporting executive community initiatives",
                ],
            },
        }

        profile = employee_lookup.get(employee_id)
        if profile:
            return json.dumps(profile, indent=2)
        else:
            return json.dumps(
                {"error": f"No employee found for ID '{employee_id}'"}, indent=2
            )


employee_tool = EmployeeProfileLookupTool()
