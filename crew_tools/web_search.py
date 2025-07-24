import json
import os
from typing import Optional

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SerperToolParameters(BaseModel):
    query: str = Field(..., description="The search query to find relevant results")


class SerperSearchTool(BaseTool):
    name: str = "serper_search_tool"
    description: str = (
        "Searches the internet using Serper and returns the top relevant results."
    )
    args_schema: Optional[type] = SerperToolParameters
    serper_api_key: str  # Must be passed when instantiating the tool

    def _run(self, query: str) -> str:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {"X-API-KEY": self.serper_api_key, "content-type": "application/json"}

        response = requests.post(url, headers=headers, data=payload)

        try:
            results = response.json().get("organic", [])
        except Exception as e:
            return json.dumps({"error": f"Failed to parse search results: {str(e)}"})

        if not results:
            return json.dumps(
                {"error": "No search results found or invalid API response."}
            )

        top_n = 3
        formatted_results = []

        for result in results[:top_n]:
            try:
                formatted_results.append(
                    {
                        "title": result["title"],
                        "link": result["link"],
                        "snippet": result["snippet"],
                    }
                )
            except KeyError:
                continue  # Skip if any field is missing

        return json.dumps(formatted_results, indent=2)


search_tool = SerperSearchTool(serper_api_key=os.getenv("SERPER_API_KEY"))
