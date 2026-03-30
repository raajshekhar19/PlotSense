from dotenv import load_dotenv
import os
load_dotenv("/Users/shresthpanigrahi/Desktop/Movie/.env")

from langchain_community.tools.tavily_search import TavilySearchResults
import json

try:
    tool = TavilySearchResults(max_results=1, include_images=True)
    res = tool.invoke({"query": "Inception movie poster"})
    print("langchain_community tool test:")
    print(res)
except Exception as e:
    print("Community tool error:", e)

print("---")

try:
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    res2 = client.search(query="Inception movie poster", search_depth="basic", include_images=True, max_results=1)
    print("Raw Client images:")
    print(res2.get("images", "No images found"))
except Exception as e:
    print("Raw client error:", e)
