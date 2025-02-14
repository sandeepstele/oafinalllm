from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
import markdown
import uvicorn

app = FastAPI()

# Allow CORS from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_wikipedia_url(country: str) -> str:
    """
    Given a country name, returns the Wikipedia URL for the country.
    """
    return f"https://en.wikipedia.org/wiki/{country}"

def extract_headings_from_html(html: str) -> list:
    """
    Extract all headings (H1 to H6) from the given HTML and return a list.
    """
    soup = BeautifulSoup(html, "html.parser")
    headings = []

    # Loop through all the heading tags (H1 to H6)
    for level in range(1, 7):
        for tag in soup.find_all(f'h{level}'):
            headings.append((level, tag.get_text(strip=True)))

    return headings

def generate_markdown_outline(headings: list) -> str:
    """
    Converts the extracted headings into a markdown-formatted outline.
    """
    markdown_outline = "## Contents\n\n"
    for level, heading in headings:
        markdown_outline += "#" * level + f" {heading}\n\n"
    return markdown_outline

@app.get("/api/outline")
async def get_country_outline(country: str):
    """
    API endpoint that returns the markdown outline of the given country Wikipedia page.
    """
    if not country:
        raise HTTPException(status_code=400, detail="Country parameter is required")

    # Fetch Wikipedia page
    url = get_wikipedia_url(country)
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=404, detail=f"Error fetching Wikipedia page: {e}")

    # Extract headings and generate markdown outline
    headings = extract_headings_from_html(response.text)
    if not headings:
        raise HTTPException(status_code=404, detail="No headings found in the Wikipedia page")

    markdown_outline = generate_markdown_outline(headings)
    return JSONResponse(content={"outline": markdown_outline})
