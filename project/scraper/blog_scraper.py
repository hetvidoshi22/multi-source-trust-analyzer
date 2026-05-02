"""
blog_scraper.py
Scrapes 3 blog posts and returns structured JSON data.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from project.utils.tagging import auto_tag
from project.utils.chunking import chunk_text
from project.scoring.trust_score import calculate_trust_score


BLOG_URLS = [
    "https://realpython.com/python-web-scraping-practical-introduction/",  # Real Python: Web Scraping
    "https://realpython.com/python-requests/",                              # Real Python: Requests library
    "https://realpython.com/beautiful-soup-web-scraper-python/",           # Real Python: BeautifulSoup
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def detect_language(text: str) -> str:
    """
    Simple heuristic language detection.
    Returns 'en' for English by default.
    In production, use langdetect library.
    """
    try:
        from langdetect import detect
        return detect(text[:500]) if text else "en"
    except Exception:
        return "en"


def extract_author_medium(soup: BeautifulSoup) -> str:
    """Extract author from Medium-style pages."""
    # Try various selectors
    selectors = [
        {"name": "meta", "attrs": {"name": "author"}},
        {"name": "meta", "attrs": {"property": "article:author"}},
    ]
    for sel in selectors:
        tag = soup.find(sel["name"], attrs=sel["attrs"])
        if tag and tag.get("content"):
            return tag["content"].strip()

    # Try JSON-LD
    import json
    ld = soup.find("script", type="application/ld+json")
    if ld:
        try:
            data = json.loads(ld.string)
            if isinstance(data, list):
                data = data[0]
            author = data.get("author", {})
            if isinstance(author, dict):
                return author.get("name", "Unknown")
            if isinstance(author, list) and author:
                return author[0].get("name", "Unknown")
        except Exception:
            pass
    return "Unknown"


def extract_published_date(soup: BeautifulSoup) -> str:
    """Extract published date from common meta tags and JSON-LD."""
    import json
    # JSON-LD
    ld = soup.find("script", type="application/ld+json")
    if ld:
        try:
            data = json.loads(ld.string)
            if isinstance(data, list):
                data = data[0]
            date = data.get("datePublished") or data.get("dateCreated")
            if date:
                return date[:10]
        except Exception:
            pass

    # Meta tags
    for prop in ["article:published_time", "datePublished", "date"]:
        tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            return tag["content"][:10]

    # time element
    time_tag = soup.find("time")
    if time_tag:
        return time_tag.get("datetime", time_tag.get_text(strip=True))[:10]

    return "Unknown"


def extract_blog_content(soup: BeautifulSoup) -> str:
    """Extract main article body text, stripping nav/ads/footer."""
    # Remove noise tags
    for tag in soup.find_all(["nav", "header", "footer", "aside", "script", "style",
                               "form", "noscript", "iframe", "button", "advertisement"]):
        tag.decompose()

    # Priority selectors for article body
    candidates = [
        soup.find("article"),
        soup.find(class_=lambda c: c and any(k in str(c).lower() for k in ["article", "post-content", "entry-content", "blog-content"])),
        soup.find("main"),
        soup.find("div", {"id": "content"}),
    ]
    for candidate in candidates:
        if candidate:
            paragraphs = candidate.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)
            if len(text) > 200:
                return text

    # Fallback: all paragraphs
    paragraphs = soup.find_all("p")
    return "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)


def scrape_blog(url: str) -> dict:
    """Scrape a single blog post and return structured data."""
    print(f"  Scraping blog: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # --- Metadata ---
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title:
            title = og_title.get("content", "")
        if not title:
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"

        author = extract_author_medium(soup)
        published_date = extract_published_date(soup)
        content = extract_blog_content(soup)

        # Language detection
        language = detect_language(content or title)

        # Region (from og:locale or default)
        og_locale = soup.find("meta", property="og:locale")
        region = og_locale["content"] if og_locale and og_locale.get("content") else "Global"

        # Topic tags
        description = ""
        og_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
        if og_desc:
            description = og_desc.get("content", "")

        topic_tags = auto_tag(title + " " + description + " " + content[:1000])
        chunks = chunk_text(content)

        # Trust score
        domain = url.split("/")[2]
        trust_score = calculate_trust_score(
            author=author,
            published_date=published_date,
            domain=domain,
            citation_count=0,
            has_medical_disclaimer="disclaimer" in content.lower() or "consult" in content.lower(),
            source_type="blog",
        )

        return {
            "source_url": url,
            "source_type": "blog",
            "title": title,
            "author": author,
            "published_date": published_date,
            "language": language,
            "region": region,
            "topic_tags": topic_tags,
            "trust_score": round(trust_score, 3),
            "content_chunks": chunks,
        }

    except requests.exceptions.RequestException as e:
        print(f"    [ERROR] Failed to fetch {url}: {e}")
        return _fallback_blog_entry(url, str(e))


def _fallback_blog_entry(url: str, error: str) -> dict:
    """Return a fallback entry when scraping fails."""
    domain = url.split("/")[2] if "//" in url else url
    return {
        "source_url": url,
        "source_type": "blog",
        "title": "Unavailable",
        "author": "Unknown",
        "published_date": "Unknown",
        "language": "en",
        "region": "Global",
        "topic_tags": [],
        "trust_score": 0.0,
        "content_chunks": [f"[Scraping failed: {error}]"],
    }


def scrape_all_blogs() -> list:
    """Scrape all configured blog URLs."""
    results = []
    for url in BLOG_URLS:
        data = scrape_blog(url)
        results.append(data)
    return results


if __name__ == "__main__":
    import json
    print("Starting blog scraper...")
    blogs = scrape_all_blogs()
    os.makedirs("output/scraped_data", exist_ok=True)
    with open("output/scraped_data/blogs.json", "w", encoding="utf-8") as f:
        json.dump(blogs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(blogs)} blog entries.")
