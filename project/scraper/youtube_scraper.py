"""
youtube_scraper.py
Scrapes 2 YouTube videos and returns structured JSON data.
Uses YouTube Data API v3 and youtube-transcript-api for transcripts.
"""

import requests
import os
import sys
import re
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from project.utils.tagging import auto_tag
from project.utils.chunking import chunk_text
from project.scoring.trust_score import calculate_trust_score

# Two YouTube video URLs (AI/ML and Data Science topics)
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=aircAruvnKk",   # 3Blue1Brown: Neural Networks
    "https://www.youtube.com/watch?v=ukzFI9rgwfU",   # Sentdex: Machine Learning
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_\-]{11})",
        r"(?:embed/)([A-Za-z0-9_\-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def fetch_transcript(video_id: str) -> str:
    """
    Fetch transcript using youtube-transcript-api.
    Supports both v0.x (class method) and v1.x (instance method) API.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        # v1.x API: instantiate first, then call fetch()
        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id)
            # fetched is a FetchedTranscript; iterate its snippets
            text = " ".join(
                snippet.text for snippet in fetched
            )
            return text
        except AttributeError:
            # Fallback to v0.x class-method style
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["en", "en-US"]
            )
            return " ".join(entry["text"] for entry in transcript_list)
    except Exception as e:
        print(f"    [WARN] Transcript unavailable for {video_id}: {e}")
        return ""


def scrape_youtube_page(url: str) -> dict:
    """
    Scrape YouTube video metadata from the page HTML (no API key needed).
    Falls back to yt-dlp if available.
    """
    video_id = extract_video_id(url)
    print(f"  Scraping YouTube video: {url} (ID: {video_id})")

    try:
        # --- Method 1: yt-dlp (most reliable) ---
        try:
            import yt_dlp
            ydl_opts = {
                "quiet": True,
                "skip_download": True,
                "extract_flat": False,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get("title", "Unknown")
                channel = info.get("uploader", info.get("channel", "Unknown"))
                upload_date_raw = info.get("upload_date", "")  # YYYYMMDD
                if upload_date_raw and len(upload_date_raw) == 8:
                    published_date = f"{upload_date_raw[:4]}-{upload_date_raw[4:6]}-{upload_date_raw[6:8]}"
                else:
                    published_date = "Unknown"
                description = info.get("description", "") or ""
                view_count = info.get("view_count", 0)
                like_count = info.get("like_count", 0)

        except ImportError:
            # --- Method 2: Scrape page HTML ---
            import json
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            html = response.text

            # Extract ytInitialData JSON
            match = re.search(r"var ytInitialData = ({.*?});</script>", html, re.DOTALL)
            if not match:
                match = re.search(r"ytInitialData\s*=\s*({.*?});\s*</script>", html, re.DOTALL)

            title = "Unknown"
            channel = "Unknown"
            published_date = "Unknown"
            description = ""
            view_count = 0
            like_count = 0

            # Try meta tags (most reliable fallback)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            og_title = soup.find("meta", property="og:title")
            if og_title:
                title = og_title.get("content", "Unknown")

            og_desc = soup.find("meta", property="og:description")
            if og_desc:
                description = og_desc.get("content", "")

            # Channel from itemprop
            channel_tag = soup.find("link", itemprop="name")
            if channel_tag:
                channel = channel_tag.get("content", "Unknown")

            # Date from meta
            date_tag = soup.find("meta", itemprop="datePublished")
            if date_tag:
                published_date = date_tag.get("content", "Unknown")[:10]

            view_tag = soup.find("meta", itemprop="interactionCount")
            if view_tag:
                try:
                    view_count = int(view_tag.get("content", 0))
                except ValueError:
                    view_count = 0

        # --- Fetch transcript ---
        transcript_text = fetch_transcript(video_id) if video_id else ""
        content = transcript_text if transcript_text else description

        # Language detection
        try:
            from langdetect import detect
            language = detect(content[:500]) if content.strip() else "en"
        except Exception:
            language = "en"

        # Topic tagging
        topic_tags = auto_tag(title + " " + description[:500] + " " + content[:500])

        # Content chunks (from transcript or description)
        chunks = chunk_text(content) if content else [description]

        # Trust score
        domain = "youtube.com"
        trust_score = calculate_trust_score(
            author=channel,
            published_date=published_date,
            domain=domain,
            citation_count=0,
            has_medical_disclaimer="disclaimer" in description.lower(),
            source_type="youtube",
            view_count=view_count,
        )

        return {
            "source_url": url,
            "source_type": "youtube",
            "video_id": video_id,
            "title": title,
            "author": channel,
            "published_date": published_date,
            "language": language,
            "region": "Global",
            "topic_tags": topic_tags,
            "trust_score": round(trust_score, 3),
            "view_count": view_count,
            "like_count": like_count,
            "content_chunks": chunks,
        }

    except Exception as e:
        print(f"    [ERROR] Failed to scrape {url}: {e}")
        return _fallback_youtube_entry(url, video_id, str(e))


def _fallback_youtube_entry(url: str, video_id: str, error: str) -> dict:
    return {
        "source_url": url,
        "source_type": "youtube",
        "video_id": video_id,
        "title": "Unavailable",
        "author": "Unknown",
        "published_date": "Unknown",
        "language": "en",
        "region": "Global",
        "topic_tags": [],
        "trust_score": 0.0,
        "view_count": 0,
        "like_count": 0,
        "content_chunks": [f"[Scraping failed: {error}]"],
    }


def scrape_all_youtube() -> list:
    """Scrape all configured YouTube URLs."""
    results = []
    for url in YOUTUBE_URLS:
        data = scrape_youtube_page(url)
        results.append(data)
    return results


if __name__ == "__main__":
    import json
    print("Starting YouTube scraper...")
    videos = scrape_all_youtube()
    os.makedirs("output/scraped_data", exist_ok=True)
    with open("output/scraped_data/youtube.json", "w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(videos)} YouTube entries.")
