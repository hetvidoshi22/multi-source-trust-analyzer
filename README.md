# Multi-Source Data Scraping & Trust Scoring System

A Python pipeline that scrapes structured content from **3 blog posts**, **2 YouTube videos**, and **1 PubMed article**, then evaluates each source using a custom **Trust Score Algorithm**.

---

## Project Structure

```
project/
├── scraper/
│   ├── blog_scraper.py       # Scrapes 3 blog posts
│   ├── youtube_scraper.py    # Scrapes 2 YouTube videos
│   └── pubmed_scraper.py     # Scrapes 1 PubMed article (via E-utilities API)
├── scoring/
│   └── trust_score.py        # Trust score algorithm (5 components)
└── utils/
    ├── tagging.py            # Automatic topic tagging (taxonomy + TF-IDF)
    └── chunking.py           # Content chunking (paragraph → sentence → word)
main.py                       # Orchestrates the full pipeline
requirements.txt
output/
├── scraped_data.json         # All 6 sources combined
└── scraped_data/
    ├── blogs.json            # 3 blog entries
    ├── youtube.json          # 2 YouTube entries
    └── pubmed.json           # 1 PubMed entry
```

---

## Tools & Libraries

| Library | Purpose |
|---|---|
| `requests` | HTTP requests for blog and PubMed pages |
| `beautifulsoup4` + `lxml` | HTML parsing and content extraction |
| `yt-dlp` | YouTube metadata extraction (title, channel, date, description) |
| `youtube-transcript-api` | YouTube transcript retrieval |
| `langdetect` | Automatic language detection |
| Python `xml.etree.ElementTree` | PubMed XML parsing via E-utilities API |
| Python `math`, `re`, `collections` | Trust scoring and tagging logic |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python -X utf8 main.py
```

### 3. Run individual scrapers

```bash
python -X utf8 main.py --blogs      # Blogs only
python -X utf8 main.py --youtube    # YouTube only
python -X utf8 main.py --pubmed     # PubMed only
```

### 4. View trust score demo

```bash
python -X utf8 main.py --trust-demo
```

> **Note:** Use `-X utf8` flag on Windows to avoid encoding issues with the console.

---

## Output Format

Each scraped record follows this schema:

```json
{
  "source_url": "https://...",
  "source_type": "blog | youtube | pubmed",
  "title": "Article or video title",
  "author": "Author or channel name",
  "published_date": "YYYY-MM-DD",
  "language": "en",
  "region": "Global | International",
  "topic_tags": ["Web Scraping", "Python", "Education"],
  "trust_score": 0.692,
  "content_chunks": ["Paragraph 1...", "Paragraph 2...", "..."]
}
```

---

## Scraping Approach

### Blogs
- HTTP GET with a browser-like `User-Agent` header
- BeautifulSoup parses `<article>`, `<main>`, or `.post-content` for body text
- Metadata extracted from `og:` meta tags and JSON-LD structured data
- Navigation, ads, scripts, and footers removed before content extraction

### YouTube
- **yt-dlp** extracts video metadata (title, channel, upload date, description) without needing an API key
- **youtube-transcript-api** fetches English transcripts (auto-generated or manual)
- Supports both v0.x and v1.x API of `youtube-transcript-api`

### PubMed
- Uses the **NCBI E-utilities API** (no API key required for basic access)
- `efetch` endpoint retrieves full XML for each PMID
- Parses title, authors, journal, abstract, MeSH terms, and publication date

---

## Trust Score Design

```
Trust Score = 0.25 × author_credibility
            + 0.25 × domain_authority
            + 0.20 × recency
            + 0.20 × citation_count
            + 0.10 × medical_disclaimer_presence
```

| Component | Scoring Logic |
|---|---|
| **Author credibility** | Known org → 0.95; real name pattern → 0.65; missing → 0.10 |
| **Domain authority** | Curated map (pubmed=1.0, realpython=0.75); TLD heuristic fallback |
| **Recency** | Exponential decay: `exp(-0.20 × years_old)`; unknown date → 0.20 |
| **Citation count** | Log-normalized (PubMed only); blogs/YouTube default to 0.50 |
| **Medical disclaimer** | PubMed=1.0; disclaimer present=0.90; health content without=0.20 |

### Abuse Prevention
- **Fake authors** (digits or ALL CAPS in name) → credibility = 0.15
- **Spammy TLDs** (`.xyz`, `.info`, `.biz`) → domain authority = 0.10
- **Misinformation keywords** (e.g. "miracle cure") → disclaimer score = 0.0
- **Future-dated content** → recency = 0.30
- **Viral but uncredible YouTube** (>10M views + low author score) → 15% penalty

---

## Topic Tagging

Tags are generated in two stages:
1. **Taxonomy matching** — text matched against a curated keyword-to-topic map (e.g. "neural network" → "AI")
2. **TF-IDF keyword extraction** — top-N words by frequency × log(word_length), filtered by stopwords

---

## Limitations

- **Medium.com** and paywalled articles block scrapers — replaced with open-access alternatives
- **YouTube transcripts** may be unavailable for some videos (auto-captions disabled)
- **yt-dlp** warns about missing JavaScript runtime (ffmpeg optional, no impact on metadata)
- Citation counts via PubMed `elink` are approximate (only counts PubMed-indexed citations)
- Language detection requires ≥50 characters of text to be accurate
- `langdetect` is non-deterministic by design — results may vary slightly between runs
