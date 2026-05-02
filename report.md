---
title: "Data Scraping & Trust Scoring - Report"
date: "2026-05-01"
---

# Data Scraping & Trust Scoring - Report

---

## 1. Scraping Strategy

The pipeline targets three distinct content platforms using platform-appropriate techniques.

### Blog Posts (3 sources — Real Python)
Blog scraping uses the `requests` library to issue HTTP GET requests with a browser-like `User-Agent` header to avoid bot detection. The response HTML is parsed with `BeautifulSoup4` using the `lxml` backend for speed. Noise elements (navigation bars, footers, sidebars, ads, `<script>`, `<style>` tags) are stripped before content extraction. The main article body is located through a priority hierarchy of CSS selectors: `<article>` → class-based selectors containing "article" or "post-content" → `<main>` → `<div id="content">`. Metadata (author, date, description) is extracted from Open Graph meta tags (`og:title`, `og:description`, `article:published_time`) and JSON-LD structured data embedded in `<script type="application/ld+json">` blocks. Language is auto-detected with `langdetect`.

**Challenge & Solution:** Paywalled (Medium) and deleted (TDS) articles returned 401/404 errors. These were replaced with open-access Real Python articles which are reliably crawlable and rich in technical content.

### YouTube Videos (2 sources)
YouTube metadata is extracted using `yt-dlp`, which reverse-engineers YouTube's internal API without requiring an official API key. This provides video title, channel name, upload date, description, and view/like counts reliably. Video transcripts are fetched separately using `youtube-transcript-api` (v1.x), which retrieves auto-generated or manual English captions as timestamped text segments that are then joined into a plain-text string and chunked.

**Challenge & Solution:** The `youtube-transcript-api` library underwent a breaking API change in v1.x (class method → instance method). The scraper was updated to handle both API versions gracefully with a try/except fallback.

### PubMed Article (1 source — PMID 37349072)
The PubMed scraper uses the **NCBI E-utilities REST API** (`efetch`) to retrieve full article XML for a given PMID. This is the recommended programmatic access method per NCBI policy and does not require API key registration for low-volume use. The XML is parsed with Python's built-in `xml.etree.ElementTree`. Extracted fields include article title, all authors, journal name, publication date, abstract with section labels (Background, Methods, Results, Conclusions), and MeSH controlled vocabulary terms. Citation counts are approximated via the `elink` API (`pubmed_pubmed_citedin` linkset).

---

## 2. Topic Tagging Method

Topic tagging operates in two complementary stages:

**Stage 1 — Taxonomy Matching:** A curated dictionary maps 15 topic categories (AI, Machine Learning, Healthcare, Web Scraping, Python, Research, etc.) to associated keywords. The input text (title + description + first 500 chars of content) is normalized (lowercased, HTML-stripped) and matched against each keyword list using `re.findall` with word boundaries. Topics are ranked by total keyword hit count and the top matches are selected.

**Stage 2 — TF-IDF Keyword Extraction:** To capture domain-specific terms not covered by the taxonomy, a simplified term frequency approach extracts the top-N words by a score of `frequency × log(word_length)`. This scoring favors longer, more informative words over common short words. Stopwords (a curated list of 100+ common English words) are filtered out.

The final tag list concatenates taxonomy labels first (more human-readable), followed by extracted keywords, with a cap of 8 tags per record.

---

## 3. Trust Score Algorithm

The Trust Score is a weighted linear combination of five normalized components, each scored in the range [0, 1]:

```
Trust Score = 0.25 × author_credibility
            + 0.25 × domain_authority
            + 0.20 × recency
            + 0.20 × citation_count
            + 0.10 × medical_disclaimer_presence
```

**Author Credibility (weight 0.25):** Checks the author string against a whitelist of ~25 known credible organizations (WHO, CDC, NIH, IEEE, named researchers). A real-name pattern (`First Last`) scores 0.65. Missing or anonymized authors score 0.10. For PubMed sources, all authors get a baseline of 0.85 since peer review implies vetting. Multiple authors are averaged.

**Domain Authority (weight 0.25):** A hand-curated map assigns authority scores to ~40 known domains (pubmed.ncbi.nlm.nih.gov=1.0, nature.com=0.97, realpython.com=0.75, medium.com=0.60, blogspot.com=0.25). Unknown domains use a TLD-based heuristic: `.edu`=0.85, `.gov`=0.90, `.org`=0.60, `.xyz`=0.10.

**Recency (weight 0.20):** Uses exponential decay: `score = exp(−0.20 × years_old)`. Content published within the past year scores ≥0.82. A missing or unparseable date scores 0.20. A future-dated article scores 0.30 (treated as suspicious).

**Citation Count (weight 0.20):** Log-normalized against a reference maximum of 500 citations: `score = log(1 + citations) / log(501)`. Blogs and YouTube default to 0.50 (neutral, since citation counts are not applicable to these media). A PubMed article with zero citations scores 0.30.

**Medical Disclaimer (weight 0.10):** PubMed articles score 1.0 implicitly (peer review serves this role). Blogs/videos with an explicit disclaimer score 0.90. Health or medical content without a disclaimer scores 0.20. Any content containing misinformation keywords (e.g., "miracle cure", "doctors don't want you to know") scores 0.0 on this component.

### Sample Results

| Source | Trust Score | Key Driver |
|---|---|---|
| PubMed (PMID 37349072) | **0.698** | Peer-reviewed + high domain authority |
| Real Python (Web Scraping) | **0.692** | Known domain + real author + recent |
| Real Python (Requests) | **0.681** | Slight recency penalty (future date flag) |
| Real Python (BeautifulSoup) | **0.661** | Same domain, older date |
| 3Blue1Brown (Neural Networks) | **0.596** | YouTube domain + 2017 publish date |
| Simplilearn (Machine Learning) | **0.491** | Unknown-tier channel + 2018 date |

---

## 4. Edge Case Handling

| Edge Case | Detection | Handling |
|---|---|---|
| Missing author | `author` is empty or "Unknown" | `author_credibility = 0.10` |
| Missing date | Cannot parse `published_date` | `recency = 0.20` |
| Transcript unavailable | Exception in transcript API | Falls back to video description for content |
| Multiple authors (PubMed) | Semicolon-separated string | Average of individual credibility scores |
| Non-English content | `langdetect` on content text | Language field stored; scoring unaffected |
| Long articles | Always runs chunker | Paragraph → sentence → word splitting |
| Spammy TLD | Regex match on TLD | `domain_authority = 0.10` |
| Fake/bot author | Digits or ALL CAPS in name | `author_credibility = 0.15` |
| Medical misinformation | Keyword match in content | `medical_disclaimer = 0.0` |
| Future publish date | `years_old < 0` | `recency = 0.30` (suspicious flag) |
| 404 / blocked URL | `requests` HTTP exception | Returns fallback entry with `trust_score = 0.0` |

---

## 5. Limitations & Future Work

- **Rate limiting:** No delays between requests. In production, add `time.sleep(1–2s)` and respect `robots.txt`.
- **Dynamic pages:** Sites using heavy JavaScript (React/Next.js SPAs) require Playwright or Selenium for full rendering.
- **Author verification:** The current cross-check is a simple whitelist. A more robust system could query Wikidata or ORCID for author identity verification.
- **Citation counts:** The elink API only counts citations indexed in PubMed/PMC, underestimating total scholarly impact.
- **Language support:** Topic taxonomy is English-only. Multi-language support would require translated keyword sets.
