"""
pubmed_scraper.py
Scrapes 1 PubMed article and returns structured JSON data.
Uses PubMed E-utilities API (no API key required for basic access).
"""

import requests
import xml.etree.ElementTree as ET
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from project.utils.tagging import auto_tag
from project.utils.chunking import chunk_text
from project.scoring.trust_score import calculate_trust_score, get_score_breakdown

# PubMed article PMIDs to scrape
PUBMED_PMIDS = ["37349072"]  # Example: a recent AI in Healthcare article

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AcademicBot/1.0; mailto:student@example.edu)"
    )
}


def fetch_pubmed_xml(pmid: str) -> str:
    """Fetch PubMed article XML via E-utilities efetch."""
    url = (
        f"{PUBMED_BASE}efetch.fcgi"
        f"?db=pubmed&id={pmid}&rettype=xml&retmode=xml"
    )
    print(f"  Fetching PubMed article PMID: {pmid}")
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.text


def parse_pubmed_xml(xml_text: str, pmid: str) -> dict:
    """Parse PubMed XML and extract structured fields."""
    root = ET.fromstring(xml_text)
    article = root.find(".//PubmedArticle")
    if not article:
        raise ValueError("No PubmedArticle found in XML")

    medline = article.find("MedlineCitation")
    if not medline:
        raise ValueError("No MedlineCitation found")

    # --- Title ---
    title_el = medline.find(".//ArticleTitle")
    title = title_el.text or "Unknown Title" if title_el is not None else "Unknown Title"
    # Clean XML tags (sometimes title has italic/bold sub-elements)
    if title_el is not None:
        title = "".join(title_el.itertext()).strip()

    # --- Authors ---
    authors = []
    for author in medline.findall(".//Author"):
        last = author.find("LastName")
        first = author.find("ForeName")
        collective = author.find("CollectiveName")
        if collective is not None:
            authors.append(collective.text or "")
        elif last is not None:
            name = last.text or ""
            if first is not None:
                name = f"{first.text} {name}"
            authors.append(name.strip())

    author_str = "; ".join(authors) if authors else "Unknown"

    # --- Journal ---
    journal_el = medline.find(".//Journal/Title")
    journal = journal_el.text if journal_el is not None else "Unknown Journal"

    # --- Publication Date ---
    pub_year = "Unknown"
    pub_month = ""
    pub_day = ""
    pub_date = medline.find(".//PubDate")
    if pub_date is not None:
        year_el = pub_date.find("Year")
        month_el = pub_date.find("Month")
        day_el = pub_date.find("Day")
        medline_date = pub_date.find("MedlineDate")
        if year_el is not None:
            pub_year = year_el.text or "Unknown"
            pub_month = month_el.text if month_el is not None else "01"
            pub_day = day_el.text if day_el is not None else "01"
            # Normalize month
            month_map = {
                "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
            }
            if pub_month in month_map:
                pub_month = month_map[pub_month]
            try:
                pub_date_str = f"{pub_year}-{pub_month.zfill(2)}-{pub_day.zfill(2)}"
            except Exception:
                pub_date_str = pub_year
        elif medline_date is not None:
            pub_date_str = medline_date.text or "Unknown"
        else:
            pub_date_str = pub_year
    else:
        pub_date_str = "Unknown"

    # --- Abstract ---
    abstract_parts = []
    for abs_text in medline.findall(".//AbstractText"):
        label = abs_text.get("Label", "")
        text = "".join(abs_text.itertext()).strip()
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    abstract = "\n\n".join(abstract_parts)

    # --- MeSH terms / Keywords as tags ---
    mesh_terms = []
    for mesh in medline.findall(".//MeshHeading/DescriptorName"):
        if mesh.text:
            mesh_terms.append(mesh.text)

    keywords = []
    for kw in medline.findall(".//Keyword"):
        if kw.text:
            keywords.append(kw.text)

    # Combine for auto-tagging
    tag_input = title + " " + abstract[:500] + " " + " ".join(mesh_terms[:10])
    topic_tags = auto_tag(tag_input, extra_tags=mesh_terms[:5])

    # --- Citation count (approximate via PubMed Central) ---
    citation_count = _get_citation_count(pmid)

    # --- Chunking ---
    content_text = f"Title: {title}\n\nAuthors: {author_str}\n\nJournal: {journal}\n\nAbstract:\n{abstract}"
    chunks = chunk_text(content_text, chunk_size=200)

    # --- Trust score ---
    trust_score = calculate_trust_score(
        author=author_str,
        published_date=pub_date_str,
        domain="pubmed.ncbi.nlm.nih.gov",
        citation_count=citation_count,
        has_medical_disclaimer=True,  # PubMed articles are peer-reviewed
        source_type="pubmed",
    )

    return {
        "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "source_type": "pubmed",
        "pmid": pmid,
        "title": title,
        "author": author_str,
        "journal": journal,
        "published_date": pub_date_str,
        "language": "en",
        "region": "International",
        "mesh_terms": mesh_terms[:10],
        "topic_tags": topic_tags,
        "trust_score": round(trust_score, 3),
        "trust_score_breakdown": {k: round(v, 3) for k, v in get_score_breakdown(
            author=author_str, published_date=pub_date_str,
            domain="pubmed.ncbi.nlm.nih.gov", citation_count=citation_count,
            has_medical_disclaimer=True, source_type="pubmed"
        )["raw_scores"].items()},
        "citation_count": citation_count,
        "content_chunks": chunks,
    }


def _get_citation_count(pmid: str) -> int:
    """
    Attempt to get citation count via PubMed elink API.
    Returns 0 if unavailable.
    """
    try:
        url = (
            f"{PUBMED_BASE}elink.fcgi"
            f"?dbfrom=pubmed&linkname=pubmed_pubmed_citedin&id={pmid}&retmode=json"
        )
        response = requests.get(url, headers=HEADERS, timeout=10)
        data = response.json()
        links = data.get("linksets", [{}])[0].get("linksetdbs", [])
        for link in links:
            if link.get("linkname") == "pubmed_pubmed_citedin":
                return len(link.get("links", []))
    except Exception:
        pass
    return 0


def scrape_pubmed(pmid: str) -> dict:
    """Scrape a single PubMed article by PMID."""
    try:
        xml_text = fetch_pubmed_xml(pmid)
        return parse_pubmed_xml(xml_text, pmid)
    except Exception as e:
        print(f"  [ERROR] Failed to scrape PMID {pmid}: {e}")
        return {
            "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "source_type": "pubmed",
            "pmid": pmid,
            "title": "Unavailable",
            "author": "Unknown",
            "journal": "Unknown",
            "published_date": "Unknown",
            "language": "en",
            "region": "International",
            "mesh_terms": [],
            "topic_tags": [],
            "trust_score": 0.0,
            "citation_count": 0,
            "content_chunks": [f"[Scraping failed: {e}]"],
        }


def scrape_all_pubmed() -> list:
    """Scrape all configured PubMed PMIDs."""
    return [scrape_pubmed(pmid) for pmid in PUBMED_PMIDS]


if __name__ == "__main__":
    import json
    print("Starting PubMed scraper...")
    articles = scrape_all_pubmed()
    os.makedirs("output/scraped_data", exist_ok=True)
    with open("output/scraped_data/pubmed.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(articles)} PubMed entries.")
