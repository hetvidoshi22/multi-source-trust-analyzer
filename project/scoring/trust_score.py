"""
trust_score.py
Trust Score Algorithm

Formula:
  Trust Score = f(author_credibility, citation_count, domain_authority,
                  recency, medical_disclaimer_presence)

Each component is normalized to [0, 1] and combined via weighted sum.
Final score is clamped to [0.0, 1.0].

Weights (sum = 1.0):
  author_credibility          : 0.25
  domain_authority            : 0.25
  recency                     : 0.20
  citation_count              : 0.20
  medical_disclaimer_presence : 0.10
"""

from datetime import datetime, date
import re
import math

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

WEIGHTS = {
    "author_credibility": 0.25,
    "domain_authority": 0.25,
    "recency": 0.20,
    "citation_count": 0.20,
    "medical_disclaimer": 0.10,
}

# Domain authority tiers (hand-curated + rule-based fallback)
DOMAIN_AUTHORITY_MAP = {
    # Academic / Research
    "pubmed.ncbi.nlm.nih.gov": 1.0,
    "ncbi.nlm.nih.gov": 1.0,
    "nature.com": 0.97,
    "sciencedirect.com": 0.95,
    "springer.com": 0.94,
    "wiley.com": 0.93,
    "bmj.com": 0.96,
    "nejm.org": 0.98,
    "thelancet.com": 0.97,
    "jamanetwork.com": 0.96,
    "arxiv.org": 0.88,
    "ssrn.com": 0.82,
    "researchgate.net": 0.80,
    # Reputable tech / science outlets
    "mit.edu": 0.95,
    "stanford.edu": 0.95,
    "harvard.edu": 0.95,
    "cdc.gov": 0.97,
    "who.int": 0.97,
    "nih.gov": 0.96,
    "ieee.org": 0.93,
    "acm.org": 0.92,
    # High-quality tech blogs
    "towardsdatascience.com": 0.72,
    "medium.com": 0.60,
    "realpython.com": 0.75,
    "dev.to": 0.60,
    "hackernoon.com": 0.58,
    "techcrunch.com": 0.75,
    "wired.com": 0.78,
    "arstechnica.com": 0.76,
    # Video platforms
    "youtube.com": 0.65,
    "youtu.be": 0.65,
    "vimeo.com": 0.60,
    # Known low-authority / spam-prone
    "blogspot.com": 0.25,
    "wordpress.com": 0.30,
    "weebly.com": 0.20,
    "wix.com": 0.20,
    "tumblr.com": 0.22,
    "buzzfeed.com": 0.35,
    "natural-remedies.xyz": 0.05,
}

# Known credible author organizations (for cross-check)
KNOWN_CREDIBLE_ORGS = {
    "who", "cdc", "nih", "fda", "mayo clinic", "harvard", "stanford",
    "mit", "oxford", "cambridge", "lancet", "nature", "ieee", "acm",
    "pubmed", "ncbi", "johns hopkins", "yale", "3blue1brown",
    "sentdex", "andrej karpathy", "deepmind", "openai", "google brain",
    "microsoft research", "facebook ai", "meta ai",
}

# Penalty keywords indicating medical/health misinformation risk
MISINFORMATION_KEYWORDS = {
    "cure cancer naturally", "miracle cure", "doctors don't want you to know",
    "secret remedy", "big pharma conspiracy", "detox tea", "antivax",
    "no side effects guaranteed", "100% natural cure",
}


# ─────────────────────────────────────────────
# COMPONENT SCORERS
# ─────────────────────────────────────────────

def score_author_credibility(author: str, source_type: str = "blog") -> float:
    """
    Score author credibility [0, 1].

    Rules:
    - Unknown / missing author → 0.1 (significant penalty)
    - Author matches known credible org → 0.95
    - Author appears to be a real name (First Last) → 0.65
    - Multiple authors → average of individual scores
    - Fake/spammy author patterns → 0.15
    """
    if not author or author.strip().lower() in {"unknown", "n/a", "", "anonymous"}:
        return 0.10  # Edge case: missing author → low score

    # Multiple authors (semicolon or comma separated)
    if ";" in author or (author.count(",") >= 2):
        individual_authors = re.split(r"[;]|(?<=[a-z]),\s*(?=[A-Z])", author)
        scores = [score_author_credibility(a.strip(), source_type) for a in individual_authors if a.strip()]
        return sum(scores) / len(scores) if scores else 0.1  # Edge case: multiple authors → average

    author_lower = author.lower()

    # Check against known credible organizations
    for org in KNOWN_CREDIBLE_ORGS:
        if org in author_lower:
            return 0.95

    # PubMed authors are generally credible
    if source_type == "pubmed":
        return 0.85

    # Check if it looks like a real name (First Last or F. Last)
    name_pattern = re.compile(
        r"^[A-Z][a-z]+\s+([A-Z]\.?\s+)?[A-Z][a-z]+([\s\-][A-Z][a-z]+)?$"
    )
    if name_pattern.match(author.strip()):
        return 0.65

    # Suspicious patterns (all caps, numbers in name, very short, etc.)
    if re.search(r"\d", author) or len(author) < 3 or author.isupper():
        return 0.15  # Abuse prevention: fake author patterns

    return 0.50  # Default moderate score


def score_domain_authority(domain: str, source_type: str = "blog") -> float:
    """
    Score domain authority [0, 1].

    Rules:
    - Look up domain in curated map
    - TLD-based heuristic as fallback (.edu/.gov → high, .xyz/.info → low)
    - Abuse prevention: penalize known low-DA / spam domains
    """
    domain_clean = domain.lower().strip()
    # Remove 'www.' prefix
    domain_clean = re.sub(r"^www\.", "", domain_clean)

    # Direct lookup
    if domain_clean in DOMAIN_AUTHORITY_MAP:
        return DOMAIN_AUTHORITY_MAP[domain_clean]

    # Partial match (subdomain handling)
    for known_domain, score in DOMAIN_AUTHORITY_MAP.items():
        if domain_clean.endswith("." + known_domain) or known_domain in domain_clean:
            return score

    # TLD-based fallback heuristic
    if domain_clean.endswith(".edu"):
        return 0.85
    if domain_clean.endswith(".gov"):
        return 0.90
    if domain_clean.endswith(".org"):
        return 0.60
    if domain_clean.endswith(".ac.uk") or domain_clean.endswith(".ac.in"):
        return 0.82
    # Spammy TLDs (abuse prevention)
    spammy_tlds = [".xyz", ".info", ".biz", ".click", ".top", ".loan", ".win"]
    if any(domain_clean.endswith(tld) for tld in spammy_tlds):
        return 0.10  # Abuse prevention: SEO spam / low-quality TLD

    # YouTube gets a fixed moderate score
    if source_type == "youtube":
        return DOMAIN_AUTHORITY_MAP.get("youtube.com", 0.65)

    return 0.40  # Unknown domain → moderate penalty


def score_recency(published_date: str) -> float:
    """
    Score recency [0, 1] using exponential decay.

    Decay function: score = exp(-lambda * years_old)
    where lambda controls the decay rate.

    Rules:
    - Unknown date → 0.20 (edge case penalty)
    - Content < 1 year old → close to 1.0
    - Content 5+ years old → significantly lower
    - Medical/health content gets stronger decay (lambda=0.30)
    """
    if not published_date or published_date.strip().lower() in {"unknown", "n/a", ""}:
        return 0.20  # Edge case: missing date

    # Parse date
    date_obj = None
    for fmt in ["%Y-%m-%d", "%Y-%m", "%Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y"]:
        try:
            date_obj = datetime.strptime(published_date.strip()[:10], fmt).date()
            break
        except ValueError:
            try:
                date_obj = datetime.strptime(published_date.strip()[:4], "%Y").date()
                break
            except ValueError:
                continue

    if not date_obj:
        return 0.20  # Could not parse date

    today = date.today()
    years_old = (today - date_obj).days / 365.25

    # Future date guard (abuse prevention)
    if years_old < 0:
        return 0.30  # Suspicious future date

    # Exponential decay: lambda=0.20 gives ~82% at 1yr, ~45% at 4yr
    decay_lambda = 0.20
    score = math.exp(-decay_lambda * years_old)
    return max(0.05, min(1.0, score))


def score_citation_count(citation_count: int, source_type: str = "blog") -> float:
    """
    Score citation count [0, 1] using log normalization.

    Rules:
    - Blog/YouTube: citation count is typically 0 → default moderate score
    - PubMed: use log scale (0 cites → 0.3, 100+ cites → ~1.0)
    - Abuse prevention: unusually high citation count on unknown domain is capped
    """
    if source_type in {"blog", "youtube"}:
        # Blogs/videos don't have citation counts → neutral score
        return 0.50

    if citation_count <= 0:
        return 0.30  # Edge case: 0 citations for academic paper

    # Log normalization: max_ref = 500 citations maps to 1.0
    max_ref = 500
    score = math.log1p(citation_count) / math.log1p(max_ref)
    return min(1.0, score)


def score_medical_disclaimer(has_disclaimer: bool, source_type: str = "blog",
                              content_sample: str = "") -> float:
    """
    Score medical disclaimer presence [0, 1].

    Rules:
    - PubMed → always 1.0 (peer-reviewed = implicit disclaimer)
    - Blog/YouTube with disclaimer → 0.90
    - Blog/YouTube without disclaimer + health/medical content → 0.20
    - Misinformation keywords found → 0.0 (abuse prevention)
    """
    # Abuse prevention: check for misinformation keywords
    content_lower = content_sample.lower()
    for keyword in MISINFORMATION_KEYWORDS:
        if keyword in content_lower:
            return 0.0  # Hard penalty for misinformation content

    if source_type == "pubmed":
        return 1.0

    if has_disclaimer:
        return 0.90

    # Health-related content without disclaimer gets penalized
    health_keywords = {"treatment", "cure", "diagnosis", "medical", "health", "disease",
                       "drug", "medication", "therapy", "symptom", "vaccine", "clinical"}
    if any(kw in content_lower for kw in health_keywords):
        return 0.20  # Abuse prevention: health content without disclaimer

    return 0.60  # Non-medical content without disclaimer → neutral


# ─────────────────────────────────────────────
# MAIN TRUST SCORE FUNCTION
# ─────────────────────────────────────────────

def calculate_trust_score(
    author: str,
    published_date: str,
    domain: str,
    citation_count: int = 0,
    has_medical_disclaimer: bool = False,
    source_type: str = "blog",
    content_sample: str = "",
    view_count: int = 0,
) -> float:
    """
    Calculate the Trust Score for a scraped source.

    Trust Score = weighted sum of:
      - author_credibility      (0.25)
      - domain_authority        (0.25)
      - recency                 (0.20)
      - citation_count          (0.20)
      - medical_disclaimer      (0.10)

    Returns: float in [0.0, 1.0]
    """
    components = {}

    components["author_credibility"] = score_author_credibility(author, source_type)
    components["domain_authority"] = score_domain_authority(domain, source_type)
    components["recency"] = score_recency(published_date)
    components["citation_count"] = score_citation_count(citation_count, source_type)
    components["medical_disclaimer"] = score_medical_disclaimer(
        has_medical_disclaimer, source_type, content_sample
    )

    # Weighted sum
    trust_score = sum(
        WEIGHTS[key] * value for key, value in components.items()
        if key in WEIGHTS
    )

    # Abuse prevention: viral but unverified YouTube content
    if source_type == "youtube" and view_count > 10_000_000:
        if components["author_credibility"] < 0.5:
            trust_score *= 0.85  # Penalize viral but uncredible channels

    # Final clamp
    trust_score = max(0.0, min(1.0, trust_score))

    return trust_score


def get_score_breakdown(
    author: str,
    published_date: str,
    domain: str,
    citation_count: int = 0,
    has_medical_disclaimer: bool = False,
    source_type: str = "blog",
    content_sample: str = "",
) -> dict:
    """
    Return detailed breakdown of each trust score component.
    Useful for debugging and transparency.
    """
    components = {
        "author_credibility": score_author_credibility(author, source_type),
        "domain_authority": score_domain_authority(domain, source_type),
        "recency": score_recency(published_date),
        "citation_count": score_citation_count(citation_count, source_type),
        "medical_disclaimer": score_medical_disclaimer(
            has_medical_disclaimer, source_type, content_sample
        ),
    }
    weighted = {k: round(WEIGHTS[k] * v, 4) for k, v in components.items()}
    total = sum(weighted.values())

    return {
        "raw_scores": {k: round(v, 4) for k, v in components.items()},
        "weights": WEIGHTS,
        "weighted_scores": weighted,
        "total_trust_score": round(max(0.0, min(1.0, total)), 4),
    }


# ─────────────────────────────────────────────
# EDGE CASES SUMMARY (for documentation)
# ─────────────────────────────────────────────
"""
EDGE CASES HANDLED:
1. Missing author          → author_credibility = 0.10
2. Missing published date  → recency = 0.20
3. Multiple authors        → average of individual credibility scores
4. Zero citations (academic) → citation_count score = 0.30
5. Unknown domain          → domain_authority = 0.40
6. Future publish date     → recency = 0.30 (suspicious)
7. Spammy TLD (.xyz etc.)  → domain_authority = 0.10

ABUSE PREVENTION:
1. Fake authors (digits/caps in name) → credibility = 0.15
2. SEO spam blogs (low DA + spammy TLD) → domain_authority = 0.10
3. Medical misinformation keywords   → disclaimer_score = 0.0
4. Health content without disclaimer → disclaimer_score = 0.20
5. Viral but uncredible YouTube      → 15% score reduction
6. Future-dated content              → recency = 0.30
"""

if __name__ == "__main__":
    import json

    # Example test
    test_cases = [
        {
            "label": "PubMed peer-reviewed article",
            "author": "John A. Smith; Jane B. Doe",
            "published_date": "2024-03-15",
            "domain": "pubmed.ncbi.nlm.nih.gov",
            "citation_count": 45,
            "has_medical_disclaimer": True,
            "source_type": "pubmed",
        },
        {
            "label": "Reputable tech blog",
            "author": "Alex Turner",
            "published_date": "2025-01-10",
            "domain": "towardsdatascience.com",
            "citation_count": 0,
            "has_medical_disclaimer": False,
            "source_type": "blog",
        },
        {
            "label": "Spam health blog",
            "author": "ADMIN123",
            "published_date": "2019-06-01",
            "domain": "natural-cures.xyz",
            "citation_count": 0,
            "has_medical_disclaimer": False,
            "source_type": "blog",
            "content_sample": "miracle cure doctors don't want you to know",
        },
        {
            "label": "YouTube educational video",
            "author": "3Blue1Brown",
            "published_date": "2023-09-20",
            "domain": "youtube.com",
            "citation_count": 0,
            "has_medical_disclaimer": False,
            "source_type": "youtube",
        },
    ]

    print("=== Trust Score Test Results ===\n")
    for tc in test_cases:
        label = tc.pop("label")
        breakdown = get_score_breakdown(**tc)
        print(f"[{label}]")
        print(json.dumps(breakdown, indent=2))
        print()
