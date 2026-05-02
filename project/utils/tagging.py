"""
tagging.py
Automatic topic tagging using TF-IDF style keyword extraction
plus a curated keyword taxonomy.
"""

import re
from collections import Counter
from typing import List

# ─────────────────────────────────────────────
# TOPIC TAXONOMY
# Keywords mapped to topic labels
# ─────────────────────────────────────────────
TOPIC_TAXONOMY = {
    # AI / ML
    "AI": [
        "artificial intelligence", "ai", "machine learning", "deep learning",
        "neural network", "neural networks", "nlp", "natural language processing",
        "computer vision", "generative ai", "large language model", "llm",
        "reinforcement learning", "transformer", "gpt", "bert", "diffusion model",
        "openai", "chatgpt", "gemini", "claude",
    ],
    "Machine Learning": [
        "machine learning", "ml", "supervised learning", "unsupervised learning",
        "regression", "classification", "clustering", "random forest",
        "gradient boosting", "xgboost", "feature engineering", "training data",
        "model evaluation", "cross validation", "overfitting", "underfitting",
    ],
    "Data Science": [
        "data science", "data analysis", "data visualization", "pandas",
        "numpy", "matplotlib", "seaborn", "exploratory data analysis", "eda",
        "statistics", "hypothesis testing", "data cleaning", "data wrangling",
    ],
    "Web Scraping": [
        "web scraping", "scraper", "scraping", "beautifulsoup", "selenium",
        "playwright", "requests", "html", "css selector", "xpath", "crawling",
        "web crawler", "data extraction", "scrapy",
    ],
    "Python": [
        "python", "pip", "virtualenv", "flask", "django", "fastapi",
        "jupyter", "notebook", "programming", "script", "library", "framework",
    ],
    "Healthcare": [
        "healthcare", "health", "medical", "medicine", "clinical", "patient",
        "hospital", "treatment", "diagnosis", "disease", "drug", "therapy",
        "biomedical", "pharmaceutical", "radiology", "pathology",
    ],
    "Genomics": [
        "genomics", "genome", "dna", "rna", "gene", "genetics", "sequencing",
        "bioinformatics", "mutation", "protein", "crispr",
    ],
    "COVID-19": [
        "covid", "covid-19", "coronavirus", "sars-cov-2", "pandemic",
        "vaccination", "vaccine", "pfizer", "moderna", "bnt162b2",
    ],
    "Cybersecurity": [
        "cybersecurity", "hacking", "malware", "ransomware",
        "firewall", "vulnerability", "zero-day", "phishing",
        "sql injection", "xss", "ddos", "intrusion detection",
        "penetration testing", "cyber attack", "data breach",
    ],
    "Cloud Computing": [
        "cloud", "aws", "azure", "gcp", "google cloud", "kubernetes",
        "docker", "serverless", "microservices", "devops", "ci/cd",
    ],
    "Research": [
        "study", "research", "paper", "journal", "pubmed", "abstract",
        "literature review", "meta-analysis", "clinical trial", "rct",
        "systematic review", "cohort", "p-value", "statistical significance",
    ],
    "Education": [
        "education", "learning", "course", "tutorial", "teach", "student",
        "university", "lecture", "training", "certification",
    ],
    "YouTube / Video": [
        "youtube", "video", "channel", "subscribe", "watch", "playlist",
        "streaming", "vlog", "podcast",
    ],
    "Finance": [
        "finance", "investment", "stock", "market", "trading", "bitcoin",
        "cryptocurrency", "blockchain", "economy", "inflation",
    ],
    "Environment": [
        "climate", "environment", "sustainability", "renewable energy",
        "carbon", "greenhouse gas", "global warming",
    ],
}

# ─────────────────────────────────────────────
# STOPWORDS (common English words to ignore)
# ─────────────────────────────────────────────
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "that", "this", "was",
    "are", "be", "been", "has", "have", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "not", "no", "so",
    "if", "then", "than", "as", "up", "out", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "each",
    "more", "most", "other", "some", "such", "only", "own", "same",
    "also", "very", "just", "because", "while", "although", "however",
    "therefore", "thus", "hence", "their", "they", "them", "we", "our",
    "you", "your", "he", "she", "his", "her", "i", "my", "me", "us",
    "which", "who", "what", "when", "where", "how", "all", "any",
    "both", "few", "more", "most", "much", "many", "these", "those",
    "using", "used", "use", "make", "made", "new", "one", "two",
    "first", "second", "based", "well", "see", "get", "like", "time",
    # URL fragments / web junk (prevents 'https', 'www', 'com' from becoming tags)
    "http", "https", "www", "com", "org", "net", "html", "href",
    "url", "link", "click", "page", "site", "website",
    # Common non-informative words found in scraped content
    "also", "still", "even", "much", "really", "things", "thing",
    "going", "right", "need", "want", "know", "look", "take",
    "come", "give", "tell", "work", "call", "show", "help",
    "different", "actually", "basically", "something", "written",
    # Transcript noise words
    "there", "here", "then", "just", "that", "this", "with", "from",
    "have", "been", "were", "they", "them", "their", "would", "could",
    "should", "think", "kind", "sort", "ways", "back", "over", "each",
    "every", "same", "said", "does", "done", "goes", "mean", "means",
    "around", "those", "these", "between", "through", "about", "after",
    "before", "because", "being", "doing", "having", "making", "getting",
    "going", "coming", "putting", "taking", "finding", "number", "numbers",
}


# ─────────────────────────────────────────────
# TAGGING FUNCTIONS
# ─────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, strip HTML tags, URLs, and remove special characters."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove full URLs (http/https/www...)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    # Lowercase
    text = text.lower()
    # Keep letters, digits, spaces, hyphens
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def taxonomy_match(text_lower: str) -> List[str]:
    """
    Match text against topic taxonomy.
    Returns list of matched topic labels sorted by match strength.
    """
    matched = {}
    for topic, keywords in TOPIC_TAXONOMY.items():
        count = 0
        for kw in keywords:
            # Count occurrences of each keyword
            occurrences = len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
            count += occurrences
        if count > 0:
            matched[topic] = count

    # Sort by match strength descending
    return [topic for topic, _ in sorted(matched.items(), key=lambda x: -x[1])]


def extract_keywords_tfidf(text: str, top_n: int = 8) -> List[str]:
    """
    Extract top keywords from text using a simplified TF-IDF approach
    (term frequency, filtered by stopwords, ranked by frequency × length).
    """
    text_norm = normalize_text(text)
    words = text_norm.split()

    # Filter stopwords and very short words
    words = [w for w in words if w not in STOPWORDS and len(w) > 3]

    # Count frequencies
    freq = Counter(words)

    # Score: frequency × log(word_length) to prefer longer, more informative words
    import math
    scored = {w: count * math.log(1 + len(w)) for w, count in freq.items()}

    # Get top N
    top_words = sorted(scored.items(), key=lambda x: -x[1])[:top_n]
    return [word for word, _ in top_words]


def auto_tag(text: str, extra_tags: List[str] = None, max_tags: int = 8) -> List[str]:
    """
    Main auto-tagging function.
    Combines taxonomy matching with keyword extraction.

    Args:
        text: Combined title + description + content sample
        extra_tags: Pre-known tags (e.g., MeSH terms from PubMed)
        max_tags: Maximum number of tags to return

    Returns:
        List of topic tag strings
    """
    if not text or not text.strip():
        return extra_tags[:max_tags] if extra_tags else []

    text_lower = normalize_text(text)

    # 1. Taxonomy-based matching (highest priority)
    taxonomy_tags = taxonomy_match(text_lower)

    # 2. Keyword extraction
    keyword_tags = extract_keywords_tfidf(text, top_n=5)

    # 3. Combine: taxonomy tags first, then keywords, then extra tags
    combined = []
    seen = set()

    for tag in taxonomy_tags:
        if tag.lower() not in seen:
            combined.append(tag)
            seen.add(tag.lower())

    for kw in keyword_tags:
        if kw.lower() not in seen and len(combined) < max_tags:
            combined.append(kw)
            seen.add(kw.lower())

    if extra_tags:
        for tag in extra_tags:
            if tag.lower() not in seen and len(combined) < max_tags:
                combined.append(tag)
                seen.add(tag.lower())

    # 4. Post-filter: remove generic/uninformative tags
    generic_tags = {
        "humans", "animals", "male", "female", "adult", "adults",
        "there", "light-sensitive", "making", "introduction",
    }
    combined = [t for t in combined if t.lower() not in generic_tags]

    return combined[:max_tags]


if __name__ == "__main__":
    # Test the tagger
    sample_text = """
    Machine learning and artificial intelligence are transforming healthcare.
    Deep learning models trained on large datasets can detect diseases from 
    medical images with high accuracy. This tutorial uses Python, TensorFlow,
    and scikit-learn to build a neural network classifier.
    """
    tags = auto_tag(sample_text)
    print("Auto-generated tags:", tags)

    # PubMed article
    pubmed_text = "A systematic review of COVID-19 treatment protocols using randomized clinical trials."
    pubmed_tags = auto_tag(pubmed_text, extra_tags=["COVID-19", "Clinical Trial", "Systematic Review"])
    print("PubMed tags:", pubmed_tags)
