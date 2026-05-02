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


def extract_keywords_tfidf(text: str, top_n: int = 10) -> List[str]:
    """
    Extract top keywords using an intelligent, phrase-aware scoring system.
    Identifies specific technical concepts (e.g., 'HTTP Requests') and
    meaningful bigrams to provide highly descriptive tags.
    """
    import math
    text_norm = normalize_text(text)
    
    # 1. Broad phrase matching (Highly descriptive technical terms)
    PHRASE_PATTERNS = [
        "http requests", "html parsing", "data extraction", "data scraping",
        "web scraping", "content chunking", "topic tagging", "trust score",
        "neural network", "deep learning", "machine learning", "natural language",
        "computer vision", "reinforcement learning", "gradient descent",
        "api integration", "rest api", "structured data", "clinical trial",
        "emergency medicine", "medical education", "peer reviewed",
        "virtual environment", "open source", "beautiful soup", "beautifulsoup",
        "regular expression", "data visualization", "exploratory data analysis",
        "predictive modeling", "random forest", "logistic regression",
    ]
    phrase_tags, seen_phrases = [], set()
    for phrase in PHRASE_PATTERNS:
        if phrase in text_norm:
            tag = " ".join(w.capitalize() for w in phrase.split())
            if tag.lower() not in seen_phrases:
                phrase_tags.append(tag)
                seen_phrases.add(tag.lower())

    # 2. Bigram extraction (Significant word pairs)
    raw_words = text_norm.split()
    bigrams = []
    for i in range(len(raw_words) - 1):
        w1, w2 = raw_words[i], raw_words[i+1]
        if (w1 not in STOPWORDS and w2 not in STOPWORDS and 
            len(w1) > 3 and len(w2) > 3):
            bigrams.append("{} {}".format(w1, w2))
    
    bigram_counts = Counter(bigrams)
    scored_bigrams = []
    for bg, count in bigram_counts.items():
        if count >= 2:
            scored_bigrams.append((" ".join(w.capitalize() for w in bg.split()), count * 1.5))
    
    # 3. Unigram scoring (TF-IDF style: freq * log(length))
    filtered = [w for w in raw_words if w not in STOPWORDS and len(w) > 3]
    unigram_counts = Counter(filtered)
    scored_unigrams = []
    for w, count in unigram_counts.items():
        score = count * math.log(1 + len(w))
        scored_unigrams.append((w.capitalize() if len(w) > 3 else w.upper(), score))

    # 4. Merge results with priority: Phrases > Bigrams > Unigrams
    results, final_seen = [], set()
    
    for pt in phrase_tags:
        if pt.lower() not in final_seen:
            results.append(pt)
            final_seen.add(pt.lower())

    sorted_bigrams = sorted(scored_bigrams, key=lambda x: -x[1])
    for bg, _ in sorted_bigrams:
        if bg.lower() not in final_seen:
            results.append(bg)
            final_seen.add(bg.lower())
        if len(results) >= top_n: break

    sorted_unigrams = sorted(scored_unigrams, key=lambda x: -x[1])
    for ug, _ in sorted_unigrams:
        if ug.lower() not in final_seen:
            results.append(ug)
            final_seen.add(ug.lower())
        if len(results) >= top_n: break

    return results[:top_n]



def auto_tag(text: str, extra_tags: List[str] = None, max_tags: int = 8) -> List[str]:
    """
    Main auto-tagging function.
    Prioritizes specific technical phrases then falls back to taxonomy.
    """
    if not text or not text.strip():
        return extra_tags[:max_tags] if extra_tags else []

    text_lower = normalize_text(text)

    # 1. Technical keyword/phrase extraction (High specificity)
    keyword_tags = extract_keywords_tfidf(text, top_n=6)

    # 2. Taxonomy-based matching (Broad categorization)
    taxonomy_tags = taxonomy_match(text_lower)

    # 3. Combine: Keywords > Taxonomy > Extra tags
    combined, seen = [], set()

    # Keywords (Intelligent phrases)
    for kw in keyword_tags:
        if kw.lower() not in seen:
            combined.append(kw)
            seen.add(kw.lower())
        if len(combined) >= 6: break # Reserve space for taxonomy

    # Taxonomy (Broad topics) - Limit to top 3
    for tag in taxonomy_tags[:3]:
        if tag.lower() not in seen:
            combined.append(tag)
            seen.add(tag.lower())

    # Extra tags (e.g., MeSH)
    if extra_tags:
        for et in extra_tags:
            if et.lower() not in seen and len(combined) < max_tags:
                combined.append(et)
                seen.add(et.lower())

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
