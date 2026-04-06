"""
enrichment/config.py
════════════════════
All configuration for the Layer 2 enrichment pipeline.

Same pattern as pipeline/config.py — every setting lives here.
No other file in the enrichment package hardcodes values.

HOW TO USE:
    from enrichment.config import ENRICH_BATCH_SIZE, CLUSTER_WINDOW_HOURS
"""

import os
from dotenv import load_dotenv

load_dotenv()          # tries .env file (standard)
load_dotenv(".env.local", override=False)  # fallback for local dev


# ─────────────────────────────────────────────────────────────────────────────
# SUPABASE — same credentials as Layer 1
# ─────────────────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE NAMES
# ─────────────────────────────────────────────────────────────────────────────
TABLE_ARTICLES  = "articles"
TABLE_ENTITIES  = "article_entities"
TABLE_CLUSTERS  = "article_clusters"


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
#
# The enrichment runner processes articles in batches to stay within Supabase
# API limits and avoid memory issues with large NLP models.
#
# ENRICH_BATCH_SIZE:
#   How many articles to fetch and process per run.
#   500 allows ~4,000 articles/day enriched (8 runs × 500). Lower if on free tier.
#   NLP is CPU-bound so concurrency doesn't help here — sequential is fine.
#
# ENRICH_MIN_WORD_COUNT:
#   Articles below this word count are marked enriched_at but skipped for all
#   expensive steps (NER, keywords, clustering, image download).
#   Typical Indian news snippet / wire brief = 30–60 words.
#   Articles with < 50 words carry almost no enrichment signal.
# ─────────────────────────────────────────────────────────────────────────────
# Raised from 500 → 1000 after Supabase Pro upgrade.
ENRICH_BATCH_SIZE     = int(os.getenv("ENRICH_BATCH_SIZE",     "1000"))
ENRICH_MIN_WORD_COUNT = int(os.getenv("ENRICH_MIN_WORD_COUNT", "50"))

# Only enrich articles published within this many hours.
# Articles older than this are stale — enriching them wastes compute and
# they will never surface in a freshness-ranked news feed.
# Set to 0 to disable the gate (enrich everything regardless of age).
ENRICH_MAX_AGE_HOURS  = int(os.getenv("ENRICH_MAX_AGE_HOURS",  "48"))

# Languages to fully enrich (NER, clustering, sentiment, keywords, classification).
# Articles in other languages get text_stats + language_detected only, then marked done.
# Set to empty set to enrich all languages.
# "en" covers en, en-in, en-us etc — the gate checks startswith in runner.
_supported = os.getenv("ENRICH_SUPPORTED_LANGUAGES", "en,hi")
ENRICH_SUPPORTED_LANGUAGES: set[str] = (
    {lang.strip() for lang in _supported.split(",") if lang.strip()}
    if _supported.strip() else set()
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP: TEXT STATS
# ─────────────────────────────────────────────────────────────────────────────
WORDS_PER_MINUTE = 200   # average adult reading speed used for reading_time_mins


# ─────────────────────────────────────────────────────────────────────────────
# STEP: LANGUAGE DETECTION
#
# langdetect needs a minimum amount of text to be accurate.
# Short snippets (< 20 chars) produce unreliable results — skip them.
# ─────────────────────────────────────────────────────────────────────────────
LANG_DETECT_MIN_CHARS = 20


# ─────────────────────────────────────────────────────────────────────────────
# STEP: NAMED ENTITY RECOGNITION
#
# SPACY_MODEL:
#   'en_core_web_sm'  — 12 MB, fast, good for English news (default)
#   'en_core_web_md'  — 43 MB, better accuracy, worth it on paid infra
#   'en_core_web_lg'  — 741 MB, best accuracy, use only if you have fast storage
#
#   Download command:  python -m spacy download en_core_web_sm
#
# NER_MIN_SALIENCE:
#   Entities below this salience score are not stored.
#   Keeps the article_entities table lean — filters out
#   incidental mentions like "he said", "the company", etc.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY CLASSIFICATION — mDeBERTa zero-shot NLI
#
# Model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
#   - State-of-the-art multilingual NLI model
#   - Understands all Indian regional languages natively
#   - Zero-shot: no training data needed, just define category labels
#   - ~560 MB download, cached in ~/.cache/huggingface/hub after first run
#
# CLASSIFY_CONFIDENCE_THRESHOLD:
#   Minimum score from the model to accept a category.
#   Below this → return 'general'. Prevents low-confidence mislabelling.
# ─────────────────────────────────────────────────────────────────────────────
DEBERTA_MODEL               = os.getenv("DEBERTA_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
# Raised from 0.15 → 0.35. With 12 NLI candidate labels, random baseline
# per label ≈ 0.083. 0.15 was barely above chance, causing mass misclassification
# into "world" (confirmed: 37% of articles classified as "world" in production).
CLASSIFY_CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFY_CONFIDENCE_THRESHOLD", "0.35"))


# ─────────────────────────────────────────────────────────────────────────────
# STORY CLUSTERING — LaBSE sentence embeddings
#
# Model: sentence-transformers/LaBSE (Language-Agnostic BERT Sentence Embeddings)
#   - Google's model trained on 109 languages
#   - Explicitly supports ALL major Indian languages
#   - Produces 768-dim normalised embeddings
#   - Two articles about the same event in Hindi and English will have
#     cosine similarity > 0.85 even with zero word overlap
#   - ~471 MB download, cached after first run
#
# CLUSTER_EMBEDDING_THRESHOLD:
#   Cosine similarity above which two articles are considered the same story.
#   LaBSE scores are higher than traditional methods — 0.82 is well calibrated
#   for Indian news wire syndication (same PTI story, different outlets).
# ─────────────────────────────────────────────────────────────────────────────
LABSE_MODEL                 = os.getenv("LABSE_MODEL", "sentence-transformers/LaBSE")
CLUSTER_EMBEDDING_THRESHOLD = float(os.getenv("CLUSTER_EMBEDDING_THRESHOLD", "0.82"))


# ─────────────────────────────────────────────────────────────────────────────
# NER (spaCy) — kept for entity extraction (tagging, display, app layer)
# ─────────────────────────────────────────────────────────────────────────────
# Upgraded from en_core_web_sm (12 MB) → en_core_web_md (43 MB) after Supabase Pro upgrade.
# Requires: python -m spacy download en_core_web_md
SPACY_MODEL              = os.getenv("SPACY_MODEL",             "en_core_web_md")
SPACY_MULTILINGUAL_MODEL = os.getenv("SPACY_MULTILINGUAL_MODEL", "xx_ent_wiki_sm")
NER_MIN_SALIENCE         = 0.1
#
# Two models based on detected language:
#   en_core_web_sm  — English articles
#   xx_ent_wiki_sm  — All other languages (Hindi, Tamil, etc.)
#
# Download commands:
#   python -m spacy download en_core_web_sm
#   python -m spacy download xx_ent_wiki_sm

# Entity types we care about for Indian news
# spaCy label → stored entity_type
ENTITY_TYPE_MAP = {
    "PERSON":   "PERSON",    # politicians, celebrities, athletes
    "ORG":      "ORG",       # parties, companies, institutions
    "GPE":      "GPE",       # cities, states, countries
    "LOC":      "GPE",       # map LOC → GPE for simplicity
    "EVENT":    "EVENT",     # named events (Budget 2024, IPL, G20)
    "PRODUCT":  "PRODUCT",   # apps, platforms, products
    "LAW":      "LAW",       # named laws and acts (CAA, RTI, Article 370)
    "NORP":     "ORG",       # nationalities/political groups → treat as ORG
    "WORK_OF_ART": "PRODUCT",
}
# Types NOT in the map (DATE, TIME, MONEY, PERCENT, CARDINAL, etc.) are skipped


# ─────────────────────────────────────────────────────────────────────────────
# STEP: KEYWORD EXTRACTION (YAKE)
#
# YAKE (Yet Another Keyword Extractor) is unsupervised and language-agnostic —
# it works on Hindi and Tamil articles as well as English, which matters for
# the Indian market.
#
# MAX_KEYWORDS:    top N keywords to store per article
# KEYWORD_MAX_NGRAM: max words in a keyword phrase (3 = "Narendra Modi budget")
# KEYWORD_DEDUP_THRESHOLD: lower = more diversity between keywords (0.9 = loose)
# ─────────────────────────────────────────────────────────────────────────────
MAX_KEYWORDS             = int(os.getenv("MAX_KEYWORDS", "10"))
KEYWORD_MAX_NGRAM        = 3
KEYWORD_DEDUP_THRESHOLD  = 0.9


# ─────────────────────────────────────────────────────────────────────────────
# STEP: ARTICLE CATEGORY CLASSIFICATION
#
# Rule-based keyword classifier for Indian news categories.
# More granular than the feed-level iab_tier1/iab_tier2.
#
# WHY RULE-BASED (not ML)?
#   - No model download, no GPU, fast
#   - Deterministic — easier to audit and fix
#   - Indian-specific keywords you control
#   - Can upgrade to zero-shot classifier later without changing the interface
#
# Category priority: categories listed first win on ties.
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_RULES: dict[str, list[str]] = {
    # ── Cricket ───────────────────────────────────────────────────────────────
    # "odi" removed — matches "commodity". Use "one day international" instead.
    # "t20" kept — \b boundary makes it safe (won't match inside other words).
    "cricket": [
        # English
        "ipl", "bcci", "cricket", "test match", "one day international", "t20",
        "virat kohli", "rohit sharma", "ms dhoni", "sachin tendulkar",
        "batting", "bowling", "wicket", "innings", "run chase", "century",
        "world cup cricket", "ranji trophy", "super over", "duck out",
        "twenty20", "powerplay",
        # Hindi (Devanagari)
        "क्रिकेट", "आईपीएल", "बीसीसीआई", "विराट कोहली", "रोहित शर्मा",
        "धोनी", "बल्लेबाज", "गेंदबाज", "विकेट",
        # Tamil
        "கிரிக்கெட்", "ஐபிஎல்",
        # Telugu
        "క్రికెట్", "ఐపీఎల్",
        # Bengali
        "ক্রিকেট", "আইপিএল",
        # Marathi
        "क्रिकेट", "आयपीएल",
    ],

    # ── Crypto ────────────────────────────────────────────────────────────────
    # Separate category — crypto/blockchain is a major news vertical.
    # Must come before business/technology to win on tie-breaks.
    "crypto": [
        # English
        "bitcoin", "ethereum", "binance", "cryptocurrency", "crypto",
        "blockchain", "defi", "nft", "web3", "altcoin", "stablecoin",
        "usdt", "bnb", "btc", "eth", "solana", "ripple", "xrp", "dogecoin",
        "coinbase", "crypto wallet", "crypto exchange", "digital currency",
        "decentralized", "mining rig", "proof of stake", "proof of work",
        "ledger nano", "metamask", "uniswap", "airdrop", "token listing",
        "bull run", "bear market crypto", "halving", "satoshi",
        # Hindi (Devanagari)
        "बिटकॉइन", "क्रिप्टो", "ब्लॉकचेन", "क्रिप्टोकरेंसी",
        # Tamil
        "கிரிப்டோ", "பிட்காயின்",
        # Telugu
        "క్రిప్టో", "బిట్‌కాయిన్",
        # Bengali
        "ক্রিপ্টো", "বিটকয়েন",
    ],

    # ── Politics ──────────────────────────────────────────────────────────────
    # Removed "cm " and "mp " (trailing spaces replaced by \b word boundaries)
    # "cm" and "mp" kept — \b now prevents "income", "trump" false matches.
    "politics": [
        # English
        "bjp", "congress", "aap", "modi", "rahul gandhi", "parliament",
        "lok sabha", "rajya sabha", "election", "chief minister", "cm",
        "minister", "mla", "mp", "governor", "president of india",
        "political party", "vote", "constituency", "bypoll", "cabinet",
        "nda", "upa", "indi alliance", "assembly election", "general election",
        "exit poll", "voter turnout", "opposition party",
        # Hindi (Devanagari)
        "भाजपा", "कांग्रेस", "चुनाव", "संसद", "प्रधानमंत्री", "मुख्यमंत्री",
        "राज्यसभा", "लोकसभा", "सरकार", "राजनीति", "मतदान", "विपक्ष",
        # Tamil
        "தேர்தல்", "பாஜக", "நாடாளுமன்றம்", "அரசியல்",
        # Telugu
        "ఎన్నికలు", "పార్లమెంటు", "రాజకీయాలు", "బిజెపి",
        # Bengali
        "নির্বাচন", "সংসদ", "রাজনীতি", "বিজেপি",
        # Marathi
        "निवडणूक", "राजकारण",
    ],

    # ── Business ──────────────────────────────────────────────────────────────
    "business": [
        # English
        "sensex", "nse", "bse", "nifty", "rbi", "rupee", "gdp", "inflation",
        "stock market", "shares", "ipo", "sebi", "mutual fund", "revenue",
        "profit", "quarterly results", "unicorn", "funding round",
        "acquisition", "merger", "fiscal deficit", "import", "export",
        "gst", "trade deal", "economic growth", "economy", "retail sales",
        "foreign investment", "interest rate", "repo rate", "balance sheet",
        "earnings report", "market cap",
        # Hindi (Devanagari)
        "शेयर बाजार", "अर्थव्यवस्था", "बजट", "रुपया", "महंगाई", "व्यापार",
        "निवेश", "आर्थिक",
        # Tamil
        "பங்கு சந்தை", "பொருளாதாரம்", "வணிகம்",
        # Telugu
        "స్టాక్ మార్కెట్", "ఆర్థిక వ్యవస్థ", "వ్యాపారం",
        # Bengali
        "শেয়ার বাজার", "অর্থনীতি", "ব্যবসা",
    ],

    # ── Entertainment ─────────────────────────────────────────────────────────
    "entertainment": [
        # English
        "bollywood", "box office", "film", "movie", "actor", "actress",
        "director", "ott platform", "netflix", "amazon prime", "hotstar",
        "web series", "music album", "song release", "celebrity", "award",
        "filmfare", "shah rukh khan", "salman khan", "deepika padukone",
        "ranveer singh", "alia bhatt", "kollywood", "tollywood", "mollywood",
        "trailer launch", "box office collection",
        # Hindi (Devanagari)
        "बॉलीवुड", "फिल्म", "अभिनेता", "अभिनेत्री", "संगीत", "पुरस्कार",
        "सिनेमा", "वेब सीरीज", "गाना",
        # Tamil
        "சினிமா", "திரைப்படம்", "நடிகர்", "நடிகை", "இசை",
        # Telugu
        "సినిమా", "చలనచిత్రం", "నటుడు", "నటి", "సంగీతం",
        # Bengali
        "চলচ্চিত্র", "সিনেমা", "অভিনেতা", "সংগীত",
        # Marathi
        "चित्रपट", "अभिनेता",
    ],

    # ── Technology ────────────────────────────────────────────────────────────
    # Removed "ai " and "app " (trailing spaces). "ai" and "app" with \b are safe.
    # Removed "tech " — "tech" with \b is fine and more precise.
    "technology": [
        # English
        "artificial intelligence", "machine learning", "deep learning",
        "tech startup", "software", "smartphone", "5g network", "cybersecurity",
        "data breach", "isro", "satellite launch", "space mission",
        "elon musk", "google", "meta", "apple", "microsoft", "nvidia",
        "semiconductor", "chip shortage", "generative ai", "chatgpt", "openai",
        "app store", "android", "ios", "electric vehicle", "ev",
        # Hindi (Devanagari)
        "तकनीक", "कृत्रिम बुद्धिमत्ता", "मोबाइल", "इंटरनेट", "साइबर",
        "अंतरिक्ष", "उपग्रह",
        # Tamil
        "தொழில்நுட்பம்", "செயற்கை நுண்ணறிவு", "அறிவியல்",
        # Telugu
        "సాంకేతికత", "కృత్రిమ మేధస్సు", "విజ్ఞానం",
    ],

    # ── Sports (non-cricket) ──────────────────────────────────────────────────
    "sports": [
        # English
        "football", "fifa", "isl", "hockey", "badminton", "tennis",
        "olympics", "commonwealth games", "asian games", "wrestling",
        "kabaddi", "pro kabaddi", "formula 1", "grand prix", "chess",
        "athlete", "gold medal", "silver medal", "bronze medal",
        "nba", "isl league", "premier league", "champions league",
        # Hindi (Devanagari)
        "खेल", "फुटबॉल", "हॉकी", "बैडमिंटन", "कुश्ती", "कबड्डी",
        "ओलंपिक", "स्वर्ण पदक", "रजत पदक",
        # Tamil
        "விளையாட்டு", "கால்பந்து", "ஹாக்கி",
        # Telugu
        "క్రీడలు", "ఫుట్‌బాల్", "హాకీ",
        # Bengali
        "খেলাধুলা", "ফুটবল", "হকি",
    ],

    # ── Health ────────────────────────────────────────────────────────────────
    # Removed "who " (trailing space). "who" with \b matches "WHO" standalone.
    # Removed "drug" — too ambiguous. Use "drug trial", "pharmaceutical" instead.
    "health": [
        # English
        "hospital", "doctor", "medicine", "disease outbreak", "covid",
        "vaccine", "health ministry", "aiims", "cancer", "diabetes",
        "surgery", "epidemic", "pandemic", "WHO", "clinical trial",
        "patient", "pharmaceutical", "health insurance", "mental health",
        "dengue", "malaria", "tuberculosis", "ayushman bharat",
        # Hindi (Devanagari)
        "स्वास्थ्य", "अस्पताल", "डॉक्टर", "बीमारी", "दवाई", "कोविड",
        "टीकाकरण", "इलाज", "मरीज",
        # Tamil
        "சுகாதாரம்", "மருத்துவமனை", "மருத்துவர்", "நோய்",
        # Telugu
        "ఆరోగ్యం", "ఆసుపత్రి", "వైద్యుడు", "వ్యాధి",
        # Bengali
        "স্বাস্থ্য", "হাসপাতাল", "ডাক্তার", "রোগ",
    ],

    # ── Education ─────────────────────────────────────────────────────────────
    "education": [
        # English
        "school", "college", "university", "jee", "neet", "upsc", "ias",
        "board exam", "cbse", "icse", "ugc", "iit", "iim", "scholarship",
        "syllabus", "admission", "student protest", "education policy", "nep",
        "exam result", "entrance exam", "higher education",
        # Hindi (Devanagari)
        "शिक्षा", "स्कूल", "कॉलेज", "परीक्षा", "विश्वविद्यालय", "छात्र",
        "प्रवेश", "छात्रवृत्ति",
        # Tamil
        "கல்வி", "பள்ளி", "கல்லூரி", "தேர்வு",
        # Telugu
        "విద్య", "పాఠశాల", "కళాశాల", "పరీక్ష",
        # Bengali
        "শিক্ষা", "স্কুল", "কলেজ", "পরীক্ষা",
    ],

    # ── Crime ─────────────────────────────────────────────────────────────────
    # Removed "ed " (trailing space). "ed" with \b matches standalone "ED".
    "crime": [
        # English
        "murder", "rape", "assault", "arrested", "police", "fir", "court",
        "judge", "verdict", "convicted", "bail", "chargesheet", "cbi",
        "enforcement directorate", "scam", "fraud", "gangster", "kidnap",
        "robbery", "terrorist", "accused", "crime scene", "death penalty",
        # Hindi (Devanagari)
        "हत्या", "गिरफ्तार", "पुलिस", "अदालत", "बलात्कार", "धोखाधड़ी",
        "अपराध", "जमानत", "गैंगस्टर", "अपहरण",
        # Tamil
        "கொலை", "கைது", "போலீஸ்", "நீதிமன்றம்",
        # Telugu
        "హత్య", "అరెస్టు", "పోలీసు", "న్యాయస్థానం",
        # Bengali
        "হত্যা", "গ্রেপ্তার", "পুলিশ", "আদালত",
    ],

    # ── Environment ───────────────────────────────────────────────────────────
    "environment": [
        # English
        "climate change", "pollution", "air quality", "aqi", "flood",
        "drought", "cyclone", "earthquake", "wildlife", "forest fire",
        "deforestation", "carbon emission", "renewable energy", "solar power",
        "wind energy", "green energy", "environment ministry", "net zero",
        "global warming", "glacier", "biodiversity",
        # Hindi (Devanagari)
        "पर्यावरण", "प्रदूषण", "बाढ़", "सूखा", "भूकंप", "चक्रवात",
        "जलवायु परिवर्तन", "वन", "सौर ऊर्जा",
        # Tamil
        "சுற்றுச்சூழல்", "மாசுபாடு", "வெள்ளம்",
        # Telugu
        "పర్యావరణం", "కాలుష్యం", "వరదలు",
        # Bengali
        "পরিবেশ", "দূষণ", "বন্যা",
    ],

    # ── World ─────────────────────────────────────────────────────────────────
    # Removed "un " (trailing space). "un" with \b matches standalone "UN".
    # Removed "usa " — "usa" with \b is fine.
    "world": [
        # English
        "united states", "usa", "white house", "china", "pakistan",
        "russia", "ukraine", "europe", "nato", "united nations",
        "g20", "g7", "foreign ministry", "diplomatic", "sanctions",
        "border dispute", "international relations", "global summit",
        "imf", "world bank", "un security council", "middle east",
        "israel", "iran", "north korea",
        # Hindi (Devanagari)
        "अमेरिका", "चीन", "पाकिस्तान", "रूस", "अंतरराष्ट्रीय",
        "विदेश मंत्रालय", "कूटनीति", "संयुक्त राष्ट्र",
        # Tamil
        "அமெரிக்கா", "சீனா", "பாகிஸ்தான்", "சர்வதேசம்",
        # Telugu
        "అమెరికా", "చైనా", "పాకిస్తాన్", "అంతర్జాతీయం",
        # Bengali
        "আমেরিকা", "চীন", "পাকিস্তান", "আন্তর্জাতিক",
    ],
}
# Fallback when no category matches
CATEGORY_DEFAULT = "general"


# ─────────────────────────────────────────────────────────────────────────────
# STEP: IMAGE DOWNLOAD + PERCEPTUAL HASHING
#
# IMAGE_DOWNLOAD_TIMEOUT: seconds to wait for image HTTP response
# IMAGE_MAX_BYTES:        skip images larger than this (5 MB)
#                         Avoids downloading hi-res photos just to hash them.
# IMAGE_HASH_SIZE:        pHash grid size. 8 = 64-bit hash (standard).
#                         Higher = more detail, larger hash.
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_DOWNLOAD_TIMEOUT = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "10"))
IMAGE_MAX_BYTES        = 5 * 1024 * 1024   # 5 MB
IMAGE_HASH_SIZE        = 8                  # produces 64-bit pHash


# ─────────────────────────────────────────────────────────────────────────────
# STEP: STORY CLUSTERING
#
# Two articles belong to the same cluster if they're about the same event.
# We detect this using two signals:
#
#   1. Entity overlap  — share ≥ CLUSTER_MIN_SHARED_ENTITIES named entities
#   2. Title similarity — title SimHash distance ≤ CLUSTER_SIMHASH_THRESHOLD
#
# CLUSTER_SCORE_THRESHOLD:
#   Combined score (entity Jaccard + title similarity) needed to join a cluster.
#   0.35 is calibrated for Indian news wire syndication patterns.
#   Raise to 0.5 if you're getting too many false clusters.
#
# CLUSTER_WINDOW_HOURS:
#   Search window for the pgvector HNSW nearest-neighbour query.
#   Pass 0 to search ALL clusters (no window) — safe with pgvector since
#   the HNSW index keeps queries fast regardless of cluster count.
#   48h is a sensible default that covers most Indian news lifecycles.
# ─────────────────────────────────────────────────────────────────────────────
# Raised from 48h → 0 (no window) after Supabase Pro upgrade.
# 0 = search ALL clusters — pgvector HNSW keeps queries fast regardless of cluster count.
CLUSTER_WINDOW_HOURS        = int(os.getenv("CLUSTER_WINDOW_HOURS", "0"))
# Singleton clusters older than this are deleted by cleanup_clusters.py
CLUSTER_SINGLETON_TTL_HOURS = int(os.getenv("CLUSTER_SINGLETON_TTL_HOURS", "72"))


# ─────────────────────────────────────────────────────────────────────────────
# STEP: SENTIMENT ANALYSIS (multilingual transformer)
#
# Model: lxyuan/distilbert-base-multilingual-cased-sentiments-student
#   - Multilingual: EN, Hindi, Tamil, Telugu, Bengali, and 100+ others
#   - ~268 MB, CPU-fast (distilled from mDeBERTa)
#   - Upgrade from VADER: Hindi/Tamil/Telugu articles now get sentiment
#     instead of NULL — covers the 40-50% of our corpus that was skipped
#
# sentiment_score = positive_prob - negative_prob → −1.0 to +1.0
# (matches VADER compound score semantics for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENT_MULTILINGUAL_MODEL = os.getenv(
    "SENTIMENT_MULTILINGUAL_MODEL",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
)
SENTIMENT_POSITIVE_THRESHOLD =  0.05
SENTIMENT_NEGATIVE_THRESHOLD = -0.05


# ─────────────────────────────────────────────────────────────────────────────
# STEP: PROPENSITY SCORING
#
# Estimates how likely an article is to go viral / drive user engagement.
#
# Phase 1 (immediate): weighted formula using outlet_count, category, entities
# Phase 2 (weekly):    LightGBM model trained on outlet_count as self-supervised
#                      label. Model file replaces the formula when present.
#
# Self-supervised training signal:
#   outlet_count on article_clusters = how many outlets reported the same story
#   High outlet_count → genuinely important news (wire syndication pattern)
#   Low outlet_count → niche or unique story
#
# PROPENSITY_MODEL_PATH: path to trained LightGBM .lgb model file
# PROPENSITY_MAX_OUTLET_COUNT: log-normalization cap for outlet virality signal
# ─────────────────────────────────────────────────────────────────────────────
PROPENSITY_MODEL_PATH       = os.getenv("PROPENSITY_MODEL_PATH", "models/propensity_model.lgb")
PROPENSITY_MAX_OUTLET_COUNT = 50   # log(1+50) ≈ 3.93 — normalization denominator

PROPENSITY_CATEGORY_WEIGHTS: dict[str, float] = {
    "cricket":       0.90,
    "politics":      0.85,
    "business":      0.75,
    "entertainment": 0.70,
    "technology":    0.70,
    "sports":        0.70,
    "health":        0.65,
    "world":         0.65,
    "crime":         0.60,
    "education":     0.55,
    "environment":   0.55,
    "crypto":        0.50,
    "general":       0.30,
}
