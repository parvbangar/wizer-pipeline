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
ENRICH_BATCH_SIZE    = int(os.getenv("ENRICH_BATCH_SIZE",    "500"))
ENRICH_MIN_WORD_COUNT = int(os.getenv("ENRICH_MIN_WORD_COUNT", "50"))


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
SPACY_MODEL              = os.getenv("SPACY_MODEL",             "en_core_web_sm")
SPACY_MULTILINGUAL_MODEL = os.getenv("SPACY_MULTILINGUAL_MODEL", "xx_ent_wiki_sm")
NER_MIN_SALIENCE         = 0.1
#
# Two models are used based on detected language:
#   en_core_web_sm  — English articles (highest accuracy on English news)
#   xx_ent_wiki_sm  — All Indian regional languages: hi, ta, te, bn, mr, gu,
#                     kn, ml, pa, ur, etc. Trained on Wikipedia across 50+ languages.
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
    "cricket": [
        # English
        "ipl", "bcci", "cricket", "test match", "odi", "t20", "virat kohli",
        "rohit sharma", "ms dhoni", "sachin", "batting", "bowling", "wicket",
        "innings", "run chase", "century", "world cup cricket",
        # Hindi (Devanagari)
        "क्रिकेट", "आईपीएल", "बीसीसीआई", "विराट", "रोहित", "धोनी", "बल्लेबाज", "गेंदबाज",
        # Tamil
        "கிரிக்கெட்", "ஐபிஎல்",
        # Telugu
        "క్రికెట్", "ఐపీఎల్",
        # Bengali
        "ক্রিকেট", "আইপিএল",
        # Marathi
        "क्रिकेट", "आयपीएल",
    ],
    "politics": [
        # English
        "bjp", "congress", "aap", "modi", "rahul gandhi", "parliament",
        "lok sabha", "rajya sabha", "election", "cm ", "chief minister",
        "minister", "mla", "mp ", "governor", "president of india",
        "political party", "vote", "constituency", "bypoll", "cabinet",
        "nda", "upa", "indi alliance",
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
        "निवडणूक", "संसद", "राजकारण",
    ],
    "business": [
        # English
        "sensex", "nse", "bse", "nifty", "rbi", "rupee", "gdp", "inflation",
        "stock market", "shares", "ipo", "sebi", "mutual fund", "revenue",
        "profit", "quarterly results", "startup", "unicorn", "funding",
        "acquisition", "merger", "budget", "fiscal", "import", "export",
        "gst", "trade", "economic", "economy", "retail", "fdi",
        # Hindi (Devanagari)
        "शेयर बाजार", "अर्थव्यवस्था", "बजट", "रुपया", "महंगाई", "व्यापार",
        "निवेश", "कंपनी", "बाजार", "आर्थिक",
        # Tamil
        "பங்கு சந்தை", "பொருளாதாரம்", "வணிகம்",
        # Telugu
        "స్టాక్ మార్కెట్", "ఆర్థిక వ్యవస్థ", "వ్యాపారం",
        # Bengali
        "শেয়ার বাজার", "অর্থনীতি", "ব্যবসা",
    ],
    "entertainment": [
        # English
        "bollywood", "box office", "film", "movie", "actor", "actress",
        "director", "ott", "netflix", "amazon prime", "hotstar", "web series",
        "music", "album", "song", "celebrity", "award", "filmfare",
        "srk", "shah rukh", "salman", "deepika", "ranveer", "alia",
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
        "चित्रपट", "सिनेमा", "अभिनेता",
    ],
    "technology": [
        # English
        "artificial intelligence", "ai ", "machine learning", "startup",
        "tech ", "software", "app ", "smartphone", "5g", "cyber", "hack",
        "data breach", "isro", "satellite", "space", "elon musk",
        "google", "meta", "apple", "microsoft", "amazon", "chip",
        # Hindi (Devanagari)
        "तकनीक", "कृत्रिम बुद्धिमत्ता", "मोबाइल", "इंटरनेट", "साइबर",
        "अंतरिक्ष", "उपग्रह",
        # Tamil
        "தொழில்நுட்பம்", "செயற்கை நுண்ணறிவு", "அறிவியல்",
        # Telugu
        "సాంకేతికత", "కృత్రిమ మేధస్సు", "విజ్ఞానం",
    ],
    "sports": [
        # English
        "football", "fifa", "isl", "hockey", "badminton", "tennis",
        "olympics", "commonwealth games", "asian games", "wrestling",
        "kabaddi", "pro kabaddi", "formula 1", "f1", "chess", "athlete",
        "gold medal", "silver medal",
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
    "health": [
        # English
        "hospital", "doctor", "medicine", "disease", "covid", "vaccine",
        "health ministry", "aiims", "cancer", "diabetes", "surgery",
        "outbreak", "epidemic", "who ", "drug", "clinical trial", "patient",
        # Hindi (Devanagari)
        "स्वास्थ्य", "अस्पताल", "डॉक्टर", "बीमारी", "दवा", "कोविड",
        "टीका", "इलाज", "मरीज",
        # Tamil
        "சுகாதாரம்", "மருத்துவமனை", "மருத்துவர்", "நோய்",
        # Telugu
        "ఆరోగ్యం", "ఆసుపత్రి", "వైద్యుడు", "వ్యాధి",
        # Bengali
        "স্বাস্থ্য", "হাসপাতাল", "ডাক্তার", "রোগ",
    ],
    "education": [
        # English
        "school", "college", "university", "jee", "neet", "upsc", "ias",
        "board exam", "cbse", "icse", "ugc", "iit", "iim", "scholarship",
        "syllabus", "admission", "student", "education policy", "nep",
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
    "crime": [
        # English
        "murder", "rape", "assault", "arrested", "police", "fir", "court",
        "judge", "verdict", "convicted", "bail", "chargesheet", "cbi",
        "ed ", "enforcement directorate", "scam", "fraud", "crime",
        "gangster", "kidnap", "robbery",
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
    "environment": [
        # English
        "climate", "pollution", "air quality", "aqi", "flood", "drought",
        "cyclone", "earthquake", "wildlife", "forest", "deforestation",
        "carbon", "renewable", "solar", "wind energy", "green",
        "environment ministry",
        # Hindi (Devanagari)
        "पर्यावरण", "प्रदूषण", "बाढ़", "सूखा", "भूकंप", "चक्रवात",
        "जलवायु", "वन", "सौर ऊर्जा",
        # Tamil
        "சுற்றுச்சூழல்", "மாசுபாடு", "வெள்ளம்",
        # Telugu
        "పర్యావరణం", "కాలుష్యం", "వరదలు",
        # Bengali
        "পরিবেশ", "দূষণ", "বন্যা",
    ],
    "world": [
        # English
        "united states", "usa ", "china", "pakistan", "russia", "ukraine",
        "europe", "nato", "un ", "united nations", "g20", "g7",
        "foreign ministry", "diplomatic", "sanctions", "border dispute",
        "international", "global",
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
#   Only cluster against articles published in the last N hours.
#   48h covers breaking news lifecycles (most Indian stories resolve in 24-36h).
#
# CLUSTER_MAX_CANDIDATES:
#   Max number of recent clusters to compare against per article.
#   Prevents slow O(n²) comparisons when there are thousands of active clusters.
# ─────────────────────────────────────────────────────────────────────────────
CLUSTER_WINDOW_HOURS        = int(os.getenv("CLUSTER_WINDOW_HOURS", "48"))
CLUSTER_MIN_SHARED_ENTITIES = 2
CLUSTER_SIMHASH_THRESHOLD   = 5     # bit distance ≤ 5 = titles are very similar
CLUSTER_SCORE_THRESHOLD     = 0.35
CLUSTER_MAX_CANDIDATES      = 500   # max clusters to compare per article


# ─────────────────────────────────────────────────────────────────────────────
# STEP: SENTIMENT ANALYSIS (VADER)
#
# VADER is English-only. For non-English articles (Hindi, Tamil, etc.)
# sentiment and sentiment_score are stored as NULL.
# Upgrade path: replace with a multilingual model (XLM-R, MuRIL) later.
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENT_POSITIVE_THRESHOLD =  0.05   # compound score above this = positive
SENTIMENT_NEGATIVE_THRESHOLD = -0.05   # compound score below this = negative
# Between these thresholds = neutral
SENTIMENT_ENGLISH_ONLY_LANGS = {"en", "en-us", "en-gb", "en-in"}
