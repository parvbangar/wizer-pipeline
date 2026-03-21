"""
dedup.py
════════
Deduplication logic for the pipeline.

WHY DO WE NEED THIS?
  The same article gets published across many RSS feeds.
  For example: a PTI wire story about a budget announcement will appear in
  The Hindu, NDTV, India Today, and 50 other feeds — all within minutes.
  Without deduplication, your articles table would have 50 identical rows.

HOW IT WORKS — TWO LAYERS:
─────────────────────────────────────────────────────────────────────────────
LAYER 1: EXACT URL DEDUPLICATION
  Step 1: Normalise the URL (explained below)
  Step 2: Hash the normalised URL with MurmurHash3 → a 64-bit integer
  Step 3: Check if that integer exists in articles.url_hash
  
  Why normalise first?
    These four URLs all point to the same article:
      https://www.thehindu.com/news/article123.cms?utm_source=twitter
      https://thehindu.com/news/article123.cms?fbclid=xyz
      https://www.thehindu.com/news/article123.cms/
      http://thehindu.com/news/article123.cms
    Without normalisation, all four would get different hashes and
    all four would be stored as separate articles.  Wrong!

LAYER 2: NEAR-DUPLICATE TITLE DEDUPLICATION (SimHash)
  Even when URLs are different, the same story often gets copied.
  PTI sends the same wire story to 50 outlets — they all publish it
  with slightly different titles.
  
  SimHash converts a title into a 64-bit fingerprint where similar
  titles produce similar bit patterns.  Two titles with a Hamming
  distance ≤ 3 (at most 3 bits different) are flagged as near-duplicates.
  
  We still STORE the article, but mark is_duplicate=True.
  This way you have the data but can filter duplicates in your app.
─────────────────────────────────────────────────────────────────────────────

WHAT IS MURMURHASH3?
  A very fast non-cryptographic hash function.
  Unlike SHA-256 (slow, cryptographically secure), MurmurHash3 is designed
  purely for speed.  Hashing a URL takes ~200 nanoseconds.
  We hash millions of URLs per day, so speed matters.

WHAT IS SIMHASH?
  A technique that maps text to a bit fingerprint where similar text
  produces similar fingerprints (unlike regular hashes where tiny changes
  completely change the output).  Used by Google for near-duplicate detection.
"""

import re
import unicodedata
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl

# MurmurHash3 — fast 64-bit hashing
try:
    import mmh3
    _HAS_MMH3 = True
except ImportError:
    # Fallback to stdlib hashlib if mmh3 not installed
    import hashlib as _hashlib
    _HAS_MMH3 = False

from pipeline.config import HASH_SEED, SIMHASH_DISTANCE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# TRACKING PARAMETERS TO STRIP FROM URLs
# These are added by analytics/marketing tools and don't change article content
# ─────────────────────────────────────────────────────────────────────────────
_STRIP_PARAMS: frozenset = frozenset({
    # Google Analytics / UTM
    "utm_source", "utm_medium", "utm_campaign", "utm_term",
    "utm_content", "utm_id", "utm_referrer",
    # Google Ads
    "gclid", "gclsrc", "gbraid", "wbraid",
    # Facebook
    "fbclid", "fb_action_ids", "fb_source", "fb_ref",
    # Twitter / X
    "twclid",
    # Microsoft Ads
    "msclkid",
    # HubSpot
    "hsa_acc", "hsa_cam", "hsa_grp", "hsa_ad", "hsa_src",
    "hsa_tgt", "hsa_kw", "hsa_mt", "hsa_net", "hsa_ver",
    # Mailchimp
    "mc_cid", "mc_eid",
    # Miscellaneous tracking
    "_ga", "igshid", "s_cid", "ncid", "cmpid", "mbid",
    "ref", "referrer", "source", "origin",
    "sid", "session_id", "share", "sharesource",
})


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: EXACT URL DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def normalise_url(url: str) -> str:
    """
    Convert any URL into a canonical form so the same article always
    produces the same string regardless of tracking parameters, www prefix,
    trailing slashes, or letter case in the domain.

    Example:
      Input:  "https://www.thehindu.com/news/art.cms?utm_source=tw&fbclid=abc"
      Output: "https://thehindu.com/news/art.cms"

    Steps applied:
      1. Strip whitespace
      2. Add https:// if missing
      3. Lowercase the domain (thehindu.com not TheHindu.COM)
      4. Remove www. prefix
      5. Remove tracking query parameters
      6. Sort remaining query parameters (order doesn't matter for content)
      7. Collapse double slashes in the path
      8. Remove trailing slash from path
      9. Drop the fragment (#comments, #section etc.)
    """
    if not url:
        return ""

    url = url.strip()

    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed = urlparse(url)
    except Exception:
        return url  # return as-is if parsing fails

    # Lowercase and strip www. from host
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    # Clean the path
    path = re.sub(r"/{2,}", "/", parsed.path)   # collapse //
    if len(path) > 1:
        path = path.rstrip("/")                  # remove trailing /

    # Filter and sort query parameters
    clean_params = sorted([
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k.lower() not in _STRIP_PARAMS
    ])
    query = urlencode(clean_params)

    # Reassemble — no fragment
    return urlunparse((
        parsed.scheme.lower(),  # http or https (lowercased)
        host,                   # domain without www
        path,                   # cleaned path
        "",                     # params (rarely used)
        query,                  # cleaned query string
        "",                     # no fragment
    ))


# def url_hash(url: str) -> int:
#     """
#     Hash a normalised URL to a 64-bit signed integer.

#     This integer is stored in articles.url_hash (a bigint column).
#     PostgreSQL bigint = 64-bit signed integer, range: -2^63 to 2^63-1.

#     Why not use the URL string directly for dedup?
#       String comparison on millions of rows is slow.
#       Integer comparison on an indexed column is instant (~microseconds).

#     Returns an integer suitable for PostgreSQL bigint.
#     """
#     normalised = normalise_url(url)
#     data = normalised.encode("utf-8")

#     if _HAS_MMH3:
#         # mmh3.hash64 returns (h1, h2) — two 64-bit hashes, we use h1
#         h1, _ = mmh3.hash64(data, seed=HASH_SEED, signed=True)
#         return h1
#     else:
#         # SHA-256 fallback: fold 256 bits → 64 bits, convert to signed
#         digest = _hashlib.sha256(data).digest()
#         unsigned = int.from_bytes(digest[:8], "big")
#         # Convert unsigned 64-bit to signed 64-bit
#         if unsigned >= (1 << 63):
#             unsigned -= (1 << 64)
#         return unsigned

def url_hash(url: str) -> int:
    """
    Return a 64-bit SIGNED integer fingerprint of the normalised URL.
    PostgreSQL bigint range: -9223372036854775808 to 9223372036854775807
    """
    data = normalise_url(url).encode("utf-8")
    if _HAS_MMH3:
        h1, _ = mmh3.hash64(data, seed=HASH_SEED, signed=True)
        return h1
    # SHA-256 fallback — fold to 64-bit signed
    digest   = _hashlib.sha256(data).digest()
    unsigned = int.from_bytes(digest[:8], "big")
    # Force into signed 64-bit range
    if unsigned >= (1 << 63):
        unsigned -= (1 << 64)
    return unsigned

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2: NEAR-DUPLICATE TITLE DEDUPLICATION (SimHash)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """
    Break text into word tokens for SimHash.
    Lowercase, remove punctuation, strip Unicode accents.
    """
    # Normalise Unicode (e.g. café → cafe)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Lowercase and extract words
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def simhash(text: str) -> int:
    """
    Compute a 64-bit SimHash fingerprint for the given text.

    SimHash works like this:
      1. Split text into tokens (words)
      2. Hash each token with a regular hash function
      3. For each bit position (0-63):
           Add +1 if that bit is 1 in the token hash
           Add -1 if that bit is 0 in the token hash
      4. For each bit position: if the sum is positive → bit=1, else bit=0
      5. Pack the 64 bits into a single integer

    Why does this work?
      Similar texts share most words → similar bit patterns → close fingerprints.
      The Hamming distance (number of bits that differ) between two fingerprints
      tells us how similar the texts are.
    """
    if not text or not text.strip():
        return 0

    tokens = _tokenise(text)
    if not tokens:
        return 0

    # 64-bit accumulator: v[i] = sum of +1/-1 for each token's i-th bit
    v = [0] * 64

    for token in tokens:
        # Hash the token (using MurmurHash3 or SHA-256 fallback)
        token_bytes = token.encode("utf-8")
        if _HAS_MMH3:
            h1, _ = mmh3.hash64(token_bytes, seed=HASH_SEED, signed=False)
            token_hash = h1
        else:
            digest = _hashlib.sha256(token_bytes).digest()
            token_hash = int.from_bytes(digest[:8], "big")

        # For each bit in the 64-bit hash
        for i in range(64):
            bit = (token_hash >> i) & 1
            v[i] += 1 if bit else -1

    # Build the final fingerprint: 1 if v[i] > 0 else 0
    fingerprint = 0
    for i in range(64):
        if v[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint


def hamming_distance(h1: int, h2: int) -> int:
    """
    Count the number of bit positions where h1 and h2 differ.

    This is called Hamming distance.
    Distance 0  = identical bit pattern = identical text
    Distance ≤ 3 = near-duplicate (same story, slightly different wording)
    Distance > 3 = different content

    Python trick: XOR gives 1 at every position where the bits differ,
    then bin().count("1") counts those positions.
    """
    return bin(h1 ^ h2).count("1")


def is_near_duplicate(title1_hash: int, title2_hash: int) -> bool:
    """
    Return True if two SimHash fingerprints are close enough to be
    considered near-duplicates (same story, different source).
    """
    return hamming_distance(title1_hash, title2_hash) <= SIMHASH_DISTANCE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# Run: python -m pipeline.dedup
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== URL Normalisation Tests ===")
    cases = [
        ("https://www.thehindu.com/news/art.cms?utm_source=tw",
         "https://thehindu.com/news/art.cms"),
        ("https://ndtv.com/india/story/?fbclid=abc",
         "https://ndtv.com/india/story"),
        ("HTTP://IndianExpress.COM/article/india/story-456/",
         "https://indianexpress.com/article/india/story-456"),
    ]
    for raw, expected in cases:
        got = normalise_url(raw)
        ok = "✓" if got == expected else "✗"
        print(f"  {ok}  {got}")

    print("\n=== SimHash Near-Duplicate Tests ===")
    title_pairs = [
        ("Budget 2024: FM announces tax relief for middle class",
         "Budget 2024: Finance Minister announces income tax relief"),   # should be near-dup
        ("IPL 2024: Mumbai Indians beat Chennai Super Kings",
         "Sensex falls 500 points amid global selloff"),                 # should NOT be near-dup
    ]
    for t1, t2 in title_pairs:
        h1, h2 = simhash(t1), simhash(t2)
        dist = hamming_distance(h1, h2)
        dup = "NEAR-DUP" if is_near_duplicate(h1, h2) else "different"
        print(f"  [{dup}] distance={dist}")
        print(f"    T1: {t1[:60]}")
        print(f"    T2: {t2[:60]}")
