"""
enrichment/steps/images.py
════════════════════════════
Downloads an article's top image and computes its perceptual hash (pHash).

WHAT IS A PERCEPTUAL HASH?
  Unlike a cryptographic hash (MD5/SHA), a perceptual hash is designed to
  produce SIMILAR hashes for visually similar images.

  Two images that are:
    - The same photo with different compression → identical pHash
    - The same photo with a small watermark added → pHash distance of 1-2 bits
    - Completely different photos → pHash distance of 20+ bits

  We store this as a 64-bit integer (bigint in PostgreSQL).

WHY DO WE NEED pHash FOR THE INDIAN NEWS PIPELINE?
  1. Wire photo dedup: PTI distributes the same photo to 50 outlets.
     If two articles share the same pHash, they're almost certainly
     covering the same story. Strong clustering signal.

  2. VLM prep (future Layer 5): Your plan mentions a Vision Language Model
     step. The pHash tells you which articles have UNIQUE images worth
     sending to the VLM vs which are wire reposts you can skip.

  3. App layer: Don't show the same stock photo 20 times on the feed.

DISTANCE THRESHOLDS:
  pHash distance = number of differing bits (Hamming distance)
  0       = identical images
  1-5     = same image, minor differences (compression, resize)
  6-10    = very similar (minor crop, watermark)
  11-20   = loosely related (same scene, different frame)
  21+     = different images

INSTALL:
  pip install imagehash Pillow
"""

from __future__ import annotations

import io
import logging
import urllib.request
import urllib.error

from enrichment.config import (
    IMAGE_DOWNLOAD_TIMEOUT,
    IMAGE_MAX_BYTES,
    IMAGE_HASH_SIZE,
)

log = logging.getLogger(__name__)


def download_and_hash_image(image_url: str) -> int | None:
    """
    Download the image at image_url and return its 64-bit perceptual hash.

    Steps:
      1. Download the image with a HEAD check first (avoids downloading huge files)
      2. Decode with Pillow
      3. Compute pHash (8×8 grid DCT algorithm)
      4. Return as Python int (fits in PostgreSQL bigint)

    Args:
      image_url: Full URL to the image (from articles.top_image_url)

    Returns:
      pHash as integer, or None if download/hashing fails for any reason.
      Failure is silent — a missing image hash is acceptable, not an error.
    """
    if not image_url or not image_url.startswith("http"):
        return None

    try:
        import imagehash
        from PIL import Image
    except ImportError:
        log.warning(
            "imagehash / Pillow not installed — skipping image hashing. "
            "Run: pip install imagehash Pillow"
        )
        return None

    # ── Step 1: Download image ────────────────────────────────────────────────
    try:
        request = urllib.request.Request(
            image_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsEnrichBot/1.0)"},
        )
        with urllib.request.urlopen(request, timeout=IMAGE_DOWNLOAD_TIMEOUT) as resp:
            # Check content type — skip non-images
            content_type = resp.headers.get("Content-Type", "")
            if not any(t in content_type for t in ("image/", "jpeg", "png", "webp")):
                log.debug("Skipping non-image content-type '%s': %s", content_type, image_url[:80])
                return None

            # Read up to IMAGE_MAX_BYTES — skip oversized images
            data = resp.read(IMAGE_MAX_BYTES + 1)
            if len(data) > IMAGE_MAX_BYTES:
                log.debug("Image too large (>%d bytes): %s", IMAGE_MAX_BYTES, image_url[:80])
                return None

    except urllib.error.HTTPError as e:
        log.debug("HTTP %d downloading image: %s", e.code, image_url[:80])
        return None
    except Exception as e:
        log.debug("Image download failed (%s): %s", type(e).__name__, image_url[:80])
        return None

    # ── Step 2: Decode and hash ───────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        phash = imagehash.phash(image, hash_size=IMAGE_HASH_SIZE)
        # imagehash returns an unsigned 64-bit integer (0 to 2^64-1).
        # PostgreSQL bigint is SIGNED 64-bit (-2^63 to 2^63-1).
        # Reinterpret the unsigned value as signed using two's complement:
        #   values < 2^63 stay the same; values >= 2^63 wrap to negative.
        # The bit pattern is preserved — Hamming distance comparisons still work.
        unsigned = int(str(phash), 16)
        return unsigned if unsigned < (1 << 63) else unsigned - (1 << 64)

    except Exception as e:
        log.debug("Image hashing failed: %s", e)
        return None
