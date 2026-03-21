# News Ingestion Pipeline

A production-grade RSS ingestion pipeline built for 25,000+ feeds.
Zero-budget friendly. Beginner-explained. Startup-quality.

---

## What this pipeline does

```
Every 5/15/60 minutes (depending on tier):

  GitHub Actions / cron
        │
        ▼
  main.py (CLI entry)
        │
        ▼
  poller.py (async engine)
  ├── Loads feeds due for polling from your feeds table
  ├── For each feed (up to 20 in parallel):
  │     ├── circuit_breaker: skip if feed is inactive/dormant
  │     ├── feedparser: fetch and parse the RSS/Atom feed
  │     ├── For each article entry:
  │     │     ├── dedup.normalise_url() → clean the URL
  │     │     ├── dedup.url_hash()      → 64-bit integer fingerprint
  │     │     ├── Check in-memory hash set (Layer 1 exact dedup)
  │     │     ├── db.url_hash_exists()  (Layer 2 exact dedup — DB)
  │     │     ├── crawler.crawl_article() → full text, OG tags, images
  │     │     └── dedup.simhash()       → near-duplicate title check
  │     ├── db.upsert_articles() → bulk insert, duplicates silently skipped
  │     └── db.update_feed_after_poll() → update fail_count, last_polled_at
  └── db.log_run() → write summary row to pipeline_runs
```

---

## Understanding each file

### `pipeline/config.py`
All settings in one place. Every column name, every constant.
When you need to change something — change it here, it propagates everywhere.

### `pipeline/dedup.py`
Two deduplication strategies:

**Exact dedup (url_hash)**
```
"https://www.thehindu.com/news/art.cms?utm_source=tw&fbclid=abc"
        ↓ normalise_url()
"https://thehindu.com/news/art.cms"
        ↓ url_hash() → MurmurHash3
-3847562910483625741   ← stored as url_hash bigint in DB
```
The UNIQUE index on articles.url_hash guarantees no two rows ever have
the same normalised URL.

**Near-dedup (title_simhash)**
Catches PTI wire stories republished by 50 outlets with slightly different titles.
Articles marked is_duplicate=True are still stored — your app filters them.

### `pipeline/circuit_breaker.py`
Two protection mechanisms:
- **Error streak**: fail_count reaches 5 → is_active=False
- **Dormancy**: no new articles in 30 days → is_active=False

Reset a disabled feed in Supabase SQL Editor:
```sql
UPDATE feeds SET is_active = true, fail_count = 0 WHERE id = 'your-feed-uuid';
```

### `pipeline/crawler.py`
Fetches each article page and extracts:
- Full article text (trafilatura → newspaper3k fallback)
- Open Graph tags (og:title, og:image, og:description, article:published_time)
- JSON-LD structured data
- Author, publication date, canonical URL

Paywalled articles: stores RSS metadata + og:image only, is_crawled=False.

### `pipeline/db.py`
All database calls. Nothing else talks to Supabase directly.
Uses your exact column names throughout.

### `pipeline/poller.py`
The async engine. Runs 20 feeds in parallel using asyncio semaphores.
Each feed gets its own error handling — one bad feed can't crash others.

---

## Setup

### Step 1 — Clone and install

```bash
git clone https://github.com/your-org/news-pipeline.git
cd news-pipeline
pip install -r requirements.txt
```

### Step 2 — Configure secrets

```bash
cp .env.example .env
# Edit .env — add your SUPABASE_URL and SUPABASE_SERVICE_KEY
# Find both at: Supabase Dashboard → Project Settings → API
# Use the SERVICE KEY (long JWT), not the anon key
```

### Step 3 — Run the database migration

Open **Supabase → SQL Editor** → paste and run `docs/migration.sql`.

This creates:
- `articles` table with your exact column schema
- `pipeline_runs` monitoring table
- All indexes including the UNIQUE index on url_hash
- Monitoring views: `feed_health`, `dormancy_watch`, `article_stats`, `recent_runs`
- Row Level Security policies

### Step 4 — Test locally

```bash
# Run all tests (no network, no Supabase needed)
pytest tests/ -v

# Dry run: parse feeds, log what would happen, NO DB writes
python main.py --tier tier1_high --dry-run --verbose

# Real run: poll tier1 feeds
python main.py --tier tier1_high --verbose
```

### Step 5 — Set up GitHub (for automation)

1. Push this folder to a GitHub repository (make it **public** — free unlimited Actions minutes)
2. Go to: **Settings → Secrets and variables → Actions → New repository secret**
3. Add two secrets:
   - `SUPABASE_URL`  →  `https://yourproject.supabase.co`
   - `SUPABASE_SERVICE_KEY`  →  the long JWT from Supabase API settings

The three workflow files in `.github/workflows/` activate automatically:
- `ingest_tier1.yml` → every 5 minutes
- `ingest_tier2.yml` → every 15 minutes
- `ingest_tier3.yml` → every hour

---

## Monitoring

Query these views in Supabase SQL Editor anytime:

```sql
-- Which feeds are producing articles? Which are dead?
SELECT * FROM feed_health ORDER BY days_since_new_article DESC NULLS LAST;

-- Feeds about to go dormant (20+ days without articles)
SELECT * FROM dormancy_watch;

-- Crawl success rate per publisher
SELECT * FROM article_stats ORDER BY total DESC;

-- History of every pipeline run
SELECT * FROM recent_runs;

-- Quick health check
SELECT
  count(*) FILTER (WHERE is_active = true)  AS active_feeds,
  count(*) FILTER (WHERE is_active = false) AS disabled_feeds,
  count(*) FILTER (WHERE fail_count >= 3)   AS degraded_feeds
FROM feeds;
```

---

## Hosting on a Free VM (Zero Budget)

GitHub Actions is great for getting started, but for 25,000 feeds with
5-minute polling, you'll eventually want a dedicated VM.
Here are your zero-budget options:

### Option A: Oracle Cloud Free Tier (RECOMMENDED)

**Why Oracle?**
Oracle gives you an ARM VM with 4 CPUs and 24GB RAM — permanently free.
This is 10× more powerful than any other free tier and never expires.

**How to get it:**
1. Create an account at cloud.oracle.com (credit card required for verification,
   NOT charged)
2. Go to Compute → Instances → Create Instance
3. Choose: **Ampere A1 (ARM)** — select "Always Free" shape
4. Configure: 4 OCPUs, 24GB RAM (the maximum free allocation)
5. Choose Ubuntu 22.04
6. Download the SSH key pair when prompted (you'll need this to connect)

**Setting up the VM:**

```bash
# Connect to your VM
ssh -i ~/Downloads/your-key.pem ubuntu@your-vm-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.12
sudo apt install -y python3.12 python3.12-venv python3-pip git

# Install the pipeline
git clone https://github.com/your-org/news-pipeline.git
cd news-pipeline
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env
cp .env.example .env
nano .env   # Add your Supabase credentials

# Test it
python main.py --tier tier1_high --dry-run --verbose
```

**Setting up automatic scheduling with cron:**

```bash
# Open the crontab editor
crontab -e

# Add these lines (adjust paths to match your setup):
# Tier1: every 5 minutes
*/5  * * * * cd /home/ubuntu/news-pipeline && /home/ubuntu/news-pipeline/venv/bin/python main.py --tier tier1_high >> /var/log/pipeline-tier1.log 2>&1

# Tier2: every 15 minutes
*/15 * * * * cd /home/ubuntu/news-pipeline && /home/ubuntu/news-pipeline/venv/bin/python main.py --tier tier2_medium >> /var/log/pipeline-tier2.log 2>&1

# Tier3: every hour
0    * * * * cd /home/ubuntu/news-pipeline && /home/ubuntu/news-pipeline/venv/bin/python main.py --tier tier3_low >> /var/log/pipeline-tier3.log 2>&1
```

**Keeping the pipeline up-to-date automatically:**

```bash
# Add a daily git pull to your crontab
0 4 * * * cd /home/ubuntu/news-pipeline && git pull && /home/ubuntu/news-pipeline/venv/bin/pip install -r requirements.txt -q >> /var/log/pipeline-update.log 2>&1
```

**Monitoring logs:**
```bash
# Watch tier1 logs live
tail -f /var/log/pipeline-tier1.log

# Check last 50 lines
tail -50 /var/log/pipeline-tier1.log

# Search for errors
grep "ERROR\|CRITICAL" /var/log/pipeline-tier1.log | tail -20
```

**Preventing the VM from running out of disk (log rotation):**

```bash
# Install logrotate config
sudo nano /etc/logrotate.d/pipeline

# Add this content:
/var/log/pipeline-*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

---

### Option B: Google Cloud Free Tier

GCP gives you an e2-micro VM (2 vCPUs, 1GB RAM) permanently free in us-central1.

**Limitation:** 1GB RAM is tight for 25,000 feeds. Use MAX_CONCURRENT_FEEDS=5 in .env.

```bash
# In GCP Console → Compute Engine → Create VM
# Choose: e2-micro, us-central1, Ubuntu 22.04
# Setup is identical to Oracle above
```

---

### Option C: GitHub Actions (easiest, limitations apply)

**What you get free:**
- Public repos: unlimited minutes
- Private repos: 2,000 min/month (~67 min/day)

**The problem at scale:**
- tier1 (5min): ~576 min/day
- tier2 (15min): ~192 min/day
- tier3 (1hr): ~24 min/day
- Total: ~792 min/day — exceeds free private repo limit

**Solution:** Make the repository public.
Your code is not secret. Your Supabase credentials are in GitHub Secrets
(encrypted, never visible). Making it public costs you nothing.

---

## How to prevent pipeline failures on your VM

### Problem 1: Cron jobs that overlap
If a tier1 run takes longer than 5 minutes, the next one starts before
it finishes — causing duplicate inserts and DB overload.

**Solution:** Use `flock` to prevent overlapping runs:
```bash
# In crontab, wrap with flock:
*/5 * * * * flock -n /tmp/pipeline-tier1.lock /home/ubuntu/news-pipeline/venv/bin/python /home/ubuntu/news-pipeline/main.py --tier tier1_high >> /var/log/pipeline-tier1.log 2>&1
```
`flock -n` means "if another run is already running, skip this one".

### Problem 2: VM reboots
Oracle VMs occasionally restart for maintenance.

**Solution:** Make the cron job survive reboots (crontab already does this —
cron itself starts automatically on boot).

But if you want more reliability:
```bash
# Create a systemd service (runs at boot AND restarts on crash):
sudo nano /etc/systemd/system/pipeline-tier1.service

[Unit]
Description=News Pipeline Tier1
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/news-pipeline
ExecStart=/home/ubuntu/news-pipeline/venv/bin/python main.py --tier tier1_high
Restart=always
RestartSec=300

[Install]
WantedBy=multi-user.target

sudo systemctl enable pipeline-tier1
sudo systemctl start pipeline-tier1
sudo systemctl status pipeline-tier1
```

### Problem 3: Memory leaks
Long-running Python processes sometimes grow in memory over time.

**Solution:** The pipeline is designed to be short-lived (each run exits
after completing). Cron + flock handles this naturally — each 5-minute run
is a fresh Python process with fresh memory.

### Problem 4: Supabase connection limits
Supabase free tier has a max 60 simultaneous DB connections.
With 20 concurrent feeds each making 2-3 DB calls, you approach this limit.

**Solution:**
```bash
# In .env, reduce concurrency:
MAX_CONCURRENT_FEEDS=10
MAX_CONCURRENT_ARTICLES=5
```

### Problem 5: Knowing when something is wrong
Without monitoring, you won't know if the pipeline stopped working.

**Solution:** Check the pipeline_runs table daily:
```sql
-- Are runs happening? (should see rows every 5 min for tier1)
SELECT run_at, tier, new_articles, errors, duration_s
FROM pipeline_runs
ORDER BY run_at DESC
LIMIT 20;

-- Alert: no tier1 runs in the last 10 minutes
SELECT CASE
  WHEN max(run_at) < now() - interval '10 minutes' THEN 'ALERT: pipeline may be down!'
  ELSE 'Pipeline is running'
END AS status
FROM pipeline_runs
WHERE tier = 'tier1_high';
```

You can set up a free Supabase Edge Function to send you a Telegram/email
alert if no new runs appear — but that's a later-stage concern.

---

## FAQ for Beginners

**Q: What is asyncio and why do we use it?**
A: Python normally does one thing at a time. If we fetch 20 RSS feeds
   sequentially and each takes 2 seconds, that's 40 seconds. Asyncio lets
   Python switch between tasks while waiting for network responses — like
   a chef managing multiple dishes simultaneously. Result: 20 feeds in ~2 seconds.

**Q: What is a semaphore?**
A: A counter that limits how many things run at once. MAX_CONCURRENT_FEEDS=20
   means at most 20 feeds are being fetched simultaneously. Without this,
   we'd send hundreds of requests at once and get IP-banned by news servers.

**Q: Why store url_hash as a bigint instead of the URL string?**
A: Comparing two 8-byte integers is instant. Comparing two 200-character
   strings is slower, and running millions of comparisons per day adds up.
   The UNIQUE index on url_hash makes duplicate checking essentially free.

**Q: What does "bozo feed" mean in the logs?**
A: feedparser's term for a feed with minor XML formatting errors. Most bozo
   feeds still have readable entries — the pipeline uses them anyway.

**Q: Why does the pipeline mark articles is_duplicate=True instead of skipping them?**
A: Near-duplicates (same PTI wire story from 50 sources) have value:
   - They show which stories are widely covered (virality signal)
   - Your propensity scoring layer (Layer 6) needs this data
   - You can always filter is_duplicate=False in your app

**Q: How do I add a new paywalled domain?**
A: Edit pipeline/config.py → PAYWALLED_DOMAINS:
   ```python
   PAYWALLED_DOMAINS: frozenset = frozenset({
       "thehindu.com",
       "your-new-domain.com",  # ← add here
       ...
   })
   ```
   Or set has_paywall=True directly in the feeds table for that feed.

**Q: What's the difference between last_polled_at and last_success_at?**
A: last_polled_at updates on EVERY poll attempt (success or failure).
   last_success_at updates only when new articles are found.
   The dormancy check uses last_success_at — if no new articles in 30 days,
   the feed is dormant even if it's technically responding.
