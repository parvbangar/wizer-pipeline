# Story clustering (Layer 2, step 10)

A **story cluster** is one real-world news event and its immediate follow-ups, told by any
number of outlets in any language. A PTI wire story reprinted by 50 sites is *one* cluster
with `outlet_count = 50`. That number is the pipeline's main virality signal. It is also the
unit the app shows readers, instead of the same story 50 times.

This document covers the algorithm, why it is built this way, the calibration evidence
behind every default, and how to operate and tune it.

---

## 1. Where it runs

```
cluster.py run   (every 30 min + after each ingestion run, .github/workflows/cluster.yml)
     │  pages through articles WITHOUT a cluster, oldest first   (wizer_fetch_unclustered)
     │  embeds each page                                          (enrichment/steps/embedding.py)
     └▶ enrichment/clustering.assign_clusters_batch ──RPC──▶ wizer_assign_cluster_batch()
                                   50 articles per round trip → wizer_assign_cluster() each,
                                   one transaction: lock → find → decide → write

enrich.py        (NLP enrichment) clusters any article it enriches that the job hasn't
                 reached yet — same SQL function, so whichever path comes first wins and
                 the other gets `existing`.

cluster.py maintain   (every 3 h, cluster_maintenance.yml)  merge twins · reconcile · prune
cluster.py backfill   `run` with a 72 h look-back, for catching up
cluster.py report     health, top stories, queue, recent runs
```

**Why clustering is its own job.** In production, ingestion produced **~50,000
articles a day**, while NLP enrichment managed **~2,600** (≈ 16 s of CPU per
article on a runner, almost all of it in the zero-shot classifier). Story size —
`outlet_count` — is only meaningful if *every* outlet's copy of a story is
clustered, not the 5 % that get enriched. Embedding a headline costs ~0.1–0.3 s,
so the clustering job keeps up with all of ingestion on one runner.

**Every article is clustered** — every language (not just the en/hi "rich" set),
crawled or not, short wire briefs included (a 40-word brief is exactly the
syndication `outlet_count` exists to measure). A clustering failure never costs an
article anything else; it is retried on the next run.

## 2. The algorithm

### 2.1 Article representation

`build_embedding_text()` sends the model **the headline, then the description**. When the
description is missing, too short, or just repeats the headline, it sends the **first
sentences of the body** instead. The text is capped at 400 characters and cut on a word
boundary. News is written as an inverted pyramid, so the who/what/where sits at the top.
Deeper body text adds background paragraphs that make *different* stories on the same
subject look alike.

Embedding model: **`intfloat/multilingual-e5-base`**, which produces 768 dimensions,
covers 100 languages including every major Indian language, and is used without the
`"query: "` prefix. Section 4 explains the choice. The whole claimed batch is embedded in
mini-batches of 32, about 4–6× faster on CPU than one call per article.

### 2.2 Assignment — `wizer_assign_cluster()`

Each call runs in **one transaction holding a global advisory lock**. Concurrent runners
serialise only on this step, which takes about 3–4 ms per article, while their NLP work
stays parallel.

1. **Idempotency.** If the article already has a `cluster_id`, return `existing` and change
   nothing. A retry after a failed save, or a `--force` re-enrichment, can never
   double-count an article.
2. **Candidates.** Search the active clusters built with the same embedding model whose time
   span is compatible with the article's `published_at`:
   - `first_seen - gap ≤ t ≤ last_seen + gap`: the story is still "live" (gap = 18 h)
   - `max(last, t) - min(first, t) ≤ span`: a story never spans more than 120 h

   The search is an **exact** cosine search over that window. It is not an HNSW scan: an
   HNSW scan with a `WHERE` clause post-filters a fixed-size candidate list and silently
   drops matches when many out-of-window clusters are nearer. The 5 clusters whose
   centroids are nearest become candidates.
3. **Scoring: average-link.** Each candidate is scored by the **mean cosine similarity
   between the article and every member**. That is computed exactly, in O(1), from the
   stored sum of member vectors: `q · Σm / n`.
4. **Decision.** The first candidate in score order that passes wins:

   | condition | action |
   |---|---|
   | `anchor_sim ≥ 0.75` (similarity to the cluster's *seed* article), always required | |
   | `avg_link ≥ 0.85` | `join` |
   | `avg_link ≥ gray` **and** (≥ 2 shared salient entities **or** a near-identical top image from a *different* outlet) | `gray_join` |
   | nothing passes | `seed` (new cluster) |

   The gray zone ships **off** (`gray = join`); see section 4.4.
5. **Write.** Add the unit vector to `centroid_sum` and refresh the normalised `centroid`
   (`halfvec`). Increment `article_count`. Add the domain to `outlet_set` and recompute
   `outlet_count`, which counts distinct domains. Merge `top_entities`, `entity_set`,
   `image_hashes` and `language_set`. Widen the time span with LEAST/GREATEST. Swap the
   **representative** article and the headline if the new article is closer to the new
   centroid than the current representative. Finally, stamp `articles.cluster_id`,
   `cluster_similarity`, `cluster_assignment` and `clustered_at`.

Zero and NaN embeddings are rejected outright. Postgres orders NaN *above* every number,
so `NaN >= threshold` is true: a broken vector would join the nearest cluster and corrupt
its centroid.

### 2.3 Maintenance — `cluster.py maintain`

Online assignment is greedy and order-dependent. If the first two outlets' versions of a
story arrive with unusually different headlines, they seed two clusters. Every 3 hours:

1. **Merge twins.** For each active cluster touched in the last 6 h, find its nearest
   compatible cluster: same model, compatible time spans, anchors agree at ≥ 0.75. Merge
   when their **cluster-to-cluster average-link similarity** is ≥ **0.845**, computed
   exactly as `Σa·Σb / (na·nb)`. The larger cluster wins. Each cluster takes part in at
   most one merge per round, and rounds repeat until nothing changes. The loser becomes a
   **tombstone** (`status = 'merged'`, `merged_into = winner`), so cached ids still resolve.
2. **Reconcile.** Recount `article_count` and `outlet_*` from the member articles. This
   returns 0 in normal operation. A non-zero result means membership changed outside the
   assignment path, for example Layer 1's table pruning deleted old articles.
3. **Prune.** Delete clusters that no article points at once they have been idle for a
   week. That covers pruned articles, aged-out tombstones, and v1 legacy rows.

---

## 3. What the removed v1 got wrong

Clustering was deleted in `374dfe0` (2026-04-07). v2 addresses each failure mode:

| v1 behaviour | consequence | v2 |
|---|---|---|
| nearest cluster found in SQL, decision and counters in Python | two shards and overlapping `workflow_run` triggers lost counter updates and seeded twin clusters | find → decide → write in one transaction under an advisory lock; concurrency test with 8 simultaneous runners |
| LaBSE at a fixed 0.82 cosine | **recall 9.5 %** on labelled same-story pairs (§4): almost every article became a singleton | E5-base + average-link at 0.85: recall 0.64–0.72 |
| running mean, re-normalised after every join | not the true mean; drifted | exact running **sum** (cosine is scale-invariant, so the sum *is* the mean for search) |
| centroid similarity decisions | big clusters' generic centroids absorb neighbouring stories ("chaining") | average-link decisions + anchor guard |
| `last_seen_at` overwritten with the article's time | the queue runs newest-first, so clusters moved *backwards* and fell out of the window | LEAST/GREATEST; window relative to the article's own time, so backfills work |
| `CLUSTER_WINDOW_HOURS = 0` (no window) | a story could absorb articles months later | 18 h gap + 120 h span cap |
| HNSW + `WHERE` filter | silent recall loss | exact search over the window |
| duplicates never reconciled | permanent twin clusters | merge sweep with tombstones |
| headline = longest title | clickbait-length headlines won | representative = most central member |

---

## 4. Calibration

All numbers come from `tools/cluster_eval/`, run on **1,245 live headlines** (1,001 English,
244 Hindi) from 17 Indian outlets, fetched 2026-10-03.

**Labelled pairs.** 400 pairs were drawn by TREC-style pooling: every article's top-4
neighbours under *both* candidate models, stratified by per-model similarity decile so that
neither model is judged only on pairs it chose itself. Each pair was hand-graded:
**2** = same specific event, **1** = same broader running story (for example, two different
developments in the flydubai cockpit-attack story), **0** = different. The grade counts are
74 / 37 / 289; 53 pairs are cross-lingual.

- *Strict* metrics treat only grade 2 as positive.
- *Loose precision* counts a same-cluster pair as correct when its grade is 1 or 2.

The pooled set deliberately over-samples hard negatives, so absolute precision in
production will be higher than shown. The numbers are for *comparing* designs.

### 4.1 Embedding model (pair level)

| model / input | ROC-AUC | best F1 (threshold) | cross-lingual AUC |
|---|---|---|---|
| LaBSE, title + description | 0.897 | 0.663 (0.57) | 0.967 |
| LaBSE **at v1's 0.82 threshold** | | P 0.875 · **R 0.095** | |
| E5-base, `"query: "` prefix | 0.915 | 0.698 (0.88) | 0.970 |
| **E5-base, no prefix** | **0.930** | **0.725 (0.85)** | |
| E5-base, title only | 0.901 | 0.715 (0.88) | |
| E5 + lexical Jaccard hybrid | ≤ 0.914 | ≤ 0.690 | |
| gte-multilingual-base | failed to load (needs `trust_remote_code`; rejected) | | |

At the best E5 threshold, 26 of the 42 "false positives" are grade-1 pairs: same running
story, different development. The model's mistakes are about granularity, not unrelated
stories.

### 4.2 Decision rule (end-to-end, real SQL)

`tools/cluster_eval/simulate.py` runs all 1,245 articles through the production code path:
production embedding text, production NER and parameters, and `wizer_assign_cluster` on a
local Postgres 18 with pgvector 0.8.1. It then scores the clusters against the labelled
pairs. Articles are processed in time order unless stated otherwise.

| rule | strict P | strict R | strict F1 | loose P | clusters | singletons | multi-outlet |
|---|---|---|---|---|---|---|---|
| centroid cosine ≥ 0.88 | 0.607 | 0.730 | 0.663 | 0.809 | 802 | 87 % | 94 |
| centroid cosine ≥ 0.85 | 0.306–0.485 | | ≤ 0.550 | ≤ 0.722 | | | (chaining) |
| **average-link ≥ 0.85** | 0.653 | 0.635 | 0.644 | **0.889** | 739 | 75 % | **157** |
| average-link ≥ 0.85 **+ merge at 0.845** | 0.639 | **0.716** | **0.675** | 0.880 | 697 | 74 % | 147 |
| same, processed newest-first | 0.589 | 0.716 | 0.646 | 0.844 | 697 | 74 % | 145 |

Per label grade, the shipped configuration (average-link 0.85 + merge 0.845, time order)
places together:

| pair grade | placed in the same cluster |
|---|---|
| 2 — same specific event | **53 / 74 (72 %)** |
| 1 — same running story, different development | 20 / 37 (54 %) |
| 0 — different stories (pooled hard negatives) | **10 / 289 (3 %)** |

Average-link finds 67 % more multi-outlet stories than the best centroid rule, at a higher
loose precision. The merge sweep recovers recall lost to arrival order. That is also why
the runner **processes each claimed batch oldest → newest**, even though the queue hands out
the newest articles first.

### 4.3 Thresholds swept

- **join** 0.80 … 0.87: 0.85 maximises strict F1 for average-link. Below 0.84, loose
  precision falls under 0.80.
- **anchor** 0.70 / 0.76 / 0.80: 0.70 and 0.76 give identical results, and 0.80 costs
  recall. It is set to **0.75** as a guard rail that doesn't bind on day-scale data but
  stops long-running clusters drifting.
- **merge** 0.80 … 0.86: 0.845 maximises strict F1 (0.675). At 0.83 and below, merging
  starts fusing related-but-different stories (loose P 0.81 → 0.61).

### 4.4 Gray zone: built, tested, shipped off

Entity and image evidence lets a pair a little below the join threshold join anyway. On
headline-level entities the best gray configuration raised strict F1 by **+0.004** and
cost **4 points of loose precision**, so the default is `CLUSTER_GRAY_THRESHOLD = CLUSTER_JOIN_THRESHOLD`
(off). The mechanism stays in place and is covered by the SQL integration tests. Production
NER runs on full article text (richer entities) and image hashes exist there, which the
calibration set lacks. Re-run the calibration with production data before turning it on.
Image evidence only counts from a *different* outlet, and blank-image hashes (0 / −1) are
ignored. Same-outlet matches are usually a logo or placeholder.

### 4.5 Known limitations

- **Granularity is "event".** Developments within a long-running story, such as a pilot's
  phone call with the PM against the co-pilot's identity being revealed, often form separate
  clusters. Pairs graded as the same running story land together 54 % of the time.
- **Live blogs and round-ups** ("Asian Games Day 15 LIVE") mention many events, so they
  attract members from several stories. Average-link limits this but does not eliminate it.
- **Recurring templated items** ("Sensex closes higher") published at the same time on
  consecutive days stay separate because of the 18 h gap. Same-day repeats can merge.
- Calibration used headline + RSS description. Production embeds headline + description,
  or headline + body lead, which is equal or richer input.

---

## 5. Data model

`article_clusters` keeps all v1 columns and adds:

| column | meaning |
|---|---|
| `centroid_sum vector(768)` | exact sum of member unit vectors, used for average-link and merges |
| `centroid halfvec(768)` | normalised centroid, the search key (fits in the heap page, no TOAST) |
| `anchor halfvec(768)` | seed article embedding (drift guard) |
| `representative halfvec(768)`, `representative_article_id` | the current most central member |
| `embedding_model` | vector space id; clusters never match across models |
| `status` | `active` / `merged` (tombstone) / `legacy` (v1, never searched) |
| `merged_into` | tombstone pointer, always to a live cluster (chains are re-pointed) |
| `image_hashes`, `language_set`, `gray_join_count` | evidence and diagnostics |

`articles` adds `cluster_similarity`, `cluster_assignment` (`seed` / `join` / `gray_join`) and
`clustered_at`. Monitoring views are `cluster_health` (24 h one-row dashboard) and
`top_stories_24h`.

---

## 6. Operating it

```bash
python cluster.py report --top 20         # health + top stories + queue + recent runs
python cluster.py backfill --hours 72     # cluster anything enriched while clustering was off
python cluster.py maintain --dry-run      # list the merges the sweep would make
```

Healthy signs: `cluster_health.clustered_pct` close to 100, `seed_pct` 60–80 %, `merges_24h`
in the tens rather than the hundreds, and `reconcile` returning 0.

**Changing the model** (`CLUSTER_EMBEDDING_MODEL`): new articles start fresh clusters in the
new vector space. Old clusters are never matched across models and age out of the window
within a day. Re-run `tools/cluster_eval/` first; thresholds are model-specific (E5 packs
similarities into roughly 0.7–1.0, LaBSE spreads them over roughly 0.3–1.0).

**Re-calibrating:**

```bash
python tools/cluster_eval/fetch_headlines.py        # fresh live sample
python tools/cluster_eval/calibrate.py embed        # embed with the candidate models
python tools/cluster_eval/calibrate.py pool         # sample pairs to label → labels/
python tools/cluster_eval/calibrate.py evaluate     # pair-level comparison
python tools/cluster_eval/simulate.py --sweep       # end-to-end threshold grid (needs local Postgres)
python tools/cluster_eval/simulate.py --merge 0.83 0.845 0.86
```

---

## 7. Tests

- `tests/test_clustering.py`: embedding text, encode contract, evidence selection, merge
  planning, and the RPC parameter contract (the Python dict keys must match the SQL
  signature exactly).
- `tests/test_clustering_sql.py`: real Postgres + pgvector. Covers seed/join/idempotency,
  exact average-link maths, a constructed chaining case, anchor, gap/span windows,
  out-of-order spans, gray zone (entities, cross-outlet images), NaN/zero/dimension guards,
  **8-way concurrency**, merges and tombstones, reconcile, prune, the claim queue, grants,
  migration idempotency, and PostgREST argument decoding.
- `tools/e2e_local.py`: the real runner with the real models against local Postgres. Last
  run: 300 articles, 2 concurrent runners, 0 double claims, 0 count drift, about 1.6 s per
  article with all models loaded.
