# Accuracy: what each enrichment step is measured to do

Every component's accuracy, with the method behind it. No number here is an estimate. Each
was measured on the evaluation set named alongside it, and can be re-run with the command
given. When a model or rule changes, re-run its evaluation and update this page in the same
change.

Literature review behind the roadmap (2026-10-06): clustering, classification, NER, linking,
sentiment, summaries and evaluation methodology, with citations. The ranked upgrade plan is
at the end of this page.

## Language identification — `enrichment/steps/language.py`

**Method.** 3,030 headlines from Indian news sitemaps (2026-10-06), each labelled with the
language its publisher declares (`<news:language>`). At most 300 per language and 40 per
sitemap. Headlines are the hard case because they are short.

**Columns.**
- *Acc* is against the publisher's label.
- *Script-consistent* counts only headlines whose script agrees with that label. Publisher
  labels are noisy: Jagbani declares Hindi but writes Punjabi; some Telugu outlets label
  their English edition `te`.

Re-run: `python tools/language_eval/evaluate.py`.

| Language | lingua only (before) | Script-first (now) | Now, script-consistent |
|---|---|---|---|
| Hindi | 89.0 % (25 → Marathi) | 97.3 % | 99.7 % |
| Marathi | 95.0 % | 94.3 % | 94.6 % |
| Bengali | 90.0 % | 90.0 % | 100 % |
| Assamese | **0 %** (all → Bengali) | 98.8 % | 98.8 % |
| Tamil | 97.3 % | 97.7 % | 100 % |
| Telugu | 97.3 % | 99.3 % | 100 % |
| Kannada | **0 %** (no answer) | 99.2 % | 100 % |
| Malayalam | **0 %** (no answer) | 87.7 % | 100 % |
| Gujarati | 99.5 % | 100 % | 100 % |
| Punjabi | 94.2 % | 94.2 % | 100 % |
| Odia | **0 %** (→ Welsh) | 100 % | 100 % |
| Urdu | 100 % | 100 % | 100 % |
| English | 88.7 % | 91.7 % | 97.5 % |
| **Overall** | **71.1 %** | **95.5 %** | **99.1 %** |

How it works:
1. **Script first.** Seven languages own their script.
2. **Shared scripts.**
   - Bengali script: ৰ/ৱ mean Assamese.
   - Devanagari: Marathi vs Hindi is decided by function words and Marathi inflections. One
     of those is the genitive -चा/-ची/-चे after a vowel sign, which keeps the Hindi चर्चा out.
3. **Everything else goes to lingua**, restricted to 30 plausible languages. That removed
   absurd guesses on short Latin headlines (Tagalog, Nynorsk, Swahili, Welsh).
4. **Romanised Hindi** is returned as `hi-latn` and kept away from the en/hi NER, sentiment
   and keyword models.

Remaining errors:
- Mostly label noise.
- Marathi headlines with no inflection to go on, which default to Hindi. Article bodies are
  long enough that this is rare there.

## Story clustering — `enrichment/memory_clustering.py` ≡ `wizer_assign_cluster`

See `docs/CLUSTERING.md`:
- **Calibration:** 400 labelled en/hi pairs; strict pair F1 0.675 at join 0.85 / anchor 0.75.
- **Parity:** the in-memory clusterer reproduces the SQL function decision for decision
  (`tests/test_clustering_sql.py::TestMemoryParity`).

## Category — `enrichment/steps/classifier.py`

**13-language gold set (2026-10-06), `tools/gold/`.**
- **Sample:** 1,400 production headlines, with descriptions when present: 150 en, 150 hi and 100 for
  each other language, at most 6 per publisher.
- **Labels:** two independent LLM labelling passes under `tools/gold/CATEGORY_RUBRIC.md`.
- **Agreement:** 94.7 % (Cohen's κ 0.941). Per language κ runs from 0.857 (ur) to 0.969 (hi).
- **Disagreements:** the 74 were settled by a third pass that saw both labels.
- **Scoring:** 20 items both passes marked `cant_tell` are excluded.

Re-run:
- `python tools/gold/evaluate.py agreement | build`
- `python tools/gold/evaluate.py score pred_<model>.json`

| Model | Accuracy (1,380) | hi | en | bn | gu | or | ur | mr | kn | as | ta | pa | te | ml |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mDeBERTa zero-shot (before 2026-10-07) | 51.9 % | 52 | 54 | 49 | 42 | 51 | 43 | 60 | 51 | 46 | 63 | 59 | 54 | 49 |
| E5 linear head, 4,101 labels | 76.7 % | 77 | 77 | 77 | 74 | 81 | 83 | 81 | 83 | 77 | 72 | 71 | 72 | 73 |
| **E5 linear head, 7,056 labels (shipped 2026-10-07)** | **77.9 %** | 79 | 78 | 77 | 75 | 80 | 86 | 81 | 85 | 82 | 72 | 70 | 72 | 76 |

**The E5 head** (`enrichment/steps/category_head.py`, model `enrichment/models/category_head.npz`):
- **Model:** a 13-way logistic regression on the multilingual-E5 vector the pipeline already
  computes for clustering.
- **Training data:** fitted on 7,056 LLM-teacher labels, one pass each under the same rubric:
  - `train_sample.json` + `labels/T*.json`: 4,101 across 13 languages;
  - `train2_sample.json` + `labels/U*.json`: 2,955 en/hi.

  They are drawn from production with no URL or headline shared with the gold set.
- **C:** chosen by publisher-held-out cross-validation on the training data (78.3 % at C = 8). The
  gold set is never used for fitting or tuning.
- **Ceiling:** the two gold labelling passes agree 94.7 %, so that is roughly the most any model
  can score here.
- **Cost:** one 13×768 matrix product instead of 13 NLI passes.

Re-run: `python tools/gold/embed.py train_sample.json train2_sample.json category_sample.json`, then
`python tools/gold/train_head.py`.

**Since 2026-10-07 the pipeline keeps English and Hindi only**, so en/hi accuracy is what counts.
Training on all 13 languages beats training on en+hi alone, because the shared multilingual vector
space transfers. On the 300 en/hi gold items:

| Training data | en/hi accuracy |
|---|---|
| all 4,101 labels (first round) | **76.6 %** |
| the 1,382 en/hi labels only | 71.2 % |
| all labels, en/hi weighted ×3 | 75.3 % |
| all 7,056 labels (+2,955 en/hi, shipped) | **78.5 %** (en 78, hi 79) |

Its most common errors (gold → predicted):
- general → world: 22
- crime → world: 21
- general → business: 16
- business → general: 15
- general → politics: 14

These are mostly the rubric's own borderline cases: foreign crime, and "general" against a specific
topic.

The shipped zero-shot classifier is far below the 66.2 % measured on the old 160-item en/hi set.
Its most common errors (gold → predicted):
- politics → crime: 39
- general → crime: 36
- world → politics: 35
- general → business: 31
- general → entertainment: 28

A linear head on the E5 vector, cross-validated on the gold set alone with publishers held out,
reaches 74.3 %. That is the measured case for roadmap item 1.

## Topic tags — `enrichment/steps/tag_head.py` (`articles.ai_tag`)

**Gold set (2026-10-07), `tools/gold/tags.py`.**
- **Sample:** the 300 en/hi items of the category gold set.
- **Labels:** each item was tagged twice, independently, under `tools/gold/TAG_RUBRIC.md` (the
  20 existing tags, 0–3 per article).
- **Agreement:** identical tag sets on 92.3 % of items; pass-vs-pass micro-F1 0.954; per-tag κ
  0.79–1.00.
- **Disagreements:** the 23 were settled by a third pass that saw both answers.
- **Training data:** 4,365 en/hi teacher labels (`labels/TT*.json`), with no overlap with the
  gold set.
- **Tuning:** each tag has its own threshold, chosen on out-of-fold, publisher-held-out
  predictions. C = 8 was chosen by CV micro-F1 (0.770).

Re-run: `python tools/gold/tags.py agreement | build | train`

| Tagger | micro-F1 | precision | recall | macro-F1 | exact tag set |
|---|---|---|---|---|---|
| mDeBERTa zero-shot (before 2026-10-07) | 0.367 | 0.274 | 0.558 | 0.354 | 26.7 % |
| **E5 per-tag heads (shipped)** | **0.735** | **0.699** | **0.775** | **0.675** | **62.3 %** |

The zero-shot tagger was wrong on about three of every four tags it assigned. The heads also
remove mDeBERTa from enrichment altogether. On 120 real hand-off articles (4 threads), the cost
per article fell from **1.76 s to 0.31 s**; the tagger alone had been 81 % of it.

Weak tags in the gold set (small support): `financial markets` (4 items), `public health` (2) and
`science` (7). Their F1 is not yet reliable in either direction.

## Roadmap (ranked by the 2026-10-06 literature review)

| # | Step | Expected effect |
|---|---|---|
| 0 | 13-language gold sets (LLM-labelled, human spot-checked), B-cubed for clustering, CI gates, weekly audit | Makes every step below decidable |
| 1 | ~~Category / tags as linear heads on the e5 vector, trained on LLM-teacher labels~~ | **Done 2026-10-07**: category 51.9 → 77.9 %, tags F1 0.367 → 0.735, 1.76 → 0.31 s/article. News-tone head not done |
| 2 | ~~Script-first language ID~~ | **Done**: 71.1 % → 95.5 % |
| 3 | Learned cluster scorer (dense + lexical + entity + time features) | +3–8 strict F1 (literature: +5–11) |
| 4 | IndicNER (11 languages) + Wikidata alias-table entity linking | Indic NER from ~0 to 72–83 F1 |
| 5 | Embedding A/B: e5-large-instruct vs e5-base (ONNX int8) | Decided on the 13-language gold set |
| 6 | ONNX int8 for e5-base | **Measured 2026-10-06, not adopted yet** (`tools/cluster_eval/onnx_int8.py`): 2.6× faster (41 → 16 ms/headline, 4 threads) and strict AUC 0.9297 → 0.9272, so ranking quality is essentially unchanged. But 33 of 400 calibrated join decisions flip at 0.85 (mean similarity shift 0.016; 105 pairs lie within ±0.02 of the threshold). Adopting it needs re-calibrated thresholds and a separate model id, so int8 and fp32 vectors never share a cluster. It is a planned cutover, worth it only if runner CPU, not the database, limits clustering throughput |
