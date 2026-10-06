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

mDeBERTa zero-shot; 66.2 % on the 160-item en/hi gold set (`tools/classifier_eval`). That set
is too small to detect changes under ~8–10 points and covers 2 of 13 languages. Building the
13-language gold set is the next accuracy step (roadmap below).

## Roadmap (ranked by the 2026-10-06 literature review)

| # | Step | Expected effect |
|---|---|---|
| 0 | 13-language gold sets (LLM-labelled, human spot-checked), B-cubed for clustering, CI gates, weekly audit | Makes every step below decidable |
| 1 | Category / tags / news-tone as linear heads on the e5 vector, trained on LLM-teacher labels | Category 66 % → ~76–84 %, and ~5 s/article of CPU freed |
| 2 | ~~Script-first language ID~~ | **Done**: 71.1 % → 95.5 % |
| 3 | Learned cluster scorer (dense + lexical + entity + time features) | +3–8 strict F1 (literature: +5–11) |
| 4 | IndicNER (11 languages) + Wikidata alias-table entity linking | Indic NER from ~0 to 72–83 F1 |
| 5 | Embedding A/B: e5-large-instruct vs e5-base (ONNX int8) | Decided on the 13-language gold set |
| 6 | ONNX int8 for e5-base | −65 % embedding time; re-calibrate |
