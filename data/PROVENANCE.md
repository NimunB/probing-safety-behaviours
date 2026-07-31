# Data provenance

Every dataset this project reads, where it came from, and how it is frozen.

**Why vendored rather than fetched:** the upstream `deception-detection` repo is 535 MB; these
three files total 500 KB. More importantly, nothing pinned the clone — an upstream edit to
`roleplaying/dataset.yaml` would silently change the thesis's data, and the activation cache is
content-blind (keyed on dataset *name* and example count, not content), so it would not even
recompute. Vendoring freezes the data and removes a network dependency from the pipeline.

---

## Vendored here (500 KB total)

Copied from **ApolloResearch/deception-detection** at commit
`f8ec4010e74927394709dffa22b97bdf8cd5a62f` ("Fix links in README.md", 2025-02-06),
<https://github.com/ApolloResearch/deception-detection>.

| File | Bytes | SHA-256 | Used for |
|---|---:|---|---|
| `repe/true_false_facts.csv` | 35,976 | `924709159b40a30d…` | Deception instructed pairs [Apollo §3.2.1] — 306 true facts |
| `how_to_catch_an_ai_liar/goal_directed_lying.json` | 26,814 | `fbb3ca465c559b14…` | Deception OOD #1, AI Liar (Pacchiardi et al.; [Apollo App. A.1]) — context-defined |
| `roleplaying/dataset.yaml` | 444,782 | `7d3e36dc9a2fa72f…` | Deception OOD #2, Roleplaying — content-defined (plan #35) |

Full digests:

```
924709159b40a30d5b51488e9ccd7fd886456a89f0e0f5849eacefba6ff55312  repe/true_false_facts.csv
fbb3ca465c559b14848a34c3a476a761fb0f8e910431d2968f797cdd06c1e7a2  how_to_catch_an_ai_liar/goal_directed_lying.json
7d3e36dc9a2fa72ff823b651c2ade2e0fc34c5bc6267cdf9b7be8b046ed2e564  roleplaying/dataset.yaml
```

Verify after any move or re-clone:

```bash
cd data && sha256sum -c <<'EOF'
924709159b40a30d5b51488e9ccd7fd886456a89f0e0f5849eacefba6ff55312  repe/true_false_facts.csv
fbb3ca465c559b14848a34c3a476a761fb0f8e910431d2968f797cdd06c1e7a2  how_to_catch_an_ai_liar/goal_directed_lying.json
7d3e36dc9a2fa72ff823b651c2ade2e0fc34c5bc6267cdf9b7be8b046ed2e564  roleplaying/dataset.yaml
EOF
```

**Licensing:** these files are redistributed from Apollo Research's repository. Check its licence
before publishing this repo publicly; if redistribution is not permitted, replace the vendored
copies with a fetch script pinned to the same commit (`git clone` + `git checkout f8ec4010`).

---

## Not vendored — fetched from HuggingFace, pinned by revision

**`Arrrlex/models-under-pressure`** — the high-stakes train set [HS §2.4] and the
`anthropic_hh_balanced` OOD split [HS Table 2].

* Revision in use: **`940d830f639617752f54fdf2bca0da7e0d856a9e`** (the `main` ref as resolved on
  this box, 2026-07-21).
* Size: ~33 MB in the HF hub cache plus ~27 MB processed — small enough to commit, but a HF
  dataset with a stable id and a pinnable revision is better fetched than copied.
* `datasets.py` passes this revision explicitly, so a fresh instance resolves the same data even
  if `main` moves.

ToolACE was part of the high-stakes OOD evaluation until 2026-07-19 and was removed (plan #34);
it is intentionally not fetched.

---

## Model weights

`meta-llama/Llama-3.1-8B-Instruct`, loaded in 8-bit, ~15 GB in the HF cache. Never vendored —
re-download on each new instance. Requires `HF_TOKEN` (gated repo).

## Sycophancy factorial data (added 2026-07-31, sycophancy_plan v2)

Source: `github.com/cincynlp/disentangle-sycophancy` at commit
`217b89808d07e978a7c8fac37e8ff363e29ad8ef` (data released with arXiv 2509.21305,
"Sycophancy Is Not One Thing"). **No license file in the source repo** — same caveat class as the
Apollo files: check before making this repo public. Train domains: cities_pos, sp_en_trans,
claims, companies; OOD domain: math.

```
sha256sum -c <<'SUMS'
655e64d88e5bddaa0a040d13cc01e4e33ac244c45f89be16ebf5b7fe5020bf28  data/sycophancy/cities_pos_factorial.json
12a629f0cb20ad4ffa0b9f11fa49256ba4e1ecab45f55051e546a4ebd6c10ade  data/sycophancy/claims_factorial.json
925a95216670258564102777344e491dea676a15bc876dafe85a0020631e2766  data/sycophancy/companies_factorial.json
b5516e5462f8de5c07eaed13327a61de21c5e5821852ff97f59d362bcb69eac8  data/sycophancy/math_factorial.json
78d8c89addf2520e72eb21ec739eb1850d97760270301f681f8bf689eaeee946  data/sycophancy/sp_en_trans_factorial.json
SUMS
```
