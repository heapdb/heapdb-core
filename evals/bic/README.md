# Bug-inducing commit ranking

Given a failure (a bug report and failing tests) or a fix under review, rank
the earlier commits by how likely each one introduced the bug. This is
HeapDB's core question ("which earlier event most likely explains this?")
applied to code changes. The benchmark is public and reproducible: it
downloads pinned inputs, checks them by sha256, runs every method under the
same time-ordered protocol and generates [RESULTS.md](RESULTS.md) from
[results.json](results.json). It reports where HeapDB loses as well as where
it wins.

## Datasets

### Survey (October 2026, checked from this sandbox)

| candidate | what it really contains | licence | size | reachable | used |
|---|---|---|---|---|---|
| Defects4J + FONTE labels (Kim et al., ICSE 2023; extension An et al., arXiv:2502.12922) | [coinse/fonte](https://github.com/coinse/fonte): BIC labels for Defects4J bugs (130 on `main`, **206 on branch `extension`**), bug reports, failing tests with stack traces, coverage matrices, per-method commit histories, style-change checks, every java-touching commit before the buggy revision (message, diff, time), and the released per-bug rankings of Bug2Commit and FBL-BERT | MIT | 285 MB (depth-1 bare clone) | yes (GitHub) | **primary** |
| InduceBenchmark (Wen et al., FSE 2019) | [justinwm/InduceBenchmark](https://github.com/justinwm/InduceBenchmark): fix/BIC pairs for 6 Apache projects + 91 Defects4J bugs (the source of FONTE's 67 Wen labels), SZZ outputs, coverage files | no licence file | ~14K files | yes | only through FONTE's curated subset |
| Locus (Wen et al., ASE 2016) | [justinwm/Locus](https://github.com/justinwm/Locus) is the tool's Java source only; the bug reports and change oracles are not in the repository | no licence file | 52 files | code only | no (no data reachable) |
| Developer-informed SZZ oracle (Rosa et al., ICSE 2021) | [grosa1/icse2021-szz-replication-package](https://github.com/grosa1/icse2021-szz-replication-package): 1,930 fix -> BIC instances over many GitHub repositories (1,115 language-filtered), labels from developers' own references | no licence file in the repo (Zenodo record) | 17 files + hundreds of repos to clone | yes | no: fix-given SZZ task with no failure/report inputs and no published ranking baselines, and it needs hundreds of clones |
| `Fixes: <sha>` trailers | any project with the convention; Linux has ~76K pairs but is too large to clone here. **DPDK** ([DPDK/dpdk](https://github.com/DPDK/dpdk)) makes the trailer mandatory for fixes: 45,904 commits, 14,946 fixes with a trailer | BSD-3-Clause (we use metadata only) | 192 MB bare clone | yes | **secondary** |
| Newer work (2025-2026) | LLM4SZZ (Tang et al., arXiv:2504.01404) and SZZ agents (Risse & Böhme, arXiv:2603.29378): LLMs given the fix, evaluated by set-selection F1 on Linux `Fixes:` pairs and GitHub/Apache samples | various | - | yes (papers, code) | no: different task output (F1, not ranking) and LLM inputs; they confirm `Fixes:` tags as developer-informed labels |

**Primary: Defects4J with FONTE's 206 labels.** It is the only reachable
dataset with labelled BICs *and* the failure-side inputs (reports, stack
traces, coverage) *and* published ranking baselines whose per-bug outputs
are released. So Bug2Commit and FBL-BERT can be compared bug by bug, on the
same candidates, with paired confidence intervals, and FONTE can be
re-implemented and checked against its released ranks.

**Secondary: DPDK `Fixes:` trailers.** Large-scale (14,913 fix -> BIC
cases), developer-written labels from git metadata alone. This is the
"suggest the `Fixes:` tag for this fix under review" setting (the SZZ
setting). Labels are noisier than curated ones, and the task gives the
methods the fix itself, not a failure.

## Task

Each bug is an effect at a time *t*; the candidates are commits before *t*.

| track | effect / query | candidates (window) | BIC outside? |
|---|---|---|---|
| `d4j/report` | bug report + failing tests (names, exceptions, stack frames) | every commit before the buggy revision that touched a `.java` file (the set Bug2Commit and FBL-BERT ranked; median 948) | never (whole history); RESULTS.md shows what a 7 d ... 5 y time window would lose |
| `d4j/fl` | the same + FL: Ochiai scores from the coverage matrix and per-method history depth (FONTE's inputs) | FONTE's C_BIC: commits that changed code covered by a failing test, minus style-only changes | reported (BIC filtered out by C_BIC) |
| `dpdk/fixes` | fix message with every trailer and hex id removed + fix author + files the fix touches | earlier commits sharing at least one file with the fix, newest 300 | reported: no shared file, or older than the newest 300 |

The Defects4J effect time is the newest candidate commit (the artifact has
no report dates); DPDK uses the fix's commit time.

**Metrics.** Acc@1/3/5/10, MRR (a BIC outside the candidates counts 0),
mean rank (over bugs whose BIC is a candidate), and the paired bootstrap
ΔMRR with a 95% CI (10 000 resamples over bugs, seed 7). Ties in score are
broken newest-first for every method (FONTE's papers use the "max" rule;
that rule is used only for the reproduction checks).

**Equal inputs.** `signals.py` builds one table per (bug, candidate): commit
time and order, lag to the failure, files/directories touched, diff size,
author, message keywords, overlap of touched files with stack-trace classes
and with class names in the report, stack-trace method names found in the
diff, and BM25 between the report + failure text and the commit message, and
between the same query and the diff. The FL track adds FONTE's vote,
max Ochiai, number of suspicious lines and history depth. Every method reads
only this table. Fixed-edge buckets of the numeric signals are part of the
table, so HeapDB (which takes binary features) and the learned baselines
(which take both) see the same information. Published systems are labelled
with their own inputs.

**Learning without leakage.** Bugs are processed in chronological order
(across projects for Defects4J). Each bug is ranked from a label-free view,
and only then is its label revealed to the learners (`core.prequential`).
Tuned methods choose from a small grid fixed in advance by
*follow-the-leader*: for bug *i*, the grid entry with the best MRR on bugs
`< i`. Each grid entry is itself a prequential run, so the choice uses only
past outcomes. Tests (`tests/test_eval_bic.py`) check that changing the
labels of later bugs never changes an earlier rank, for HeapDB and logistic
regression, that a spy method only ever sees earlier labels, that runs are
deterministic, and that the metrics match hand-checked values.

**Development slice.** The bucket edges, the prior grid and the feature set
were adjusted while looking only at the first 25% of each stream. RESULTS.md
reports that slice and the remaining 75% separately.

## Methods

- **Baselines (no learning):** recency; file overlap with the stack trace
  and report (Defects4J) or Jaccard file overlap with the fix (DPDK); BM25 on
  diffs and on messages; same author as the fix (DPDK).
- **Learned baselines (same table):** a tuned linear combination of z-scored
  signals; logistic regression (C picked by follow-the-leader); LambdaMART
  (LightGBM). All are retrained on every bug seen so far.
- **Published (labelled inputs):** Bug2Commit and FBL-BERT, from their
  released per-bug outputs re-scored on our candidates; FONTE (FL track),
  re-implemented from its MIT code and checked against its released ranks;
  max aggregation of FL scores. The papers' own numbers are quoted separately.
- **HeapDB:** each candidate commit is a cause event at its commit time,
  with the candidate's binary features. The failure is the effect `bic`, and
  `top_causes` ranks by `kernel(lag) x dose x noisy-OR(links)`. After ranking,
  `feedback` confirms the true BIC (proportional/EM credit) and rejects the
  other candidates. Links are hierarchical Beta links with the pooled global
  layer uncapped, because each bug is its own entity. Grid: flat or
  exponential (half-life 1 y) lag kernel, dose on/off (`log1p(lines)`),
  Beta prior mean 1e-3 or 1e-2 (pseudo-count 10). Ablations: signal buckets only, or
  identity features (author/dir/keyword) only; on the FL track, HeapDB with
  only FONTE's inputs (`abl:fl-inputs-only`: FL vote rank, max Ochiai,
  history depth), the apples-to-apples row against FONTE. Transfer: leave
  one Defects4J project out.
- **HeapDB, logistic combine mode** (`heapdb+logistic`, `+tuned`): the same
  events, effect, feedback and protocol, with the effect registered as
  `combine="logistic"`: `top_causes` ranks by
  `kernel(lag) x dose x sigmoid(b + sum of feature weights)`, the weights
  learned online (FTRL-Proximal, L2 towards seeds from the priors) from the
  same confirm/reject feedback. Grid: the same kernels and dose, learning
  rate 0.1 or 0.5 in place of the prior (HeapDB's default prior Beta(1, 4)
  seeds the intercept). The formulation was chosen on the development slices
  among simpler alternatives (naive-Bayes products of the Beta links, with
  several credit rules); `combine_dev.py` re-runs that comparison and
  RESULTS.md reports it.
- **Bisection cost:** test runs that `git bisect` needs over the candidates,
  and FONTE's score-weighted bisection with each method's ranking.

## Reproduce

```bash
pip install . -r evals/bic/requirements.txt   # builds heapdb (maturin) + numpy, scipy, pandas, scikit-learn, lightgbm
export HEAPDB_BIC_CACHE=/path/with/2GB/free  # optional; default ~/.cache/heapdb-bic
python -m evals.bic.run          # downloads (~500 MB of git data), builds, runs; resumes from cache
python -m evals.bic.report       # results.json -> RESULTS.md (about a minute: paired bootstraps)
pytest tests/test_eval_bic.py
```

`run` can be interrupted and restarted: every method's per-bug ranks are
cached under `$HEAPDB_BIC_CACHE/runs/`, and `--datasets d4j` or
`--datasets dpdk` reruns one dataset while keeping the other's results.

Downloads go to `$HEAPDB_BIC_CACHE` (default `~/.cache/heapdb-bic`):

- a depth-1 clone of `coinse/fonte` at the pinned commit;
- a bare clone of `DPDK/dpdk`;
- treeless clones of the 16 upstream project repositories (authors only).

The input-file manifest (sha256 over every FONTE file read), the derived
author table and DPDK's `git log --numstat` output are checked against the
sha256 values recorded in `d4j.py` and `dpdk.py`. A mismatch stops the run
unless `--allow-mismatch` is given.

FONTE ships coverage and history as pandas pickles. They are loaded with an
allowlisting unpickler that permits only the pandas/numpy reconstructors
those files use, never with `pickle.load`.

No dataset content is committed here. `results.json` holds bug ids, projects,
times, candidate counts and per-method ranks; it holds no report text or
code.

A full run from an empty cache took 47 minutes on 2 vCPUs and 8 GB of RAM
(resident memory was about 1.6 GB when sampled during DPDK, not a measured
peak), mostly logistic regression and LambdaMART refits. The cache holds
about 1 GB: the clones plus the per-method results.

## Results in one paragraph

On both Defects4J tracks, HeapDB with feedback beats recency, file overlap,
BM25 on commit messages and the released Bug2Commit and FBL-BERT rankings,
and ties BM25 on diffs; on the FL track it also ties FONTE and FONTE's
max-aggregation baseline. On DPDK it beats every unsupervised signal. Against
the learned rankers on the same inputs, it ties logistic regression on the
Defects4J report track and trails logistic regression and LambdaMART on the
FL track and on DPDK. Every number,
with its CI, is in [RESULTS.md](RESULTS.md); the summary there is generated
from `results.json` and states wins, ties and losses only from paired
bootstrap CIs.
