# Contributing

The rule that matters: **a change that moves a collision verdict is rejected, whatever it
gains.** Two speed-ups that made the harness measurably faster were rejected on exactly
this basis, and both are written up with their numbers in `fastvla/server/DESIGN.md` and
`fastvla/server/FORWARD.md`.

## Before you open a PR

```bash
make lint      # ruff check + format check
make test      # the CPU suite: no GPU, no checkpoint
```

If your change can touch rollout behaviour at all — the worker loop, the server, the
labeler, the action path — also run the gate on GPU 1 and put the output in the PR:

```bash
RUN_ID=my_change bash bench/m3a_run.sh ctrl60 8 1 0
make accept RUN=my_change
```

It must report 29 collisions, completions in {25, 26}, and **zero** collision flips.
Aggregate counts are not enough: a configuration once passed the totals with four flipped
verdicts that happened to cancel, which is why the per-start transition matrix is checked.

## House rules

- **Every number in a document names the artefact it came from.** The README table is
  generated from `bench/out/*.json` by `bench/render_table.py`; `make bench-check` fails
  if the two disagree. Do not hand-edit the table.
- **Do not change a measured number, or a bench script's behaviour, to make CI pass.** If
  a test cannot run on the runner, skip it with a named reason.
- **External facts get verified, not remembered.** URLs, version pins and citations go in
  `docs/PROVENANCE.md` with how they were checked. A digest that was never fetched is
  worse than no digest; this repository has made that mistake once already.
- **GPU jobs run on GPU 1 only, one at a time.** GPU 0 belongs to someone else. Check
  `nvidia-smi` first, and never touch a process you did not start.
- Save the arrays. Every bench script writes its raw output before it prints a summary,
  so a claim can be re-derived later without re-running the GPU.

## Style

`ruff` with the settings in `pyproject.toml` is the whole style guide; `make format`
applies it. Comments should say why, not what — the surrounding code is the what.
