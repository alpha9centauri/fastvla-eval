# Stories

Nine things that happened while building this, written out because the interesting part of
each is the reasoning, not the result. Every number here is in the repository with the
artefact that produced it.

## The audit that found the labeler, not the model

The project was supposed to be about making a 7B policy forward cheaper. The first
milestone was an audit rather than an optimisation, and it measured three things on the
untouched harness: wall clock per control step, GPU SM utilisation sampled at 1 Hz, and a
cProfile of one rollout. The GPU sat at **9.9% mean utilisation** across a 60-start run.
Something that was not the GPU was taking 90% of the time. The profile named it: the
safety monitor — the thing computing obstacle clearance — was **142 ms of the 190 ms
control step**, and inside it the clearance labeler was almost all of it, at 63.6 ms per
call, called twice per control step by the reference behaviour.

That single measurement redirected the entire project. The work that followed was a C++
GJK labeler and parallel simulators, not a faster forward; the forward optimisation
milestone, when it eventually ran, rejected every candidate it tried. The lesson is
narrow and worth stating plainly: the component everyone assumes is the bottleneck in a
VLA system — the big transformer — was idle most of the time, and the bottleneck was a
geometry routine nobody had profiled. Had the project started with `torch.compile` it
would have optimised 9.9% of the problem and shipped a 2% win.

## The C++ port, and proving it didn't change anything

Rewriting the clearance labeler in C++ took it from 63.6 ms to **0.578 ms per call**,
about 110×. The interesting half was proving the two agreed. Clearance is a float computed
by an iterative convex-distance algorithm (GJK) over decomposed hulls, so "the same" is
not obvious: different summation orders give different last bits, and a labeler that is
subtly different changes which rollouts count as collisions, which is the only output
anybody cares about.

The parity harness replays 1000 recorded monitor calls through both implementations and
compares the returned `min_clearance` field for field. The full C++ stage agrees to a
**maximum absolute difference of 2.08e-16 m** — a couple of ULPs on a metre-scale value —
with 782 of 1000 calls bit-identical, and the threaded build is bit-identical to the
sequential C++ on all 1000. Two decisions fell out of that work and are recorded with
their numbers: `-ffast-math` was never enabled, because reassociation is exactly what
would break this; and `-march=native` was measured at 4.7% faster with parity intact and
**rejected anyway**, because the result ships in a container image that may not run on
this CPU.

## Rejecting dynamic batching, which was 13% faster

Batching requests from several workers into one GPU forward is the obvious win, and it
worked: at B=4 the harness ran at **618–624 rollouts/hour against 544.8**, about 13%
faster, at every queue delay tried. It was rejected, because at B=4 the bf16 GEMM
reduction order changes with batch composition, actions change by up to 6.6e-3, and
trajectories in contact-marginal scenes diverge.

The measurement that killed it is the per-start transition matrix, and it is the reason
that matrix exists. At a 10 ms delay the run reports **29 collisions and 26 completions**
— it passes the aggregate acceptance. It also flips **four** individual collision
verdicts (oa3 inits 0, 1, 6 and oa5 init 6), and they happen to net to zero. A reviewer
checking totals would have accepted a configuration that got four of sixty starts wrong.
Across the delays tried the flips ranged from three to five. The acceptance rule was
rewritten after this to require zero per-start flips rather than matching counts, and
every change since has been held to it.

## Rejecting torch.compile, which won its microbenchmark

`torch.compile(mode="default")` made the forward **168.0 → 156.7 ms, a 6.7% win** in a
microbenchmark that timed the forward in isolation with CUDA events, over hundreds of
calls, after warm-up. On that evidence it should have shipped.

End to end it was **459.0 rollouts/hour against 544.8 — 15% slower** — and it flipped
three collision verdicts. The microbenchmark measured a steady state the real workload
never reaches: eight workers send requests with two different prompt lengths, so the
compiled graph is re-specialised, and recompilation cost swamps the per-call saving. The
general lesson is the one every performance engineer learns twice: a microbenchmark
measures the thing you built it to measure, and an end-to-end benchmark measures the thing
you care about. The harness keeps both, and the acceptance gate is what makes the
disagreement visible instead of theoretical.

## The monitor's operating point, and why 90% recall was a parking brake

The collision monitor is a ridge probe on the policy's own activations predicting log
clearance. On held-out scenes it reaches **Spearman +0.868** and **AUROC 0.918** for
contact within 8 control steps, which looks like a usable alarm. The first operating point
was the one the milestone specified: the threshold giving 90% recall at a 16-step horizon.
At that threshold **every single clean rollout raised an alarm** — 31 of 31 unshifted,
8 of 8 shifted, 6.5 alarms per 100 control steps.

Wired to the brake, that produced a spectacular-looking result: collisions 29 → 1 on the
60-start set. It was meaningless. The monitor braked **all 108 rollouts** at a median of
control step 8 out of 300, so completions went 25 → 1 at the same time. One cause produced
both numbers: the arm stopped moving almost before it started. Reporting the collision
number alone would have been a lie of omission. Re-choosing the threshold at the knee of
the recall / false-alarm curve and adding a two-forward debounce gave something that is
actually a trade — **18 of 29 collisions prevented, zero starts made worse, 13 completions
lost**, braking 46 of 60 rollouts at a median of step 92. Both runs are kept in `EVAL.md`,
because the contrast is the argument.

## Finding out the harness is not run-to-run reproducible

While comparing two intrusion-5 runs, one completion verdict disagreed. Both runs used the
same manifest, the same configuration, the same code; the only difference was that one had
activation capture enabled. The obvious suspect was the capture hook.

It was not. Two runs with capture **off** differ from each other on 4 of 60 traces, so the
hook was innocent. Checking where each divergent trace first differs settled the cause: in
all 8 divergent traces across both sets, the first difference falls inside the first
control step of a forward — never mid-chunk, where a simulator discrepancy would show. The
simulator is deterministic; the policy forward is not bit-reproducible across runs at
`TORCH_DETERMINISTIC=0`, which is the configuration every number in this repository was
measured in. The practical consequence is a floor on what any per-start comparison can
claim: **zero collision flips on 60 starts, completions within 0 to 1**. Every
"identical" claim in the repository was rewritten to say that instead, and the
acceptance-gate documentation now says it is a verdict identity measured once, not a
property of the harness.

## The edit that silently emptied a file

Three times, a file was modified with the idiom
`open(p, "w").write(open(p).read().replace(a, b))`. Python evaluates the arguments before
the call, and `open(p, "w")` truncates the file at that moment — so when the inner read
had already happened the edit worked, and when it had not, the file became **zero bytes**.
`audit/t2_breakdown.py` sat empty for several commits. Nothing caught it: an empty Python
file imports cleanly, lints cleanly, and formats cleanly.

Two fixes. `tests/test_no_empty_sources.py` fails if any tracked source file is empty,
which turns a silent corruption into a red test in under a second. And every subsequent
string replacement asserts that its pattern matched before writing — because the same
class of bug reappeared in a quieter form, where a replacement simply did not match, the
script printed success, and the edit never happened. That one was found by a CI traceback
showing code that should no longer have existed. Verify the edit, not the intention to
edit.

## The benchmark table that went stale on every commit

The README's benchmark table is generated from the JSON artefacts under `bench/out/`, with
a `make bench-check` target that fails if the rendered table and the artefacts disagree.
Each row carries the commit it was measured at, which is the useful part: it tells a
reader when a number was last re-derived rather than when the file was last touched.

The first implementation put `HEAD` in that column. That made the table wrong the moment
anything else was committed, and it failed in the most instructive way possible — a fresh
clone running `make bench-check` found the table naming one commit while the clone sat at
the next. The fix is `git log -1 --format=%h -- <artefact>`: the commit that last touched
*that row's artefact*, which moves only when the number is re-measured. A provenance
column that changes when the provenance has not changed is worse than no column, because
it trains the reader to ignore it.

## The digest that was invented

The GPU `Dockerfile` pinned its base image by sha256 digest, which is the right way to pin
a base image: tags move, digests do not. The digest was fabricated — a plausible-looking
64-hex string that had never been fetched, never pulled, and corresponded to nothing. It
would have failed at the first `docker build`, which is the lucky case; the unlucky case
is a digest that resolves to some other image entirely.

It was removed and replaced with a tag pin, a comment saying exactly what happened, and
the command to obtain the real digest on a host that can pull. The wider response is
`docs/PROVENANCE.md`: every URL, version pin, citation and external fact in the repository
listed with how it was checked — fetched, read out of the installed package, read out of
the container image. Writing it turned up a second instance of the same failure mode: a
milestone brief cited a results file, `FT_LIFTONLY_RETEST.txt`, that does not exist
anywhere on the machine, and the number attributed to it belonged to a different
checkpoint whose rows were mislabelled. The rule that came out of both: a fact you cannot
point at is not a fact, and "I remember reading it" is not a citation.
