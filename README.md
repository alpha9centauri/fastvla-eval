# FastVLA-Eval

FastVLA-Eval is a fast and correct closed-loop evaluation harness for vision-language-action
(VLA) policies on LIBERO-Safety. It aims to make closed-loop evaluation substantially faster
than the current single-environment driver — parallel simulation workers against a batched
policy server, with a C++ clearance labeler and the LIBERO-Safety scoring fixes built in —
without changing results, and to prove that with bit-exact parity checks in a reference mode
and statistical equivalence checks in fast mode. All performance and accuracy numbers are
pending the audit; nothing here is measured yet.

Author: Varun Chandra
