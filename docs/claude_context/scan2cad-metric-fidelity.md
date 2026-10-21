---
name: scan2cad-metric-fidelity
description: "CAOA evaluation must follow the official Scan2CAD benchmark exactly; report problems in Scan2CAD itself and ask before deviating"
metadata:
  type: feedback
---

The new codebase's evaluation must follow the official Scan2CAD benchmark code as closely as possible, because the paper compares against Scan2CAD-evaluated methods. If the official metric has a problem (a bug or an ambiguity), tell the user and ask whether to change it. Don't change it unilaterally.

**Why:** the user said so on 2026-10-20, after I found that the legacy metric deviates from Scan2CAD in several ways (see [[caoa-legacy-code-findings]]).

**How to apply:** the "official" mode in `caoa/eval/metrics.py` must replicate the Scan2CAD script, including its quirks. The legacy mode stays only to reproduce the old numbers. Applies to [[caoa-replication-project]].
