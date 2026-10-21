---
name: preserve-legacy-evaluation
description: "Every evaluation method of the legacy CAOA code must be preserved and reproduced as-is first; fixes come only afterwards"
metadata:
  type: feedback
---

All evaluation methods used in the legacy codebase must be preserved, and results recreated exactly as the legacy code produced them, before anything is fixed. This covers the metrics, the report formats (per-class CSVs with macro/micro averages, MAEs, timing), the dataset composition behind reported numbers, and the SoftGroup, completion and symmetry evaluations.

**Why:** the user said so on 2026-10-20, after reviewing the plan: "I first want to recreate it as it is, and then think about fixing anything that's wrong."

**Code over paper (user, 2026-10-20):** when the paper and the legacy code disagree, reproduce the code. Note the paper mismatches (PLAN.md "Paper vs codebase mismatches") but don't chase them: "there might have been some post-processing/corrections involved".

**Functional parity, not format (user, 2026-10-20):** "faithful" means the same numbers and behaviour. Wrapping results so they are presented or output differently is fine (e.g. JSON/markdown instead of the legacy CSV layout).

**How to apply:** each phase's first gate reproduces the legacy numbers with the legacy evaluation. Official Scan2CAD numbers ([[scan2cad-metric-fidelity]]) are reported in addition, never instead. Don't drop or "improve" a legacy evaluation path without asking. Applies to [[caoa-replication-project]].
