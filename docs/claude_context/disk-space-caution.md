---
name: disk-space-caution
description: "User is worried about disk space: don't extract or copy datasets unless a pilot actually needs it"
metadata:
  type: feedback
---

Don't extract, copy or cache datasets unless the current pilot actually needs them. Ask before any bulk data job.

**Why:** on 2026-10-19 the user stopped a background ScanNet extraction (~15 GB) they had approved minutes earlier, saying they were "concerned about the space available on disk". At that point `/home/data` had 147 GB free and `/` had 219 GB.

**How to apply:** prefer working from data already on disk, read lazily or on the fly, and cache only small per-object artifacts. Give a size estimate before writing anything over a few GB. Applies to [[caoa-replication-project]].
