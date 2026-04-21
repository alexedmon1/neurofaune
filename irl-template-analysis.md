---
author: Alex Edmondson
affiliation: CCHMC
email: alex.edmondson@cchmc.org
study: {{STUDY_NAME}}
project_name: {{PROJECT_NAME}}        # e.g. "dti-white-matter", "fc-maturation"
project_slug: {{PROJECT_SLUG}}        # filesystem-safe: matches project_dir leaf
phase: analysis
cohorts: [{{COHORT_LIST}}]
modalities: [{{MODALITIES}}]          # modalities this project uses
project_dir: ~/research/{{STUDY_NAME}}/{{PROJECT_SLUG}}
study_dir: /mnt/arborea/{{STUDY_NAME}}
pipeline_dir: /home/edm9fd/sandbox/neurofaune
pipeline_venv: /home/edm9fd/sandbox/neurofaune/.venv/bin/python
atlas: /mnt/arborea/{{STUDY_NAME}}/atlas/{{ATLAS_NAME}}
---

<!-- AI Instructions:
Analysis-phase IRL project for a neurofaune animal MRI study.
- Preprocessing is DONE — $DERIV, $TEMPLATES, $TRANSFORMS, $EXCL are populated and maintained by the preprocessing project
- $PROJECT (this repo) is small, git-tracked; holds plan + result writeups + per-project scripts
- $STUDY on /mnt/arborea holds all big data, SHARED across projects
- This project's analysis outputs go to $ANALYSIS_OUT and $NETWORK_OUT (namespaced by project slug)
- Result summaries (markdown, small figures, small tables) live in $RESULTS inside $PROJECT
- Long jobs (randomise, permutation testing) MUST use nohup, logs to $STUDY/logs
- Before launching CPU-heavy jobs: `ps aux | grep -E '(randomise|ants|python.*run_)'`
-->

# {{PROJECT_NAME}} ({{STUDY_NAME}}) — Analysis Plan

## 📁 Paths — Single source of truth

### `$PROJECT` — this IRL project (small, git-tracked on home drive)

- **`$PROJECT`** — `~/research/{{STUDY_NAME}}/{{PROJECT_SLUG}}` — this repo root
- **`$PLAN`** — `$PROJECT/plans` — main-plan.md, activity log, CSV log
- **`$RESULTS`** — `$PROJECT/results` — markdown summaries, figures, small tables
- **`$SCRIPTS`** — `$PROJECT/scripts` — wrappers specific to this analysis

### `$STUDY` — study data (big, on arborea, shared across projects)

- **`$STUDY`** — `/mnt/arborea/{{STUDY_NAME}}`
- **`$BIDS`** — `$STUDY/raw/bids` — read-only
- **`$DERIV`** — `$STUDY/derivatives` — preprocessed subjects (maintained by preprocessing project)
- **`$TEMPLATES`** — `$STUDY/templates/anat`
- **`$ATLAS`** — `$STUDY/atlas/{{ATLAS_NAME}}`
- **`$TRANSFORMS`** — `$STUDY/transforms`
- **`$EXCL`** — `$STUDY/exclusions` — canonical CSVs (shared, not edited here)
- **`$ANALYSIS_OUT`** — `$STUDY/analysis/{{PROJECT_SLUG}}` — voxelwise outputs, namespaced to THIS project
- **`$NETWORK_OUT`** — `$STUDY/network/{{PROJECT_SLUG}}` — ROI outputs, namespaced to THIS project
- **`$LOGS`** — `$STUDY/logs`

### Pipeline (read-only)

- **`$PIPELINE`** — `/home/edm9fd/sandbox/neurofaune`
- **`$PY`** — `$PIPELINE/.venv/bin/python`

Rule: every section below refers to these by shorthand. Add new paths here before using them.

---

## 🔧 First Time Setup — Run once when starting this analysis

<!-- 👤 AUTHOR AREA: Fill in scope before first loop -->

### Research question
<!-- 2–4 sentences: what does this project investigate? What's the hypothesis? -->

### In-scope
- **Cohorts:** <!-- which cohorts this project uses -->
- **Modalities:** <!-- which modalities -->
- **Primary analyses:** <!-- TBSS, VBM, fALFF, ReHo, CovNet, MCCA, NBS, classification, regression -->
- **Contrasts / targets:** <!-- group comparison, regression target, interaction -->

### Out-of-scope
<!-- What this project will NOT touch; prevents scope creep -->

### Verify preprocessing is complete
```bash
ls $DERIV | wc -l        # expect: all subjects preprocessed
ls $TEMPLATES            # expect: one dir per cohort
ls $EXCL/*.csv           # expect: populated CSVs per modality
```
If any are missing, redirect to the preprocessing project before proceeding.

### Create project-namespaced output dirs
```bash
mkdir -p $ANALYSIS_OUT $NETWORK_OUT
```

### Common skill library
<!-- Uncomment to use -->
<!-- Install Scientific Writing: https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/scientific-writing -->
<!-- Install PubMed Search: https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/pubmed-database -->
<!-- Install PPTX Posters: https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/pptx-posters -->

---

## ✅ Before Each Loop

- **Clean git tree** in `$PROJECT`: `git status`
- **Running-jobs check**: `ps aux | grep -E '(randomise|ants|python.*run_)'`
- **Disk check**: `df -h /mnt/arborea`
- **Exclusions current**: `ls -la $EXCL/*.csv` — changes since last loop? (Maintained by preprocessing project.)
- **Pipeline version**: `cd $PIPELINE && git log -1` — record hash
- Scripts must be **idempotent** — re-running produces identical output or no-op
- Only `## One-Time Instructions` is plan-editable without explicit permission

---

## 🔁 Instruction Loop — Define the work for each iteration

<!-- 👤 AUTHOR AREA: Edit each loop. -->

### Loop task (current)

- **Analysis:** <!-- e.g. "TBSS FA per-cohort", "CovNet MSME", "MCCA 2-view DWI+MSME" -->
- **Cohorts:** <!-- subset of project cohorts -->
- **Metric(s):** <!-- FA, MD, AD, RD, T2, MWF, fALFF, ReHo, ALFF, etc. -->
- **Contrast / target:** <!-- group comparison, regression target -->
- **Exclusion CSV:** <!-- $EXCL/dwi_exclusions.csv -->
- **Output dir:** <!-- $ANALYSIS_OUT/tbss/per_{cohort} or $NETWORK_OUT/covnet/... -->

### Command templates

**Voxelwise (TBSS, VBM, fALFF, ReHo, MVPA):**
```bash
nohup $PY $PIPELINE/scripts/run_voxelwise_fmri_analysis.py \
    --analysis-dir $ANALYSIS_OUT/{{DESIGN}} \
    --metrics {{METRIC}} \
    --exclusion-csv $EXCL/{{MOD}}_exclusions.csv \
    --target {{TARGET}} \
    --n-permutations 5000 \
    > $LOGS/{{DESIGN}}_{{METRIC}}_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**ROI-based (classification, regression, CovNet, MCCA, NBS):**
```bash
nohup $PY $PIPELINE/scripts/run_{{METHOD}}.py \
    --config $STUDY/config.yaml \
    --output-dir $NETWORK_OUT/{{ANALYSIS}} \
    --exclusion-csv $EXCL/{{MOD}}_exclusions.csv \
    --target {{TARGET}} \
    > $LOGS/{{METHOD}}_{{ANALYSIS}}_$(date +%Y%m%d_%H%M).log 2>&1 &
```

### One-Time Instructions — Tasks that should only execute once

<!-- 👤 AUTHOR AREA: Add tasks. Move to Completed once done. -->

- [ ] Generate design matrices for primary contrast (`$PIPELINE/scripts/generate_designs.py`)
- [ ] Validate ROI extraction exists for all non-excluded sessions
- [ ] Draft preregistration / analysis plan summary in `$RESULTS/00_analysis_plan.md`

#### Completed (don't re-run)
<!-- Move checked items here with date -->

### Formatting Guidelines

- **Result summaries** → `$RESULTS/{analysis_name}.md` with: design, N, exclusions applied, significant clusters/ROIs (p_FWE<0.05), effect direction, figure refs
- **Figures** → `$RESULTS/figures/` as PNG/SVG; never inline binary blobs
- **Tables** → `$RESULTS/tables/` as small CSVs; render markdown summary table in the `.md`
- **Large outputs** (NIFTI maps, full HTML reports) live in `$ANALYSIS_OUT` / `$NETWORK_OUT`, not in `$PROJECT`
- **Paths in reports** — always shorthand from `## Paths` or absolute; never `../../`
- **Number formatting** — p-values 3 sig figs, effect sizes 2 decimals, N as integer

---

## 📝 After Each Loop

- **Update activity log** (`$PLAN/main-plan-activity.md`, append 1–2 lines):
  - Analysis, design, output path
  - Timestamp (UTC), `$PROJECT` hash, `$PIPELINE` hash
  - Sessions excluded beyond canonical CSVs (with reason)

- **Update plan log** (`$PLAN/main-plan-log.csv`):
  `timestamp,analysis,cohort,metric,target,n_subjects,output_dir,status,project_hash,pipeline_hash`

- **Commit `$PROJECT`** — plan edits, new result writeups, figures, tables, scripts
  - Never commit anything from `$STUDY`; only `$PROJECT` is under this repo's git
  - Message format: `{analysis}: {one-line result}` (e.g. `TBSS per-cohort FA: sig clusters in CC`)

- **Feedback to AUTHOR**:
  1. What was done, results summary, next steps
  2. Idempotency or stale `## One-Time Instructions` issues
  3. Critical reasoning errors or QC concerns
  4. Pipeline quirks worth filing upstream in neurofaune

---

## 📚 Skill Library — Community skills (optional)
<!-- Uncomment to use -->
<!-- Install Scientific Writing -->
<!-- Install BioRx Search -->
<!-- Install Flowcharts -->

---

## 📌 Study-specific conventions

### Exclusion system
- `$EXCL/*.csv` is the only source of truth (maintained by the preprocessing project)
- Scripts consume via `--exclusion-csv`
- Never apply ad-hoc filters
- Flag any exclusion that would change cohort N by >5% for author review before proceeding

### Design naming
<!-- 👤 AUTHOR AREA: Define the naming scheme for this project's designs -->
- Voxelwise: `per_{cohort}`, `pooled`, `{target}_response_{cohort|pooled}`
- Network: `{metric}_{target}_{cohort}`

### Output namespacing
All outputs that this project writes to `$STUDY` go under `{{PROJECT_SLUG}}`:
- `$STUDY/analysis/{{PROJECT_SLUG}}/...`
- `$STUDY/network/{{PROJECT_SLUG}}/...`

This prevents collisions with other analysis projects on the same study.
