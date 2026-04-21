---
author: Alex Edmondson
affiliation: CCHMC
email: alex.edmondson@cchmc.org
study: {{STUDY_NAME}}
species: {{SPECIES}}                  # rat | mouse
cohorts: [{{COHORT_LIST}}]            # e.g. [p30, p60, p90]
modalities: [{{MODALITIES}}]          # subset of: T2w, DTI, fMRI, MSME, MTR, MRS
phase: preprocessing
project_dir: ~/research/{{STUDY_NAME}}/preprocessing
study_dir: /mnt/arborea/{{STUDY_NAME}}
pipeline_dir: /home/edm9fd/sandbox/neurofaune
pipeline_venv: /home/edm9fd/sandbox/neurofaune/.venv/bin/python
atlas: /mnt/arborea/{{STUDY_NAME}}/atlas/{{ATLAS_NAME}}   # e.g. SIGMA (rat), DSURQE/AMBMC (mouse)
---

<!-- AI Instructions:
Preprocessing-phase IRL project for a neurofaune animal MRI study.
- $PROJECT (this repo) is small, git-tracked; holds plan, QC notes, activity log
- $STUDY on /mnt/arborea holds all big data and pipeline artifacts
- All preprocessing outputs (derivatives, templates, transforms) go to $STUDY; never into $PROJECT
- Long jobs (batch preproc, template build, ANTs) MUST use nohup, logs to $STUDY/logs
- Before launching CPU-heavy jobs: `ps aux | grep -E '(ants|python.*batch_|python.*build_templates)'`
-->

# {{STUDY_NAME}} — Preprocessing Plan

## 📁 Paths — Single source of truth

### `$PROJECT` — this IRL project (small, git-tracked on home drive)

- **`$PROJECT`** — `~/research/{{STUDY_NAME}}/preprocessing` — this repo root
- **`$PLAN`** — `$PROJECT/plans` — main-plan.md, activity log, CSV log
- **`$QC_NOTES`** — `$PROJECT/qc-notes` — markdown notes on QC passes and exclusion decisions

### `$STUDY` — study data (big, on arborea, shared across all projects)

- **`$STUDY`** — `/mnt/arborea/{{STUDY_NAME}}` — study data root
- **`$BIDS`** — `$STUDY/raw/bids` — input BIDS data, read-only
- **`$DERIV`** — `$STUDY/derivatives` — preprocessed per-subject outputs
- **`$TEMPLATES`** — `$STUDY/templates/anat` — cohort population templates
- **`$ATLAS`** — `$STUDY/atlas/{{ATLAS_NAME}}` — atlas in study acquisition space
- **`$TRANSFORMS`** — `$STUDY/transforms` — ANTs transform registry
- **`$EXCL`** — `$STUDY/exclusions` — canonical per-modality exclusion CSVs
- **`$LOGS`** — `$STUDY/logs` — nohup log destinations
- **`$QC`** — `$STUDY/qc` — HTML QC reports produced by pipeline

### Pipeline (read-only)

- **`$PIPELINE`** — `/home/edm9fd/sandbox/neurofaune` — neurofaune repo
- **`$PY`** — `$PIPELINE/.venv/bin/python`

Rule: every section below refers to these by shorthand. If you need a new absolute path, add it here first.

---

## 🔧 First Time Setup — Run once when establishing the study

1. **Init study in neurofaune** (creates `$STUDY/config.yaml`):
   ```bash
   cd $PIPELINE
   uv run python scripts/init_study.py $STUDY --name "{{STUDY_NAME}}" --bids-root $BIDS
   ```
2. **Reorient atlas** into study acquisition space → `$ATLAS`
3. **Initialize exclusion CSVs** with headers `subject,session,reason,date_added`:
   ```bash
   for mod in anat dwi func msme; do
     echo "subject,session,reason,date_added" > $EXCL/${mod}_exclusions.csv
   done
   ```
4. **Snapshot `$STUDY/config.yaml`** into `$PROJECT` (copy, don't symlink) so the version used by preprocessing is committed in this repo's git history
5. **Commit baseline** in `$PROJECT`: plan, config snapshot, empty QC notes

### Common skill library
<!-- Uncomment to use -->
<!-- Install Quarto: https://github.com/posit-dev/skills/tree/main/quarto/authoring -->

---

## ✅ Before Each Loop

- **Clean git tree** in `$PROJECT`: `git status`
- **Running-jobs check**: `ps aux | grep -E '(ants|python.*batch_|python.*build_templates)'`
- **Disk check**: `df -h /mnt/arborea`
- **Pipeline version**: `cd $PIPELINE && git log -1` — record for reproducibility
- Any step that writes to `$DERIV`, `$TEMPLATES`, or `$TRANSFORMS` must be idempotent (re-run = no-op or byte-identical output)
- Only `## One-Time Instructions` is plan-editable without explicit permission

---

## 🔁 Instruction Loop — Define the preprocessing work for each iteration

<!-- 👤 AUTHOR AREA: Edit each loop. -->

### Loop task (current)

- **Phase:** <!-- anatomical preproc | template build | atlas registration | QC pass | exclusion triage -->
- **Cohort / subjects:** <!-- which batch -->
- **Modality:** <!-- T2w / DTI / fMRI / MSME / ... -->
- **Expected output:** <!-- $DERIV, $TEMPLATES/{cohort}, $TRANSFORMS, etc. -->

### Command templates

**Anatomical batch preprocessing (for template building):**
```bash
cd $PIPELINE
nohup uv run python scripts/batch_preprocess_for_templates.py $BIDS $DERIV \
    > $LOGS/preproc_anat_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Build cohort template:**
```bash
cd $PIPELINE
nohup uv run python scripts/build_templates.py --config $STUDY/config.yaml --cohort {{COHORT}} \
    > $LOGS/build_template_{{COHORT}}_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Registration QC pass:**
- Review HTML reports in `$QC`
- For each failure, add a row to the appropriate `$EXCL/*_exclusions.csv` with a concrete `reason`
- Record rationale in `$QC_NOTES/{cohort}_{modality}_qc.md`

### One-Time Instructions — Tasks that should only execute once

<!-- 👤 AUTHOR AREA: Add tasks. Move to Completed once done. -->

- [ ] `init_study.py` for `$STUDY`
- [ ] Reorient atlas → `$ATLAS`
- [ ] Initialize empty exclusion CSVs
- [ ] Snapshot `config.yaml` into `$PROJECT`
- [ ] First full batch of anatomical preprocessing
- [ ] Build template for each cohort
- [ ] First registration QC pass, per cohort × modality
- [ ] First DTI preprocessing pass
- [ ] First fMRI preprocessing pass
- [ ] First MSME preprocessing pass

#### Completed (don't re-run)
<!-- Move checked items here with date -->

### Formatting Guidelines

- **QC notes** → `$QC_NOTES/{cohort}_{modality}_qc.md` with: date, sessions reviewed, failures + reason, exclusions added
- **Exclusion rows** — every row in `$EXCL/*.csv` needs a concrete `reason` string; expand rationale in `$QC_NOTES` if needed
- **Paths** — always shorthand from `## Paths`; never `../../`

---

## 📝 After Each Loop

- **Update activity log** (`$PLAN/main-plan-activity.md`, append 1–2 lines):
  - Phase, subjects processed, outputs produced
  - Timestamp (UTC), `$PROJECT` git hash, `$PIPELINE` git hash
  - Exclusions added this loop (count + pointer to `$QC_NOTES/`)

- **Update plan log** (`$PLAN/main-plan-log.csv`):
  `timestamp,phase,cohort,modality,n_processed,n_excluded,output_path,status,project_hash,pipeline_hash`

- **Commit `$PROJECT`** — plan edits, QC notes, log updates, config snapshots only
  - Never commit anything from `$STUDY`; `$STUDY` is not under this repo's git
  - Commit message: `preproc: {phase} {cohort}/{modality} — {outcome}`

- **Feedback to AUTHOR**:
  1. Phase progress, subjects remaining
  2. QC findings needing attention
  3. Pipeline issues worth filing upstream in neurofaune

---

## 📚 Skill Library — Community skills (optional)
<!-- Uncomment to use -->

---

## 📌 Study-specific conventions

### Transform chain
```
Subject functional (FA / BOLD / MSME) → Subject T2w → Cohort Template → $ATLAS
              (rigid/affine)                (SyN)            (SyN)
```

### Exclusion system
- `$EXCL/*.csv` is the only source of truth for downstream analyses
- Never apply ad-hoc filters outside this system
- Every exclusion carries a `reason` string
- Detailed rationale lives in `$QC_NOTES`

### Handoff to analysis projects
When preprocessing is complete (all cohorts, all modalities, exclusions populated), tag a `preprocessing-v1` commit in `$PROJECT` and note it in the activity log. Analysis projects (see `neurofaune-analysis` template) will reference this snapshot.
