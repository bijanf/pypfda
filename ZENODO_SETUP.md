# Zenodo archiving for the GMD submission

GMD's Code and Data Policy requires a **live, frozen, version-pinned DOI accessible at
submission** (not "on acceptance"), and **GitHub is not an acceptable archive of record**.
Two DOIs are needed: one for the **code** (`pypfda`) and one for the **dataset** (OSSE
diagnostics + run configs + plotting scripts). No SSH key is involved — the GitHub↔Zenodo
link is an OAuth (browser) authorization.

---

## DOI 1 — pypfda code (automated GitHub → Zenodo, recommended)

This is the "wise method" that keeps GitHub and Zenodo synced automatically: every GitHub
**Release** is archived and gets a new versioned DOI, plus a stable **concept DOI** that
always points to the latest version.

1. Go to <https://zenodo.org/account/settings/github/> and log in (ORCID/GitHub).
2. Find **`bijanf/pypfda`** in the repository list and flip the toggle **ON**.
   (If the repo is not listed, click *Sync* — it lists your GitHub repos.)
3. Back in this repo, cut a release so Zenodo archives it:
   ```bash
   cd ~/pypfda
   git add .zenodo.json CITATION.cff
   git commit -m "Add Zenodo metadata for v1.0 archival"
   git tag -a v1.0 -m "pypfda v1.0 (GMD submission)"
   git push origin main --tags
   ```
   Then on GitHub: **Releases → Draft a new release → choose tag `v1.0` → Publish**.
4. Within a minute Zenodo creates the deposit and mints the DOI. Copy **both**:
   - the **concept DOI** (cite this in the paper — always latest), and
   - the **version DOI** for v1.0.
   `.zenodo.json` (already in this repo) auto-fills the title, authors, license, and keywords,
   so the record needs no manual editing.

> Authorship note: `.zenodo.json` lists all five paper authors as creators to match the
> manuscript's code citation. Trim to taste before publishing the release if the software
> authorship should differ from the paper authorship.

## DOI 2 — OSSE dataset (manual Zenodo upload; NetCDFs are too big for git)

1. Go to <https://zenodo.org/uploads/new>.
2. Upload the staged bundle (TRUTH/FREE/DA AMOC & SST time series, ESS/weight/genealogy
   records for CM2Mc-BLING + CLIMBER-X + the PlaSim-LSG limitation experiment, the OSSE
   namelists/`field_table`, `run_online_da.py`, `da_framework/compute_costs_osse_multiyear.py`,
   and `figs/*.py`). **Important for GMD:** include the PlaSim-LSG diagnostics too — its
   Δr=−0.23 null is a quantitative claim in the paper, so its data must be archived.
3. Metadata: Upload type **Dataset**; License **Creative Commons Attribution 4.0**;
   authors = the five paper authors; title = "Online Particle-Filter OSSE Ensemble
   Diagnostics for AMOC Reconstruction".
4. **Publish** → copy the dataset DOI.

## After both DOIs exist — paste them back

Send me the two DOIs (or edit `sample.bib` directly). I will:
- replace `Zenodo DOI to be assigned upon acceptance` in `@pypfda2026` and `@osse_data2026`
  with the real `doi = {...}` strings, and
- drop the "[DOI to be minted on acceptance]" placeholder wording in the manuscript's
  *Code and data availability* section.

That closes GMD's single biggest desk-check item.
