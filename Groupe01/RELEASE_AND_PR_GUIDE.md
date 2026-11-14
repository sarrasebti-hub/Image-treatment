# Release and Pull Request guide (Group 1)

Two things remain to be done **on a machine that has the generated data** (Nour's or Sarah's PC):
publish the data release, then submit the Pull Request to the instructor.
Follow the instructor's *GitHub Fork and Pull Request Student Guide* for the PR.

## 1. Publish the data release `g1-v1.2`

1. Make sure the embeddings and indexes exist locally (`artifacts/embeddings/`, `artifacts/index/`,
   `experiments/test_1000_val2017/embeddings/`). If not, run notebook `04_embeddings_index.ipynb`
   and `python scripts/run_test_1000.py`.
2. Run the full checks, which now include the embedding-based ones (no SKIP expected):

       python scripts/check_leak_test1000.py
       python scripts/qc_test_1000.py

3. Package the data (about 0.8 GB; add `--with-images` for the 1.2 GB of RGB images):

       python scripts/package_release.py

4. Create the release (GitHub CLI) with the command printed by the script, for example:

       gh release create g1-v1.2 release_assets/*.zip release_assets/SHA256SUMS.txt \
           --title "G1 data release g1-v1.2" --notes-file RELEASE_NOTES.md

   No GitHub CLI? On GitHub: *Releases → Draft a new release → tag `g1-v1.2`*, drag the files
   from `release_assets/` into the page, paste `RELEASE_NOTES.md` as the description, *Publish*.
5. Put the release URL in the Pull Request description (step 2 below).

## 2. Submit the Pull Request to the instructor

Repository: https://github.com/sarrasebti-hub/IMAGE-TRAITMENT-PROJECT

1. **Fork** the instructor's repository on GitHub (button *Fork*).
2. Clone **your fork**, then create the submission branch:

       git clone https://github.com/YOUR-USERNAME/IMAGE-TRAITMENT-PROJECT.git
       cd IMAGE-TRAITMENT-PROJECT
       git checkout -b group-01-submission

3. Copy the content of this repository (branch `g1-v1.2-final` once merged into `main`) into the
   fork, for example in a `group-01/` folder, **without** `.venv/`, `data/`, `release_assets/` or
   any `.zip`:

       git add .
       git status          # check: no data, no passwords, no large temporary files
       git commit -m "Submit Group 01 project (Phase 1)"
       git push -u origin group-01-submission

4. On GitHub open your fork → *Compare & pull request*. Check:
   base repository = the instructor's, head = your fork / `group-01-submission`.
5. Title: `Group 01 — Image Processing Project (Phase 1)`. Description:
   - members: Cerine Benmedghar, Nour El Houda Akhdari, Sarah Gherbi (and anyone else on the title page);
   - project: *A comparative study of RAG techniques for generating image captions* — Naive / Dense RAG;
   - official test set: 1,000 images (legacy 100-image test kept); link to the data release;
   - key results: see README section "Results on the official test".
6. If the instructor asks for changes, commit and `git push` on the same branch: the PR updates
   automatically. Do not open a second PR.

## Checklist before submitting

- [ ] `README.md` states 1,000 images = official test; legacy 100 kept
- [ ] `python scripts/qc_test_1000.py` → PASS with 0 SKIP on the machine with the data
- [ ] `pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt` works in a clean venv
- [ ] Release `g1-v1.2` is public and `SHA256SUMS.txt` is attached
- [ ] Report PDF (`reports/phase1_technical_report.pdf`) is the updated one (20 pages, section 11.3)
- [ ] The commit history is visible (not a single commit)
