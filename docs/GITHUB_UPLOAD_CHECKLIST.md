# GitHub Upload Checklist

## What the course PDF explicitly asks for

The course project PDF asks for a GitHub link containing:

- Well-commented, readable implementation code.
- Clear running instructions in a README, including environment, dependencies, and training instructions.
- Runnable source code in the final ZIP file.

The final report should also include the GitHub link and Kaggle submission log screenshots. The final ZIP should include the presentation PPT/PDF.

## Does all code have to be in one file?

No explicit requirement says all code must be merged into one Python file. A modular repository is acceptable as long as there is one clear entrypoint and the code can be run.

Recommended entrypoint:

```bash
python src/spaceship_raw_trio_groupaware_retrain.py --threads 8 --xgb-device cpu --apply-residual-stage --strict-candidates
```

## Should intermediate CSV files be uploaded?

Not all intermediate CSV files are needed.

Upload:

- Source code under `src/`.
- `README.md`.
- `requirements.txt`.
- Compact reproducibility artifacts under `artifacts/probability_tables/`.
- Generated submission examples under `artifacts/submissions/`.
- Small audit/summary files under `artifacts/summaries/`.

Do not upload by default:

- Large exploratory submission batches.
- Large exploratory output folders.
- Pickle caches.
- Optuna journal files.
- `catboost_info/`.
- PyCharm `.idea/`.
- Raw Kaggle data, unless the repository is private or the instructor explicitly requests it.

## Final ZIP suggestion

For iSpace final submission, include:

- GitHub link in the report.
- Runnable source code folder.
- Final report PDF.
- Presentation PPT/PDF.
- Kaggle submission log screenshots.
- Optional: final submission CSV and compact audit summaries.
