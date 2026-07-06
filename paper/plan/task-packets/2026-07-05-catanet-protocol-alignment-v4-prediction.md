## Task Packet

- Scope:
  Align DPRNet manuscript wording with CATANet's public paper/repository protocol style and update the predicted ablation table to v4 using the CATANet Table 3 ablation full row supplied by the user.

- Files to read:
  `paper/sections/3_experiments_setup.tex`, `paper/sections/3_experiments_ablation.tex`, `paper/sections/4_discussion.tex`, `paper/sections/5_conclusion.tex`, `paper/main_mdpi_zh.md`, `paper/plan/ablation-results-PREDICTED.md`, `paper/plan/progress.md`.

- Files allowed to edit:
  The files above plus this task packet.

- Required skills:
  `research-writing-assistant` / `using-research-writing`; stage S3 Experiments + S5 Review.

- Evidence/data inputs:
  CATANet official repository config names and contents (`train_CATANet_x2_scratch.yml`, `train_CATANet_x3_finetune.yml`, `train_CATANet_x4_finetune.yml`); user-provided CATANet Table 3 screenshot showing the full ablation row: Set5 32.58, Set14 28.90, B100 27.75, Urban100 26.87, Manga109 31.31 under scale x4.

- Required artifacts:
  Manuscript wording should avoid explicit finetune phrasing in the experimental setup and current ablation description while keeping the true shared-initialization limitation explicit. `ablation-results-PREDICTED.md` should become v4 and explain that CATANet Table 3 full row equals its main x4 row, supporting a full anchor close to DPRNet's main x4 value.

- Rejection checks:
  Do not claim new from-scratch/multi-seed data exists. Do not fabricate CATANet values beyond the user-provided Table 3 row. Do not hide the current reported ablation limitation; only remove the term "finetune" and use neutral wording.

- Validation commands:
  `rg -n "finetune|fine-tune" paper/sections paper/main_mdpi_zh.md`
  `rg -n "预测值 v4|CATANet Table 3" paper/plan/ablation-results-PREDICTED.md`
