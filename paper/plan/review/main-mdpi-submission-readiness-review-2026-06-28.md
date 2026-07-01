# Main MDPI Submission Readiness Review - 2026-06-28

## Overall verdict

The manuscript is technically coherent and much closer to a submission-ready MDPI article than the earlier draft. The core story is self-consistent: DPR replaces CATANet's center-based routing, keeps the backbone/interface unchanged, reports standard SR benchmarks, and frames the gains over CATANet conservatively.

I would not submit this exact PDF yet. It needs a pre-submission revision focused on metadata/back matter, one experiment-protocol wording error, visual figure readability, and tighter evidence framing around the very small CATANet margins.

## Submission blockers

1. Front matter and author statements still use placeholders.

   `main_mdpi.tex:47-63` still renders `Author Name`, `author@example.edu`, and `Department, Institution, City, Country`. `main_mdpi.tex:134-136` also uses the placeholder macro in Author Contributions. Applied Sciences submissions are normally not anonymous unless the journal or special issue explicitly asks for anonymization, so this must be replaced before submission.

2. The routing-temperature initialization is described incorrectly.

   `sections/3_experiments_setup.tex:36-38` says the learnable routing temperature is initialized at `e^6` and clamped at 10. The implementation uses `router_logit_scale = log(router_scale_init)` with `router_scale_init: 6.0`, then `exp(...).clamp(max=10)` (`CATANet/basicsr/archs/catanet_arch.py:174,208`). The manuscript should say: initialized to `6.0` (implemented as a learnable log-scale parameter with `theta = log 6`) and clamped at `tau_max = 10`.

3. Missing or incomplete MDPI back-matter statements.

   The template includes `\institutionalreview{...}`, `\informedconsent{...}`, and `\acknowledgments{...}`. For this non-human/non-animal ML paper, add explicit `Not applicable` statements if the target Applied Sciences submission system expects them. Also decide whether an AI-tool acknowledgment is needed, because the MDPI template explicitly asks authors to disclose GenAI use when applicable.

## Major scientific and presentation issues

1. Checkpoint selection may be read as test-set tuning.

   `sections/3_experiments_setup.tex:56-58` says the unified checkpoint is selected by best benchmark-averaged PSNR/SSIM over Set5--Manga109. That is transparent, but these are the reported benchmark test sets. For a stronger submission, select by a separate validation split or report a fixed final checkpoint. If the current protocol is kept, explicitly frame it as a common SR reporting practice and avoid wording that implies an untouched test-set evaluation.

2. The direct CATANet evidence is still the most likely reviewer attack point.

   The main comparison is against published CATANet numbers, and the gains are only around 0.01--0.07 dB with DPRNet using more params/MACs. The ablations are honest but short-budget, shared-initialization, and single-seed. This is acceptable only if the contribution is framed as a confidence-aware, interpretable routing replacement that preserves or slightly improves CATANet-level accuracy. A same-code/same-training CATANet/TAB baseline, direct EMA-center vs. DPR ablation, or multi-seed x4 run would materially strengthen the paper.

3. Fig.4 is not visually persuasive enough for an SR paper.

   The current qualitative comparison fits three samples, seven methods, and GT into one page. The crops are small, labels are tiny, and the differences between CATANet and DPRNet are hard to see. The supporting CSV also shows DPRNet is below CATANet on two Urban100 hard crops (`19.62` vs. `19.76`, `21.05` vs. `21.53`) and essentially tied on Manga109 (`21.42` vs. `21.41`). The caption's "on par" wording is defensible, but "sharper, less aliased" should be softened or supported with larger zooms. Prefer splitting into two full-width figures or showing fewer methods per row with larger crops.

4. Fig.2 is too dense at its rendered size.

   The DPR flow is conceptually useful, but many labels and arrows are barely readable in the PDF. Split into "prototype generation/refinement" and "confidence-aware aggregation" panels, or enlarge it and shorten the caption. The figure currently asks reviewers to trust the caption rather than read the diagram.

5. Fig.7 cluster maps are useful but visually noisy.

   The maps support the interpretability story, but the color palette is very saturated and some panels look random rather than semantically grouped. Add boundaries/alpha over the LR image, use a smaller discrete palette for the displayed slots, or add a short quantitative side note such as active slots/entropy per displayed block. The caption already states colors are not comparable across blocks, which is good.

## Data and wording corrections

1. Fig.5 and the routing-analysis text over-specify the confidence ratio.

   `fig5_xscore_hist.tex:88-90` and `sections/3_experiments_routing_analysis.tex:51-55` state the block-mean confidence is about `1.5--1.9 x 1/M`. The stats files show the plotted blocks are roughly `1.50--2.14 x` depending on dataset/block, and all blocks range wider than that. Replace the fixed range with data-safe wording, e.g. "above the uniform floor for all plotted blocks" plus the two numeric examples.

2. Some "clearly ahead" wording is stronger than the numbers.

   `sections/3_experiments_comparison.tex:29-37` says DPRNet and CATANet are "clearly ahead" of the remaining methods. This is true on many Urban100/Manga109 cells, but on several Set5/Set14/BSD100 cells the gap is only a few hundredths of a dB. Use "consistently top-tier and usually ahead" or "slightly ahead on most cells".

3. Abstract is accurate but dense.

   The abstract's mechanism sentence lists nearly every module. It is correct, but hard to parse. Shorten it slightly if page/word count allows; the method section carries the full detail.

## What is already strong

- The Introduction has a clean logic chain from CNN locality, to window attention, to content routing, to CATANet/TAB limitations.
- The Method is self-contained and maps equations to contributions well.
- Table 1 is dense but valuable; best/second-best highlighting matches the visible numbers.
- Table 2 is now much more defensible than an all-method table with incompatible MAC/latency conventions.
- The ablation caveats are unusually honest and reduce the risk of overclaiming.
- The final LaTeX build has no unresolved references/citations or fatal warnings in `main_mdpi.log`.
- The generated bibliography no longer shows the earlier repeated "Proceedings of the Proceedings" problem, and the reference list is broadly compatible with MDPI numeric style.

## MDPI style alignment

I spot-checked a recent Applied Sciences SISR article, "A Lightweight Single-Image Super-Resolution Method Based on the Parallel Connection of Convolution and Swin Transformer Blocks" (Appl. Sci. 2025, 15(4), 1806, https://www.mdpi.com/2076-3417/15/4/1806), and related MDPI SR papers in Electronics/Applied Sciences. The current structure is compatible with MDPI style: Abstract, Keywords, Introduction, Method, Experiments, Discussion, Conclusion, back matter, numeric references, architecture figures, quantitative tables, visual comparison, and ablation/efficiency analysis.

The main style gaps are practical rather than structural: real metadata, complete back matter, readable figures, and avoiding computer-vision conference phrasing that overemphasizes tiny PSNR deltas.

## Recommended revision order

1. Replace author metadata and complete MDPI back matter.
2. Correct the routing-temperature initialization wording.
3. Enlarge/split Fig.4 and Fig.2; make Fig.7 less visually noisy if time allows.
4. Soften data wording around Fig.5 confidence ratios and "clearly ahead" accuracy claims.
5. Decide whether to keep benchmark-based checkpoint selection; if kept, disclose it more carefully.
6. If time permits, add one stronger direct CATANet/TAB evidence item: same-protocol CATANet run, EMA-center ablation, or multi-seed x4 repeat.

## Verification performed

- Read `research-writing-assistant`, `using-research-writing`, `paper-orchestration`, and `peer-review` instructions.
- Read `main_mdpi.tex`, all section files, table files, figure files, data CSVs, `references.bib`, generated `.bbl`, and final `.log`.
- Rendered all 23 PDF pages with Ghostscript and visually checked the pages containing front matter, figures, tables, back matter, and references.
- Compared key numerical claims against `paper/data/*.csv`, `fig7_assets/*.csv`, and `fig58_assets/*_xscore_stats.csv`.
- Checked CATANet implementation/config for the routing-temperature initialization.
- Spot-checked MDPI similar-paper style via MDPI article pages.
