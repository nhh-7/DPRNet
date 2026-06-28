# MDPI Full-Paper Review -- 2026-06-28

## Overall Verdict

The manuscript is close in structure and technical framing, but it is not ready for submission today. I would treat it as a major pre-submission revision: the core story is coherent, the method is described clearly, and the claims are mostly conservative; however, several items would be visible to an editor or reviewer immediately.

The main blockers are front-matter placeholders, reference formatting, a clipped routing-confidence figure, draft/internal wording leaked into the manuscript, incomplete unified efficiency evidence, and a few claim/evidence mismatches around "unchanged budget" and the direct CATANet comparison.

## What Works Well

- The paper has a defensible narrative: content routing is motivated from CATANet, DPR changes only the routing mechanism, and the Discussion correctly avoids claiming a large PSNR gain.
- The experiment section reports standard datasets, PSNR/SSIM protocol, checkpoint selection, and ablation caveats instead of hiding the small margins.
- Table 1 is dense but useful: it covers five benchmarks, three scales, and a strong baseline set.
- Figures 1, 2, 3, 4, 6, and 7 are present and generally compile cleanly; the visual comparison and routing maps support the story qualitatively.
- The bibliography has 45 cited entries and no obvious missing DOI/url/eprint fields by rough scan.

## Submission Blockers

1. Front matter is still anonymous/placeheld.

   `main_mdpi.tex` has `Anonymous Author`, `anonymous@example.com`, `Affiliation 1`, and placeholder ORCID. This is fine only for an anonymous review build, but Applied Sciences submissions are normally single-blind and require real author metadata unless the journal explicitly asks for anonymization.

   Relevant source: `paper/main_mdpi.tex:43-57`, `paper/main_mdpi.tex:127-135`.

2. The "budget unchanged" claim conflicts with Table 2.

   The abstract/introduction/conclusion say DPR keeps the lightweight budget unchanged, but the efficiency section states DPRNet is larger and more expensive than CATANet at x4: 660K/52.14G vs. 535K/46.8G. This is not fatal, but the wording should become "keeps the backbone and tensor interface unchanged while remaining in the lightweight regime" or "comparable lightweight budget".

   Relevant source: `paper/main_mdpi.tex:67-78`, `paper/sections/1_introduction.tex:101-112`, `paper/sections/3_experiments_efficiency.tex:43-55`, `paper/sections/4_discussion.tex:57-67`.

3. The strongest scientific claim still needs stronger direct evidence.

   The CATANet margin is tiny, and the paper already admits the ablations are short-budget, shared-initialization, and single-seed. For submission-level robustness, add at least one of:

   - a same-code/same-training CATANet or TAB baseline under the exact DPRNet protocol;
   - a direct EMA-center vs. DPR ablation for x4;
   - multi-seed runs for the main x4 comparison and A3;
   - a stronger reframing that the main contribution is interpretability/mechanism rather than accuracy.

   Relevant source: `paper/sections/3_experiments_comparison.tex:39-55`, `paper/sections/3_experiments_ablation.tex:31-45`, `paper/sections/4_discussion.tex:71-86`.

4. The efficiency comparison is visibly incomplete.

   Table 2 has many `-` cells for Multi-Adds and latency, and the prose says the unified table is deferred to the final version. A submitted paper should either measure all baselines under one protocol, remove the unavailable latency column for competitors, or move the incomplete comparison to a caveat without saying "final version".

   Relevant source: `paper/tables/table2_efficiency.tex:32-56`, `paper/sections/3_experiments_efficiency.tex:66-76`.

5. Draft/internal wording leaks into the manuscript.

   The phrase "left for the final version" and "we do not fabricate" sound like draft process notes, not paper prose. Table 7's note includes "it must not be asserted before those runs exist", which is an internal instruction and should be removed. Replace with neutral wording such as "not evaluated in this study" or "not available under a unified protocol".

   Relevant source: `paper/sections/3_experiments_efficiency.tex:73-76`, `paper/tables/table2_efficiency.tex:54-56`, `paper/tables/table5_ablation_a3.tex:50-54`.

6. Figure 5 is clipped.

   The source sets `ymax=36`, but the plotted series include values around 66, 75, 94, and 95. In the PDF the peaks are cut off, which weakens the routing-confidence evidence. Raise the y-axis limit, use a log scale, or plot cumulative/normalized density differently.

   Relevant source: `paper/figures/fig5_xscore_hist.tex:40-50`, `paper/figures/fig5_xscore_hist.tex:63-74`.

7. References render as "In Proceedings of the Proceedings of ...".

   This comes from `booktitle` fields already starting with "Proceedings of ...", while the MDPI bibliography style adds "In Proceedings of the". Remove the leading "Proceedings of the" from conference `booktitle` values or change the BibTeX fields so the generated `.bbl` reads naturally.

   Relevant source: `paper/references.bib:12-120`, generated output in `paper/main_mdpi.bbl:9-60`.

8. Back matter is missing likely MDPI not-applicable statements.

   The MDPI template includes `\institutionalreview{...}` and `\informedconsent{...}` with "Not applicable" options. For a non-human/non-animal ML paper, add explicit not-applicable statements unless the target submission system says to omit them. Also decide whether an AI-tool acknowledgment is needed if generative AI was used in drafting.

   Relevant source: `paper/mdpi_template_reference.tex:401-420`, `paper/main_mdpi.tex:124-137`.

## Figure And Table Notes

- Figure 1: readable and logically useful. Caption is long but acceptable. If space allows, enlarge the TAB dataflow labels slightly.
- Figure 2: method flow is strong, but the diagram is small and the caption is very dense. Consider splitting "DPR pipeline" and "ablation switches" or increasing figure height.
- Figure 3: good summary plot, but the top DPRNet label is close to the axis boundary and CARN sits at the far right. Add a little x/y padding.
- Figure 4: the qualitative crops are too small for SR texture differences. Split into two figures or enlarge crops by reducing the number of compared methods per row. The current figure supports the story but is not visually persuasive enough for SR reviewers.
- Figure 5: must be fixed because of clipping.
- Figure 6: acceptable; could be improved by adding the mean entropy arrow or value labels.
- Figure 7: useful and visually strong. The caption correctly states colors are not comparable across blocks.
- Table 1: strong but very dense. Keep if page budget allows; otherwise move some older CNN baselines to supplementary.
- Tables 4--7: scientifically honest, but the notes are too process-oriented. Rewrite in publication prose and remove internal instructions.

## Content And Experiment Suggestions

- Add a direct same-protocol CATANet reproduction if possible. The current CATANet comparison is based on published numbers; when the gain is only 0.01--0.07 dB, reviewers can reasonably ask whether the difference survives implementation and checkpoint variance.
- Add at least one deployment-relevant timing point if the paper keeps "mobile/edge/resource-constrained" framing. A single RTX 4090 latency is useful for reproducibility but weak evidence for mobile or edge deployment.
- Consider a small from-scratch ablation for the most central switch: original TAB-like routing vs. dynamic prototype routing at x4. This would directly support C1, which is currently supported only indirectly.
- If additional experiments are too expensive, tighten the claims: present DPR as a confidence-aware and interpretable routing replacement that preserves CATANet-level accuracy with modest overhead, rather than as a clear accuracy improvement.
- The method section has one small prose error: line 138--139 in `sections/2_method.tex` opens a parenthesis but does not close it.

## MDPI Style Alignment

I checked the Applied Sciences instructions and two similar MDPI SR articles (`app15041806`, `app14020917`). The overall structure is compatible with MDPI style: Abstract, Keywords, Introduction, Method/Experiments, Discussion, Conclusion, and back matter. The style differences to fix are practical rather than structural:

- real author metadata for the submission build;
- MDPI-required/not-applicable back matter statements;
- cleaner ACS-style references;
- no draft-process wording in tables or prose;
- figure captions less overloaded where the figure is already dense.

## Verification Performed

- Confirmed PDF has 23 pages via macOS metadata.
- Extracted text page by page with PDFKit.
- Rendered every page and visually checked the pages containing Figures 1--7 and Tables 1--7.
- Searched compile logs for warnings/errors; no unresolved citations or fatal LaTeX errors were found. Remaining warnings are mostly hyperref PDF-string warnings from math in front matter and repeated `fancyhdr` `headheight` warnings.
- Searched references and generated `.bbl`; found the repeated "Proceedings of the Proceedings of" bibliography issue.

## Recommended Revision Order

1. Fix manuscript hygiene: placeholders, back matter, internal wording, missing parenthesis, reference rendering.
2. Fix Figure 5 and enlarge Figure 4.
3. Tighten all "unchanged budget" claims to match the measured cost.
4. Either complete unified efficiency measurements or remove the incomplete competitor latency framing.
5. Strengthen the direct CATANet/TAB evidence, or explicitly reposition the contribution around interpretability and routing behavior.
