# Content provenance

Sources supplied by the author:

- `neurips_2026(4).pdf` → `assets/paper.pdf` (35 pages).
- `Barybind.pdf` (20 slides): used for source verification and image extraction only; not distributed in this revision.

## Mapping

| Website content | Source |
| --- | --- |
| Title, five authors, affiliation, repository, abstract | Paper p. 1 |
| NeurIPS 2026 Oral designation | Explicit author brief; paper identifies NeurIPS 2026 |
| WB distribution objective | Paper Eq. 3, p. 4 |
| Map, initializers, dual potentials and alternating optimization | Paper pp. 4–5, Eq. 5; Appendix B.7 |
| Gaps and Gram determinant volume convention | Paper Eq. 9, p. 5; slide 11 |
| Complete symmetric BVC loss | Paper Eq. 10, p. 6 |
| DAM description and combined training objective | Paper Section 3.4, Eq. 12; slide 13 |
| Original embedding figure | Paper Figure 4, p. 6 |
| Zero-shot retrieval across four datasets | Paper Table 2, p. 8 |
| Classification | Paper Table 1, p. 7 |
| Missing video at inference | Paper Table 4, p. 8 |
| Missing audio during training | Paper Table 3, p. 8 |
| Five-modality retrieval | Paper Table 2, p. 8 |
| Training overhead | Paper Table 19, p. 25 |
| Generation FID | Paper Table 24, p. 27; slide 18 |
| Bird and dog generation images | Crops of original slide 18 |
| Semantic-consensus captions and identifiers | Paper Tables 20–21, p. 26; slide 14 |
| Beach and gaming retrieval comparison images | Original slide 16 |

## Explicit discrepancy decisions

1. The title slide's joint-author line omits Yan Yang. The website uses the complete five-author list on the paper's first page.
2. DiDeMo T-VA BaryBind T2V is **56.3** in Table 2, while surrounding prose says 56.1. The page uses **56.3 / 54.0** from Table 2.
3. The scalability appendix Table 22 contains values different from Table 2. The page consistently uses **Table 2** for its main and five-modality retrieval panels, and labels that choice. It does not silently mix those tables.
4. The paper's prose states an MSR-VTT T-VA gap of 2.8; Table 2's 56.1 and 53.6 give **2.5**. The page derives gaps directly from the displayed table values (absolute difference in percentage points).
5. Classification uses Table 1: BaryBind A **45.7 / 75.2**, V **48.3 / 76.4**, A+V **55.6 / 83.4**. Missing-video inference uses Table 4, with **49.4 / 78.3** under a distinct setting.
6. Paper Eq. 11 and the slides differ in their written sign for DAM. The page describes its binary-matching role and shows the combined objective without introducing a conflicting standalone DAM equation.
7. The volume is shown using the paper's convention `Vol² = det(RᵀR)`; no simplex factorial normalization is silently introduced. Graphics are schematic projections, not measured experimental volumes.

## Illustrations and derived quantities

Hero, anchor comparison, method diagrams, and simplex diagrams are conceptual. The toy explorer uses deterministic clouds and an iterative weighted geometric median for each fixed correspondence. Displayed energy is the average weighted Euclidean point distance divided by 200 coordinate units. Balance is normalized entropy `-sum(λ log λ)/log(5)`. At least one weight remains nonzero. These are not reported scientific results.

No external paper URL, author homepage, DOI, official page range, video, or audio file has been invented. The local paper PDF and the repository specified on paper p. 1 are the primary links.
