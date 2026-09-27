# BaryBind — NeurIPS 2026 Oral

A self-contained academic project page for **Binding Multiple Modalities via Multimodal Wasserstein Barycenter**. Plain HTML, CSS and JavaScript; no build, backend, API keys, external fonts, or runtime dependencies.

## Open locally

Open `index.html` directly in a modern browser. All assets and scripts use relative paths. Alternatively, with Node.js installed:

```sh
npm start
```

Open the local address printed by the server. No `npm install` is required. You can also use `python3 -m http.server 8000` from this directory.

## Publish on GitHub Pages

1. Create a repository for the project page, or use a dedicated branch of an existing repository.
2. Upload this folder's **contents**, with `index.html` at the repository root. Include `assets`, `css`, `js`, and `.nojekyll`.
3. In the repository, open **Settings → Pages**.
4. Choose **Deploy from a branch**, select your branch (typically `main`) and **/ (root)**, then save.
5. GitHub will display the deployed address when the deployment finishes.

Relative URLs work for both a user site and a project subdirectory. No base-path configuration is necessary. The optional `package.json` and `preview.cjs` are only for local preview. This deliverable has not been pushed to GitHub or published.

## Files

- `index.html`: narrative, equations, paper metadata and abstract.
- `css/style.css`: responsive theme and reduced-motion support.
- `js/barycenter.js`: deterministic point-cloud illustrations and weight explorer.
- `js/simplex.js`: projected gap vectors and positive/negative geometry.
- `js/main.js`: sourced experiment tables, tabs, retrieval examples and citation copying.
- `assets/paper.pdf`: supplied paper. Slides are intentionally not distributed.
- `assets/figures`, `assets/generation`, `assets/retrieval`: original scientific imagery extracted from those materials.
- `CONTENT_SOURCES.md`: source locations and discrepancy decisions.
- `QA.md`: checks and limits.

## Scientific fidelity

Experimental values are transcribed from the supplied paper. Each results panel identifies its source table. Some tables and nearby prose differ; the selected source is recorded in `CONTENT_SOURCES.md`. Do not combine values from different experimental tables without identifying their configuration.

The explorer is a **toy fixed-correspondence geometric-median surrogate**, not a trained model, exact general OT solver, or experimental embedding. Its energy and normalized weight entropy are illustrative metrics. The simplex graphics are schematic projections; the high-dimensional Gram determinant is not measured from the displayed polygon.

Original qualitative images are not regenerated. Audio descriptions are conditioning labels, not playable audio. The supplied PDFs contain no playable recordings. The original class embedding visualization is reproduced as an image rather than fabricated point data.

## Editing

Change links and text in `index.html`; reported numbers live in `js/main.js`. Replace the paper PDF without renaming it to retain its link. No official volume or page numbers have been invented for the citation. The Oral designation follows the author's project brief.

The Pause motion control stops ambient animation. The page also honors `prefers-reduced-motion`. Tables and long equations scroll within their own containers on narrow screens. Tabs support Left/Right, Home and End; controls are keyboard accessible.

## Rights

The supplied paper, slides, and extracted scientific figures retain their original authorship and rights. This package does not grant additional rights to those materials or to third-party baseline outputs.
