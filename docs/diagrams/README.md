# Architecture diagrams

The repository preserves the original submission-era Mermaid/SVG diagram set and now also contains the **current HydroCore-v5 Mermaid sources** used by the rebased documentation.

## Current V5 sources

| Mermaid source | Audience | Embedded/referenced in |
|---|---|---|
| [authority-architecture-v5.mmd](authority-architecture-v5.mmd) | Judges / technical reviewers | [Authority and safety](../AUTHORITY_AND_SAFETY.md) |
| [hydrocore-v5.mmd](hydrocore-v5.mmd) | Technical / research reviewers | [Architecture](../ARCHITECTURE.md), [Final system](../FINAL_SYSTEM.md) |
| [model-lifecycle-v5.mmd](model-lifecycle-v5.mmd) | Technical / scientific reviewers | [Evaluation](../EVALUATION.md) |
| [offline-deployment-v5.mmd](offline-deployment-v5.mmd) | Judges / operators | [Installation](../INSTALLATION.md) |

These V5 sources are the current architecture/lifecycle/deployment diagrams. Matching V5 SVG exports have **not** been committed yet; do not substitute or relabel the older V4-era SVGs as V5. The Mermaid sources render natively in GitHub-flavored Markdown where referenced.

## Preserved submission-era/static exports

The older six-diagram set remains intact for historical provenance and existing release/offline-viewer references:

| Source / SVG | Historical audience/use |
|---|---|
| [judge-product-flow.mmd](judge-product-flow.mmd) / [SVG](judge-product-flow.svg) | Judge-facing product flow |
| [authority-architecture.mmd](authority-architecture.mmd) / [SVG](authority-architecture.svg) | Pre-V5 authority architecture |
| [hydrocore-v4.mmd](hydrocore-v4.mmd) / [SVG](hydrocore-v4.svg) | HydroCore-v4 architecture |
| [model-lifecycle.mmd](model-lifecycle.mmd) / [SVG](model-lifecycle.svg) | Pre-final-V5 lifecycle |
| [reference-incident-flow.mmd](reference-incident-flow.mmd) / [SVG](reference-incident-flow.svg) | Reference-incident workflow; still useful as workflow evidence. The `.mmd` source's final-stage label was corrected for the post-completion Replay-unavailable truth (see [Reference demo](../REFERENCE_DEMO.md#replay-availability)); the committed SVG export has not been regenerated to match and remains stale on that one label pending a browser-capable render |
| [offline-deployment.mmd](offline-deployment.mmd) / [SVG](offline-deployment.svg) | Historical deployment diagram |

## Rendering

The historical SVGs were rendered with `@mermaid-js/mermaid-cli@10` using Playwright Chromium. On a browser-capable machine, generate the V5 SVGs from the current sources, for example:

```bash
npx -y @mermaid-js/mermaid-cli@10 \
  -i docs/diagrams/hydrocore-v5.mmd \
  -o docs/diagrams/hydrocore-v5.svg
```

Repeat for the other three `*-v5.mmd` files and review the rendered SVGs before committing them. Until then, the `.mmd` files—not the historical SVGs—are the current V5 diagram authority.
