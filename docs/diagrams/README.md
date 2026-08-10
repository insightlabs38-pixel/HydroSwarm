# Architecture diagrams

Six diagrams per submission.txt SS14, each with a Mermaid source and a committed static
SVG export. Mermaid also renders natively in GitHub-flavored markdown; the SVGs make the
same reviewed content available to release artifacts and offline viewers.

| Source / SVG | Audience | Embedded in |
|---|---|---|
| [judge-product-flow.mmd](judge-product-flow.mmd) / [SVG](judge-product-flow.svg) | Judges | [README](../../README.md#operator-workflow) |
| [authority-architecture.mmd](authority-architecture.mmd) / [SVG](authority-architecture.svg) | Judges / technical reviewers | [Final system](../FINAL_SYSTEM.md) |
| [hydrocore-v4.mmd](hydrocore-v4.mmd) / [SVG](hydrocore-v4.svg) | Technical / research reviewers | [Model card](../MODEL_CARD.md) |
| [model-lifecycle.mmd](model-lifecycle.mmd) / [SVG](model-lifecycle.svg) | Technical reviewers | [Final system](../FINAL_SYSTEM.md) |
| [reference-incident-flow.mmd](reference-incident-flow.mmd) / [SVG](reference-incident-flow.svg) | Judges / users | [Reference demo](../REFERENCE_DEMO.md) |
| [offline-deployment.mmd](offline-deployment.mmd) / [SVG](offline-deployment.svg) | Judges / users | [Installation](../INSTALLATION.md) |

The SVGs were rendered with `@mermaid-js/mermaid-cli@10` using the repository's installed
Playwright Chromium. To regenerate on a browser-capable machine:

```bash
npx -y @mermaid-js/mermaid-cli@10 -i docs/diagrams/judge-product-flow.mmd -o docs/diagrams/judge-product-flow.svg
```
