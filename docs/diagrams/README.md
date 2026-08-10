# Architecture diagrams

Six diagrams per submission.txt SS14, each as a `.mmd` (Mermaid source) file. Mermaid
renders natively in GitHub-flavored markdown (fenced ` ```mermaid ` blocks) and in the
Artifacts viewer, so these are viewable without a separate renderer -- see each file
embedded in context below.

| File | Audience | Embedded in |
|---|---|---|
| [judge-product-flow.mmd](judge-product-flow.mmd) | Judges | [README](../../README.md#operator-workflow) |
| [authority-architecture.mmd](authority-architecture.mmd) | Judges / technical reviewers | [Final system](../FINAL_SYSTEM.md) |
| [hydrocore-v4.mmd](hydrocore-v4.mmd) | Technical / research reviewers | [Model card](../MODEL_CARD.md) |
| [model-lifecycle.mmd](model-lifecycle.mmd) | Technical reviewers | [Final system](../FINAL_SYSTEM.md) |
| [reference-incident-flow.mmd](reference-incident-flow.mmd) | Judges / users | [Reference demo](../REFERENCE_DEMO.md) |
| [offline-deployment.mmd](offline-deployment.mmd) | Judges / users | [Installation](../INSTALLATION.md) |

## Static SVG exports: not generated in this sandbox

submission.txt SS14.7 asks for a committed rendered `.svg` alongside each `.mmd` source.
`@mermaid-js/mermaid-cli` (`mmdc`) needs to launch a real Chromium browser process to
render SVG, and Chromium's own internal process-sandboxing needs the same kernel
namespace/`CAP_SYS_ADMIN`-class privileges this sandbox has confirmed it withholds for
Docker (see `reports/submission-readiness/sub3-docker-sandbox-limitation.md` for the
root-cause trail) -- `mmdc` fails with `Failed to launch the browser process!` /
`Syntax error: Unterminated quoted string` (Chromium's own launcher failing to exec its
child process) regardless of `--no-sandbox` flags passed through.

To generate the `.svg` files on a machine without this restriction:

```bash
npx -y @mermaid-js/mermaid-cli@10 -i docs/diagrams/judge-product-flow.mmd -o docs/diagrams/judge-product-flow.svg
# repeat for the other five .mmd files
```

The `.mmd` sources are the real, reviewed, correct diagram content; only the static
raster/vector export step needs a browser-capable environment this sandbox does not have.
