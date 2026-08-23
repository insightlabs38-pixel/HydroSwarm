# HydroSwarm judge documentation portal

This branch contains the presentation layer for the hackathon documentation site. The canonical project documentation remains in the frozen repository snapshot.

- Frozen source: `4bbf6fa3ff9f68c99e111ca3abdeaeb6e4a6c2f9`
- Release: `v0.2.1`
- Site framework: Material for MkDocs
- Deployment: GitHub Pages via Actions
- Default reading theme: light
- Project identity: HydroSwarm navy with optional dark mode

## Source model

The workflow checks out this `docs-site` branch for site configuration and separately checks out the exact frozen source SHA above into `_frozen`. `prepare_site.py` then builds `docs-portal/generated/` from an explicit allowlist in `content-map.json`.

Do not edit `generated/`; it is a build artifact and is not committed.

## One-time GitHub Pages setup

After the workflow exists, open **Repository Settings → Pages → Build and deployment** and set **Source** to **GitHub Actions**. If the first deployment job ran before Pages was enabled, rerun the failed workflow after changing that setting.

The expected public URL is:

`https://insightlabs38-pixel.github.io/HydroSwarm/`

## Local validation

For an exact local build, make a separate checkout/worktree at the frozen SHA, install `requirements.txt`, then run:

```bash
python docs-portal/scripts/prepare_site.py \
  --source-root /path/to/frozen/HydroSwarm \
  --portal-root docs-portal \
  --output docs-portal/generated \
  --frozen-ref 4bbf6fa3ff9f68c99e111ca3abdeaeb6e4a6c2f9

mkdocs build --strict --config-file docs-portal/mkdocs.yml
```

The build fails if an allowlisted source is missing, a generated internal Markdown link is broken, or a mutable `main` GitHub artifact URL remains in the rendered documentation source.
