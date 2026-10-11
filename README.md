# Cancer Detection API

Detecting cancer from bio-images.

## CIAO documentation and architecture sync

This repository can automatically generate a project documentation report and an architecture diagram whenever source changes are pushed. The generated files are committed to the repository root as `documentation.pdf` and `architecture.png` by default.

### Enable automatic sync

1. In GitHub, open **Settings → Actions → General** and make sure Actions are enabled.
2. Under **Workflow permissions**, select **Read and write permissions**, then save. The workflow requests `contents: write` so it can commit the generated files.
3. Add `.github/workflows/update-ciao-docs.yml` to this repository and push it to GitHub. The workflow uses CIAO from `176Anjali/CIAO-demo` and runs Ollama on the GitHub Actions runner; you do not need to install Ollama or Docker on your computer for GitHub sync.
4. Push a source-code change, or open **Actions → Update CIAO documentation → Run workflow** to start it manually.
5. Watch the run under **Actions → Update CIAO documentation**. The first run can take longer while the runner installs Ollama and downloads the model. On success, download `documentation.pdf` and `architecture.png` from the repository root.

The report records the analyzed commit and generation time. Generated commits include `[ciao-generated]` to prevent a repeat run loop. A failed generation does not replace the last successful files.

### Optional repository variables

In **Settings → Secrets and variables → Actions → Variables**, add any of these if you want different formats or model:

| Variable | Allowed values | Default |
| --- | --- | --- |
| `CIAO_DOCUMENT_FORMAT` | `pdf`, `docx` | `pdf` |
| `CIAO_IMAGE_FORMAT` | `png`, `svg` | `png` |
| `CIAO_OLLAMA_MODEL` | Any model available to Ollama | `llama3.2:3b` |

The selected report and diagram format are committed to the repository root. The workflow checks out the exact source commit that triggered generation.

## Installation

Install the Python packages listed in `requirements.txt`, then follow the application entry point and run instructions for your local environment. GitHub documentation sync is configured separately by the workflow above.

## Project sections

- `app/`, `prediction/`, `preprocessing/`, and `training/` contain application and model pipeline code.
- `configs/` and `models/` contain configuration and model assets.
- `dataset/` contains project data.
- `tests/` contains automated checks.
