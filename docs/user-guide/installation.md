# Installation

This page gives you a focused, step‑by‑step path to get FineFoundry running on your machine.

If you just want the fastest way to go from **clone → running app → first dataset**, you can:

- Follow the **[Quick Start Guide](quick-start.md)** for an end‑to‑end walkthrough, or
- Use the commands below as a concise reference.

______________________________________________________________________

## Supported platforms

FineFoundry is developed and tested with:

- **Python**: 3.10 or newer
- **Operating systems**: Windows, macOS, Linux
- **GPU (optional but recommended)**: NVIDIA GPU with recent drivers for training and 4‑bit models

You will also need:

- **Git** (optional but recommended) for cloning the repository
- **Internet access** to install dependencies and (optionally) talk to external services

For publishing datasets or using remote training, you will later configure:

- A **Hugging Face** account and access token (see [Authentication](authentication.md))
- Optionally, a **Runpod** account and API key for remote training

______________________________________________________________________

## Option 1: Install and run with uv (recommended)

`uv` is a fast Python package manager that this project’s CI and docs use. If you want the least amount of setup, use this option.

### 1. Install uv (if needed)

If you do not already have `uv` installed, you can install it into your current Python environment:

```bash
pip install uv
```

If you prefer to avoid adding `uv` globally, you can also use `pipx` or follow the official `uv` installation instructions, but the simple `pip install uv` is usually sufficient for local development.

### 2. Clone the repository

```bash
# Clone the repository
git clone https://github.com/SourceBox-LLC/FineFoundry.git FineFoundry-Core
cd FineFoundry-Core
```

### 3. Run the application

With `uv`, you do **not** need to create a virtual environment or install requirements manually. `uv` will create an isolated environment, resolve dependencies, and run the app in one command:

On macOS/Linux, you may need to run the `chmod +x` step **once** to mark the script as executable.

```bash
chmod +x run_finefoundry.sh
./run_finefoundry.sh

# Alternative (without the launcher script)
uv run src/main.py
```

The first run will take longer while dependencies are resolved and installed. Subsequent runs should be much faster.

If you want to pre-sync dependencies (for example, before travelling or working offline), you can run:

```bash
uv sync
```

and then launch the app with `./run_finefoundry.sh` (recommended) or `uv run src/main.py` as above.

______________________________________________________________________

## Option 2: Install with pip and a virtual environment

If you prefer to manage environments yourself, you can use the classic `venv` + `pip` workflow.

### 1. Clone the repository

```bash
git clone https://github.com/SourceBox-LLC/FineFoundry.git
cd FineFoundry-Core
```

> If you cloned into a different directory name, `cd` into that folder instead.

### 2. Create and activate a virtual environment

From the project root:

```bash
python -m venv venv
```

Then activate it:

- **Windows (PowerShell)**

  ```bash
  ./venv/Scripts/Activate.ps1
  ```

- **macOS/Linux**

  ```bash
  source venv/bin/activate
  ```

If your system uses `py` to select Python, you can run `py -3.10 -m venv venv` instead.

### 3. Install dependencies

With the virtual environment active, install the required packages:

```bash
pip install -e .
```

### 4. Launch the GUI

From the project root (with the virtual environment active):

```bash
python src/main.py
```

Alternatively, if you have `flet`’s CLI on your PATH, you can use:

```bash
flet run src/main.py
```

______________________________________________________________________

## Verifying your installation

After running the app (via `./run_finefoundry.sh`, `uv run src/main.py`, or `python src/main.py`):

- A **desktop window** should appear with tabs such as **Data Sources**, **Dataset Analysis**, **Merge Datasets**, **Training**, **Inference**, **Publish**, and **Settings**.
- You should not see import errors in the terminal. If you do, re-run dependency installation (`uv sync` or `pip install -e . --upgrade`).

To quickly exercise the app after installation:

- Follow **Your First Dataset** in the **[Quick Start Guide](quick-start.md)** to:
  - Scrape a small sample from 4chan.
  - Build a tiny train/validation split.
  - Optionally run a basic dataset analysis.

If those steps work, your installation is healthy.

______________________________________________________________________

## Common installation problems

If you hit issues during installation, check the **[Troubleshooting Guide](troubleshooting.md)**, especially:

- **"Python command not found"** or `python` vs `py` on Windows.
- **"uv command not found"** when using the uv-based flow.
- **"Module not found"** errors after launch (usually fixed by re-running dependency installation).
- **SSL/certificate errors** when installing dependencies.

Relevant sections:

- [Troubleshooting – Installation Issues](troubleshooting.md#installation-issues)
- [Troubleshooting – Logging & Debugging](troubleshooting.md#logging--debugging) for enabling debug logs

If the Troubleshooting guide does not cover your case, please open an issue on GitHub with:

- Your OS and Python version
- The exact command you ran
- The full error message and any relevant logs

______________________________________________________________________

## Next steps

Once FineFoundry is installed and launching correctly:

- Walk through the **[Quick Start Guide](quick-start.md)** to create your first dataset.
- Read the **[GUI Overview](gui-overview.md)** to understand the main tabs.
- Set up tokens and API keys in **[Authentication](authentication.md)** and the **Settings** tab.

From there, you can explore the detailed tab guides (Data Sources, Publish, Training, Inference, Merge, Analysis) and start iterating on your own datasets and models.
