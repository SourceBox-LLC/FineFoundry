# Training Tab

The Training tab lets you fine‑tune language models on your own data, either on **Runpod pods** or **locally via Docker**, and then quickly test the resulting adapter.

Use this tab to:

- Fine‑tune models with beginner‑friendly presets or full expert control
- Run the **same training script** on cloud GPUs (Runpod) or your local machine
- Save and reload complete training setups as reusable configurations (stored in database)
- Sanity‑check a completed local run with **Quick Local Inference**

![Training Tab](../../img/new/ff_training.png)

______________________________________________________________________

## Overview

Typical workflow:

1. Choose a **Training target** (Runpod pod or local Docker).
1. Pick a **Skill level** and (optionally) a **Beginner preset**.
1. Configure **dataset source**, **hyperparameters**, and **output directory**.
1. For Runpod: ensure infrastructure (volume + template) and start training on a pod.
1. For local: configure Docker image, host mount, GPU use, and start local training.
1. After a successful local run, use **Quick Local Inference** to test the adapter.
1. At any time, **Save current setup** as a config and reload it later.

______________________________________________________________________

## Layout at a Glance

### 1. Target & Skill Level

- **Training target**
  - **Runpod - Pod**: train on a remote GPU pod using Runpod.
  - **Local**: train in a local Docker container on your own GPU/CPU.
- **Skill level**
  - **Beginner**: simplifies choices and exposes safe presets.
  - **Expert**: shows full hyperparameter and infra controls.
- **Beginner preset** (visible in Beginner mode only)
  - Presets adapt to the current training target:
    - **Runpod - Pod**:
      - `Fastest (Runpod)` – favors throughput on stronger GPUs.
      - `Cheapest (Runpod)` – favors smaller/cheaper GPUs with more conservative params.
    - **local**:
      - `Quick local test` – short run with small batch for fast sanity checks.
      - `Auto Set (local)` – uses detected GPU VRAM to pick batch size, grad accumulation, and max steps that push your GPU reasonably hard without being reckless.

### 2. Dataset & Output

- **Dataset source**
  - **Database Session**: Select from your scrape history.
  - **Hugging Face**: `Dataset repo`, `Split`, optional `Config`.
- **Output directory**
  - Path used **inside the container** (usually under `/data/outputs/...`).
  - Mapped back to a real directory on your Runpod network volume or local host mount.

### 3. Hyperparameters

- **Base model** (default `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`).
- **Epochs**, **Learning rate**, **Per‑device batch size**, **Gradient accumulation steps**.
- **Max steps** – upper bound on steps; useful for short experiments.
- **Packing** – packs multiple short examples to improve throughput.
- **Auto‑resume / Resume from** – continue from latest or a specific checkpoint.
- **Push to Hub** – upload final adapters/weights; requires HF token and repo id.

### 4. Runpod (Remote) Section

Visible when Training target = **Runpod - Pod**:

- **Infrastructure helpers** – create/ensure network volume and template.
- **GPU selection** – choose GPU type and region in the template.
- **Start training on pod** – launches `train.py` inside the selected pod.
- **Logs & status** – streaming logs, status messages, and exit codes.

### 5. Local Docker Section

Visible when Training target = **local**:

- **Host data dir** – local folder mounted as `/data` in the container.
- **Docker image** – training image (e.g. `sbussiso/unsloth-trainer:latest`).
- **GPU usage** – whether to expose local GPUs (`--gpus all` when available).
- **Pass HF token to container** – forwards `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` so `--push` works inside the container.
- **Container name** – name for the training container.
- **Start Local Training / Stop** buttons.
- **Progress & logs** – status text, progress bar, timeline of log lines, and a **Download logs** button.

### 6. Quick Local Inference

Quick Local Inference is your **instant demo station** for freshly trained adapters. It appears after a **successful local Docker run** with a valid adapter path:

- **Status & meta**
  - Status line (idle/loading/generating, errors).
  - Meta line showing adapter path and base model.
- **Controls**
  - **Preset** dropdown: Deterministic / Balanced / Creative.
  - **Sample prompts from dataset** – Dropdown with 5 random prompts from your training dataset for quick testing. Click the refresh button to get new samples.
  - Prompt text area – Enter your own prompt or select a sample above.
  - Temperature slider.
  - Max new tokens slider.
  - **Run Inference** button.
  - **Clear history** button.
  - **Export chats** button – Save your prompt/response history to a text file.
- **Output**
  - Scrollable list of prompt/response pairs.
  - Placeholder text when there are no responses yet.

> 💡 **Where to see it:** The Quick Local Inference panel is visible near the bottom of the Training tab UI (see the Training tab screenshot in the main README and at the top of this guide).

> 💡 **Sample prompts:** After training completes, 5 random prompts from your training dataset are automatically loaded into the sample prompts dropdown. This lets you quickly verify your model learned from the training data without manually copying prompts.

For deeper prompting and a dedicated chat experience after you have a good adapter, see the [Inference Tab](inference-tab.md), which lets you select **any saved dataset** for sample prompts and includes a Full Chat View dialog.

When you click **Run Inference**:

- The button is disabled.
- A small **progress ring** appears next to the buttons.
- Status shows either:
  - *"Loading fine‑tuned model and generating response..."* (first call), or
  - *"Generating response from fine‑tuned model..."* (subsequent calls).
- The model uses proper **chat templates** for instruct models and includes **repetition penalty** to prevent degenerate outputs.
- Once the response is ready, the spinner disappears, the button is re‑enabled, and the output appears in the list.

### 7. Configuration (Saved Setups)

- **Mode selector / Configuration section** – manage saved training configs.
- **Saved config dropdown** – list of configs from the database, filtered by current training target.
- **Actions**
  - **Refresh list** – reload configs from the database.
  - **Load** – apply a saved config to all UI fields (dataset, hyperparameters, target, infra).
  - **Rename** – change config name from within the app.
  - **Delete** – remove a config from the database.
- **Save current setup** buttons
  - One in the Configuration section.
  - Additional convenience buttons near training controls.

Configs include:

- Training target (Runpod vs local).
- Dataset source and splits.
- Hyperparameters and skill level / Beginner preset.
- Runpod infra settings or local Docker settings.

All configurations are stored in the SQLite database. The last used config is tracked and **auto‑loads on startup**.

______________________________________________________________________

## Beginner vs Expert Flow

### Beginner Mode

Designed for users who want guardrails and good defaults.

- **Skill level = Beginner**:
  - Shows the **Beginner preset** dropdown.
  - Applies preset‑dependent defaults whenever skill level / preset / target changes.

#### Runpod (Beginner)

- **Fastest (Runpod)**
  - Higher learning rate and per‑device batch size.
  - Lower gradient accumulation.
  - Higher max steps (for more training under good GPUs).
- **Cheapest (Runpod)**
  - Lower learning rate.
  - Smaller batch sizes.
  - Higher gradient accumulation to keep effective batch size reasonable.

#### Local (Beginner)

- **Quick local test**
  - Short run with a small batch and low max steps.
  - Intended for smoke tests: checks wiring, dataset, and HF token without long runs.
- **Auto Set (local)**
  - Reads **local specs** (especially GPU VRAM) and assigns:
    - Per‑device batch size,
    - Gradient accumulation steps,
    - Max steps,
    - A safe learning rate for your tier.
  - Heuristics favor avoiding OOM on common consumer GPUs (e.g., 8–12 GB cards).
  - Good default for first runs when you're unsure what hyperparameters to pick.

### Expert Mode

- **Skill level = Expert**:
  - Exposes full hyperparameter controls directly.
  - Hides the Beginner preset dropdown.
  - Meant for users who already know which batch size / grad accum / learning rate they want.

You can still use **Save current setup** in Expert mode to snapshot tuned configurations.

______________________________________________________________________

## Runpod vs Local Flows

### Runpod Flow (Remote Training)

1. In **Training target**, choose **Runpod - Pod**.
1. Configure **dataset source**, **output dir**, and **hyperparameters**.
1. Use **Skill level** and **Beginner presets** if desired.
1. In the Runpod section:
   - Ensure **network volume** and **template** exist (Infrastructure helpers).
   - Pick a GPU type and region via the template.
1. Start training on the pod.
1. Monitor progress via the **Logs / Status** panel.
1. Optionally push adapters/weights to the Hub when training completes.

### Local Flow (Local Docker Training)

1. In **Training target**, choose **local**.
1. (Optional) Set **Skill level = Beginner** and pick a preset:
   - `Quick local test` for very short runs.
   - `Auto Set (local)` for tuned defaults based on your GPU.
1. Configure **dataset source**, **output dir**, and **hyperparameters**.
1. In the local Docker section:
   - Set **Host data dir** (e.g., `~/Desktop/test_data`).
   - Check **Use GPU** if you have a CUDA‑capable GPU.
   - Enable **Pass HF token to container** if you need `--push`.
1. Click **Start Local Training**.
1. Watch logs in the timeline; use **Stop** if needed.
1. After a successful run, the **Quick Local Inference** panel becomes visible and shows the detected adapter + base model.
1. Enter a prompt and click **Run Inference** to verify quality.

______________________________________________________________________

## Training Configurations (Save / Load)

Training configs are stored in the database and capture the **entire training setup**.

### Saving

- Click a **Save current setup** button.
- Enter a descriptive name.
- The app saves a configuration capturing:
  - Training target (Runpod vs local).
  - Dataset source + split / session ID.
  - Hyperparameters, skill level, and Beginner preset key.
  - Runpod infra or local Docker settings.
- The name is recorded as the **last used config** for auto‑load on startup.

### Loading

- Use the **Saved configuration** dropdown in the Configuration section.
- Click **Load** to apply it to the UI.
- The app:
  - Switches the Training target to match the config.
  - Populates dataset, hyperparameters, infra, and presets.
  - Updates the last used config marker.

Configurations are **filtered by Training target** in the dropdown to reduce mistakes (e.g., accidentally applying a Runpod‑specific config while in local mode).

______________________________________________________________________

## Under the Hood: Training stack

Under the hood, both **Runpod** and **local Docker** training paths run the
same `train.py` script inside an Unsloth‑based trainer image
(default `docker.io/sbussiso/unsloth-trainer:latest`). That image uses:

- **PyTorch** for accelerated training on CPU/GPU.
- **Hugging Face Transformers** for loading the base model (for example
  `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` by default).
- **bitsandbytes** for 4‑bit quantization where supported.
- **PEFT / LoRA** (via Unsloth) for parameter‑efficient fine‑tuning.

The default base models offered in the Training tab are Unsloth‑optimized
variants of popular open models. Fine‑tuned adapters and checkpoints are
written under your chosen **Output directory** inside the container
(typically `/data/outputs/...`) and then picked up by **Quick Local
Inference** and the [Inference Tab](inference-tab.md).

## Tips & Best Practices

- Start in **Beginner** mode until you've found stable hyperparameters.
- Use **Quick local test** for small, rapid experiments.
- Use **Auto Set (local)** for first runs on a new GPU.
- Save a **known‑good config** after a successful run; reload it for variants.
- Use **Packing** when most examples are short to improve throughput.
- Keep an eye on logs for early signs of OOM or dataset issues.

______________________________________________________________________

## Offline Mode

When **Offline Mode** is enabled, the Training tab enforces local-only workflows:

- **Runpod cloud training is disabled**.
  - The training target is forced to `Local`.
  - Runpod infra and control actions are disabled.
- **Hugging Face datasets and Hub push are disabled**.
  - Hugging Face remains visible in dropdowns but is disabled.
  - If you were previously set to Hugging Face as a dataset source, the UI resets to **Database**.
  - Push-to-Hub controls are disabled and the push toggle is forced off.

The UI shows an Offline banner at the top of the tab and inline helper text under key controls explaining why they are disabled.

______________________________________________________________________

## Troubleshooting

For detailed troubleshooting, see the main guide:

- [Troubleshooting](troubleshooting.md)

Highlights:

- CUDA OOM and exit code 137 for local Docker runs.
- HF authentication issues inside containers.
- Config‑related mistakes and how to avoid them with saved setups.

______________________________________________________________________

## Related Topics

- [Quick Start Guide](quick-start.md) – overall workflow.
- [Inference Tab](inference-tab.md) – run inference on trained adapters with Prompt & responses and Full Chat View.
- [Merge Datasets Tab](merge-tab.md) – combine multiple datasets.
- [Troubleshooting](troubleshooting.md) – common training issues.
- [Logging Guide](../development/logging.md) – debugging training runs.

______________________________________________________________________

**Next**: [Merge Datasets Tab](merge-tab.md) | **Previous**: [Build & Publish Tab](build-publish-tab.md) | [Back to Documentation Index](../README.md)
