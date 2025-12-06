# Training API

This page provides a brief overview of FineFoundry's programmatic training interfaces and where to look in the codebase.

The primary implementation lives under:

- `src/helpers/training.py`
- `src/helpers/training_pod.py`
- `src/runpod/ensure_infra.py`
- `src/runpod/runpod_pod.py`

These modules support both **local Docker training** and **Runpod-based training** as exposed in the GUI Training tab. Rather than re-implementing the entire fine-tuning stack, FineFoundry orchestrates an **Unsloth-based trainer image** (by default `docker.io/sbussiso/unsloth-trainer:latest`) that runs LoRA fine-tuning on top of:

- PyTorch
- Hugging Face Transformers
- bitsandbytes
- PEFT / LoRA (via Unsloth)

From the API perspective, the helper functions are responsible for:

- Building a hyperparameter/configuration dictionary from UI or JSON config files.
- Ensuring Runpod infrastructure (network volume, template) or local Docker settings are in place.
- Constructing and launching training jobs with the appropriate container image, dataset flags, and Runpod/Docker parameters.
- Streaming logs and tracking basic training state.

Today, the best way to understand the training API is to:

- Read the **[Training Tab (GUI)](../user-guide/training-tab.md)** guide for high-level behavior.
- Inspect the helper modules above to see how configuration objects are built and how jobs are launched and monitored.

Future iterations of this page may add concrete examples and function-level references, but it is already safe to treat the helpers as a stable entry point for advanced, programmatic usage.
