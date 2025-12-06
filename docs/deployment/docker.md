# Docker Deployment

FineFoundry is primarily designed to run as a desktop application on your local machine. It does **not** currently ship with an official, supported Docker image for the full GUI.

This page explains the current state of container usage and points you to the most relevant pieces.

______________________________________________________________________

## Local training via Docker

The main place Docker is used today is for **local training** from the Training tab:

- The Training tab can launch training jobs inside a local Docker container.
- A host directory is mounted into the container (typically mapped to `/data`).
- Checkpoints and outputs are written under that directory.
- By default, FineFoundry uses the same **Unsloth-based trainer image** as the Runpod flow (`docker.io/sbussiso/unsloth-trainer:latest`), so local and remote runs share the same LoRA fine-tuning stack.

For details, see:

- **[Training Tab](../user-guide/training-tab.md)** – look for the local Docker training section.

This flow is focused on training only and does **not** containerize the full GUI.

______________________________________________________________________

## Running the GUI in a container

Running the full Flet GUI inside a container is not currently documented or officially supported. If you attempt this yourself, keep in mind:

- You will need to handle display/GUI forwarding (for example, via a browser, VNC, or other remote desktop options).
- GPU access and drivers must be configured correctly inside the container if you want to use GPU‑accelerated features.

For most users, the recommended setup is to:

- Run the GUI on a local machine (using `uv run src/main.py` or `python src/main.py`), and
- Offload heavy training workloads to **Runpod** or local Docker training launched from the GUI.

See **[Runpod Setup](runpod.md)** for remote training, and the Training tab docs for local Docker training.
