# Docker & Containerized Workflows

FineFoundry runs as a desktop app, not a containerized service. Local training runs natively via a Python subprocess—no Docker required. Docker is only used for Runpod cloud training, where the same trainer image runs inside Runpod pods.

## Local Training (Native)

When you select "Local" as your training target, FineFoundry runs the Unsloth trainer directly on your machine as a Python subprocess. Checkpoints and outputs are written to your configured output directory. This approach is simpler to set up and avoids Docker permission issues.

See the [Training Tab](../user-guide/training-tab.md) for configuration details.

## Runpod Training (Docker)

Runpod training uses the `sbussiso/unsloth-trainer:latest` Docker image, which bundles the same LoRA fine-tuning stack. When you train on Runpod, the pod pulls this image and runs your training job inside a container. The `/data` mount path stores checkpoints on the network volume.

## Running the GUI in a Container

This isn't officially supported. If you want to try it, you'll need to handle display/GUI forwarding (via browser, VNC, or similar) and configure GPU access correctly inside the container.

For most users, the recommended setup is to run the GUI on your local machine and offload heavy training to [Runpod](runpod.md) or run locally via native subprocess.
