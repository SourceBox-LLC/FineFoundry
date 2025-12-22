# Docker Deployment

FineFoundry runs as a desktop app, not a containerized service. However, Docker plays a key role in local training—the Training tab can launch training jobs inside a Docker container on your machine.

## Local Training with Docker

When you select "Local" as your training target, FineFoundry runs the training script inside a Docker container. A host directory gets mounted as `/data` in the container, and checkpoints and outputs are written there. The default image (`sbussiso/unsloth-trainer:latest`) is the same one used for Runpod training, so you get the same LoRA fine-tuning stack whether you train locally or in the cloud.

See the [Training Tab](../user-guide/training-tab.md) for configuration details.

## Running the GUI in a Container

This isn't officially supported. If you want to try it, you'll need to handle display/GUI forwarding (via browser, VNC, or similar) and configure GPU access correctly inside the container.

For most users, the recommended setup is to run the GUI on your local machine and offload heavy training to [Runpod](runpod.md) or local Docker training launched from the GUI.
