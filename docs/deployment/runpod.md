# Runpod Setup

Runpod lets you train on remote GPUs instead of your local machine. FineFoundry integrates directly—you configure everything in the app and it handles launching pods, managing volumes, and streaming logs.

For the Training tab UI details, see the [Training Tab Guide](../user-guide/training-tab.md).

## How It Works

When you select Runpod as your training target, FineFoundry connects using your API key, ensures a network volume exists (mounted at `/data`), creates or reuses a pod template, and launches training. Checkpoints and adapters get written to the network volume so they persist after the pod terminates.

The default trainer image (`sbussiso/unsloth-trainer:latest`) bundles the same LoRA fine-tuning stack used for local Docker training.

## What You Need

- A Runpod account with billing set up
- A Runpod API key (see [Authentication](../user-guide/authentication.md))
- GPU availability in your desired region

## Setup Steps

### 1. Add Your API Key

In the Settings tab, paste your Runpod API key and click Test to verify it works. Then Save.

### 2. Create a Network Volume

In the Runpod console, create a network volume sized for your datasets and checkpoints. Back in FineFoundry's Training tab, use the Ensure Infrastructure controls to verify the volume is available and mounted at `/data`.

### 3. Set Up a Pod Template

Create a pod template in Runpod that uses the trainer image, mounts your volume at `/data`, and specifies your desired GPU/CPU/RAM. In the Training tab, point your configuration at this template and verify it with Ensure Infrastructure.

### 4. Launch Training

Select Runpod as your target, pick your dataset, configure hyperparameters, set an output directory under `/data/outputs/...`, and start. FineFoundry launches the pod, streams logs, and writes results to the network volume.

## Output Location

Training outputs go to `/data/outputs/...` on the network volume. They persist after the pod terminates, so you can download them from the Runpod console or any system that can access the volume.

## Troubleshooting

Common issues: wrong API key, volume not mounted at `/data`, template pointing to wrong volume or image, insufficient credits, no GPU availability. See [Training Issues](../user-guide/troubleshooting.md#training-issues) in the Troubleshooting Guide.
