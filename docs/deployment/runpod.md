# Runpod Setup

FineFoundry can use **Runpod** to run training jobs on remote GPUs instead of your local machine. This page describes how that integration works and how to get set up.

For a user‑level walkthrough of the Training tab itself, see the **[Training Tab](../user-guide/training-tab.md)** guide.

______________________________________________________________________

## How FineFoundry uses Runpod

When you select **Runpod – Pod** as the training target in the Training tab, FineFoundry:

- Connects to Runpod using your **Runpod API key** (configured in the **[Settings Tab](../user-guide/settings-tab.md)**).
- Ensures a **Network Volume** exists and is mounted at `/data` inside pods.
- Ensures a **Pod Template** exists for your chosen image and hardware.
- Launches pods from that template to run your training job.
- Writes logs, checkpoints, and final adapters under `/data/outputs/...` on the network volume.
- By default, uses an **Unsloth-based trainer image** (`docker.io/sbussiso/unsloth-trainer:latest`) that runs LoRA fine-tuning on top of PyTorch, Hugging Face Transformers, bitsandbytes, and PEFT.

This lets you keep your training artifacts persistent across pods and accessible both from Runpod and (optionally) your local machine.

______________________________________________________________________

## Prerequisites

Before using Runpod with FineFoundry, you should have:

- A **Runpod account** with billing/credits set up.
- A **Runpod API key** (see **[Authentication](../user-guide/authentication.md)**).
- At least one supported **GPU type** available in your desired region.

______________________________________________________________________

## Step 1: Configure your Runpod API key

1. Open the **Settings** tab in FineFoundry.
1. In the **Runpod Settings** section, paste your API key.
1. Click **Test** to confirm it works, then **Save**.

If the test fails, refer to **Runpod API key issues** in the **[Troubleshooting Guide](../user-guide/troubleshooting.md#authentication-issues)**.

______________________________________________________________________

## Step 2: Create a Network Volume (mounted at `/data`)

In the Runpod console:

1. Create a **Network Volume** (size depends on your dataset and checkpoint needs).
1. Note its identifier.

In the Training tab:

1. Choose **Runpod – Pod** as the target.
1. In the Runpod infrastructure section, make sure the volume is referenced and mounted at `/data` (this is where FineFoundry expects to read/write training artifacts).

FineFoundry’s **Ensure Infrastructure** controls help you verify that the volume is available and correctly mounted.

______________________________________________________________________

## Step 3: Create or reuse a Pod Template

Still in the Runpod console:

1. Create a **Pod Template** that:
   - Uses a compatible training image. The recommended default is `docker.io/sbussiso/unsloth-trainer:latest`, which bundles the Unsloth training stack (PyTorch, Transformers, bitsandbytes, PEFT).
   - Mounts the Network Volume at `/data`.
   - Exposes the resources (GPU, CPU, RAM) you want.
1. Note the template identifier.

Back in FineFoundry’s Training tab:

1. Point the Runpod configuration at your template.
1. Use **Ensure Infrastructure** to confirm that the template can be queried and reused.

______________________________________________________________________

## Step 4: Launch a training job

In the Training tab:

1. Set **Training target** to **Runpod – Pod**.
1. Choose your dataset source (JSON or Hugging Face repo/split).
1. Configure hyperparameters (base model, LoRA, batch size, max steps, etc.).
1. Set an **Output dir** under `/data/outputs/...`.
1. Optionally enable **Push to Hub** (requires working Hugging Face auth).
1. Start the training job.

FineFoundry will launch a pod from your template, monitor status, and stream logs (where supported). Checkpoints and the final adapter will be written under the configured output directory on the network volume.

______________________________________________________________________

## Where logs and outputs are stored

By default, when training on Runpod:

- Logs and checkpoints are written under `/data/outputs/...` inside the pod.
- These paths map to your Network Volume, so they persist after the pod terminates.
- You can browse or download results from the Runpod console or any system that can mount/sync the volume.

For local runs (Docker on your own machine), the Training tab uses a similar pattern but mounts a host directory instead of a Runpod volume.

______________________________________________________________________

## Troubleshooting Runpod setup

If things do not work as expected, check:

- **Runpod API key issues** and **Network volume not found** in the **[Troubleshooting Guide](../user-guide/troubleshooting.md#training-issues)**.
- The **[Training Tab](../user-guide/training-tab.md)** documentation for details on the Runpod controls and status indicators.

Common issues include:

- Incorrect or missing API key.
- Network Volume not mounted at `/data`.
- Pod Template referencing the wrong volume or image.
- Insufficient credits or GPU availability in the selected region.
