# Inference Tab

The Inference tab lets you run **local inference** against fine‑tuned adapters, using the same `local_inference` helpers that power **Quick Local Inference** in the Training tab – but with a richer prompt history view and a dedicated **Full Chat View** overlay.

Use this tab to:

- Run your **latest local adapter** or any adapter directory on disk
- Control **temperature** and **max new tokens** with simple sliders
- Quickly switch between **Deterministic / Balanced / Creative** presets
- View a running list of **prompt / response pairs**
- Pop out a focused **Full Chat View** dialog for multi‑turn chats

![Inference Tab](../../img/ff_inferance.png)

______________________________________________________________________

## Overview

Typical workflow:

1. Pick a **base model** and **adapter directory**, or import from **latest local training**.
1. Wait for the adapter path to be **validated** (spinner + snackbar feedback).
1. Once validated, the **Prompt & responses** section unlocks.
1. Type a prompt, choose a preset, and click **Generate** to get responses.
1. Optionally click **Full Chat View** to open a focused dialog for multi‑turn chat.

The Inference tab is intended as a **lightweight, local playground** for your fine‑tuned models, separate from training. It shares model loading and caching with Quick Local Inference but adds:

- Immediate **adapter validation** on directory selection
- A **shared conversation history** between the main view and Full Chat View
- A full‑screen‑style chat experience suitable for demos and deeper testing

______________________________________________________________________

## Layout at a Glance

### 1. Model & Adapter

At the top of the tab you configure which model to run:

- **Status line**
  - Shows the current state: idle, validating adapter, ready, or error.
  - Errors are highlighted in **red** when the adapter directory is invalid.
- **Meta line**
  - Shows the currently selected adapter path and base model once validated.
- **Base model**
  - Text field for the base model id (e.g. `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`).
- **Adapter directory**
  - Text field for the adapter folder on disk (e.g. `/path/to/outputs/run/adapter`).
  - **Browse…** button opens a directory picker.
  - **Use latest local training** imports the adapter + base model from the most recent local training run (as tracked by the Training tab).

#### Adapter validation

Any time you select or change the adapter directory – either via **Browse…** or **Use latest local training** – the app:

1. Shows a **loading spinner** while it checks the folder.
1. Verifies that the directory exists and contains typical **LoRA adapter artifacts**, such as:
   - `adapter_config.json`, or
   - Weight files like `*.safetensors` or `*.bin`.
1. If validation **fails**:
   - Status text turns red with a descriptive error.
   - A **snackbar** appears explaining the problem.
   - The **Prompt & responses** controls are **locked**.
1. If validation **succeeds**:
   - Status shows: *"Adapter directory validated. Ready for inference."*
   - Meta line updates with adapter + base model.
   - The **Prompt & responses** section is **unlocked**.

This prevents silent failures when pointing at the wrong folder (e.g., parent run directory instead of the adapter subdirectory).

______________________________________________________________________

## 2. Prompt & Responses

Once the adapter is validated, the lower half of the tab becomes active.

- **Preset** dropdown
  - `Deterministic` – lower temperature, shorter max tokens.
  - `Balanced` – default middle‑of‑the‑road settings.
  - `Creative` – higher temperature and longer responses.
- **Prompt** text area
  - Multi‑line field for your instruction or question.
- **Sliders**
  - **Temperature** – controls randomness.
  - **Max new tokens** – upper bound on generated tokens.
- **Buttons**
  - **Generate** – runs inference using the shared `local_inference` helper.
  - **Clear history** – clears the shared conversation history (both here and in Full Chat View).
  - **Full Chat View** – opens the full‑screen‑style chat dialog (see below).
- **Output list**
  - Shows a scrollable list of **Prompt / Response** pairs.
  - A subtle placeholder message appears when there are no responses yet.

> 💡 **On the UI:** In the Inference tab screenshot above, this section lives under the "Prompt & responses" header. The **Full Chat View** button sits just below the history box on the right.

### Generation behavior

When you click **Generate**:

- The app checks that the adapter directory still looks valid.
- The **Generate** button is temporarily disabled and a small **progress ring** appears.
- Status shows either:
  - *"Loading fine‑tuned model and generating response..."* on first call, or
  - *"Generating response from fine‑tuned model..."* on subsequent calls.
- The response is appended to the output list and recorded in the shared chat history.
- Status returns to an idle message once complete.

If inference fails (e.g., due to a model error), the status text shows the exception in red.

______________________________________________________________________

## 3. Full Chat View (Focused Chatbot)

Click **Full Chat View** to open a focused, demo-ready chat dialog built around the same adapter and base model:

![Full Chat View](../../img/ff_inferance_full_chat_view.png)

The dialog includes:

- A **header** with icon and title (*"Full Chat View"*).
- A large **chat area** showing alternating user and assistant bubbles.
- A **composer** row with:
  - Multiline message box,
  - **Send** button,
  - Small progress spinner while generating.
- **Actions** at the bottom:
  - **Clear history** – resets the shared conversation and both UIs.
  - **Close** – dismisses the dialog.

### Shared conversation history

The main Inference tab and Full Chat View share a single in‑memory **chat history**:

- Messages generated via **Generate** are added to the same history and mirrored into Full Chat View as user/assistant bubbles.
- Messages sent via **Send** in the dialog:
  - Are added to the shared history.
  - Are mirrored back into the main **Prompt & responses** list as prompt/response entries.
- When you reopen Full Chat View, it **rebuilds** its bubble list from the shared history, so the conversation looks the same in both places.
- **Clear history** (either in the main tab or in the dialog) wipes:
  - The shared chat history,
  - The prompt/response list in the main view,
  - The bubbles in Full Chat View,
  - And resets placeholders + status to an idle state.

______________________________________________________________________

## Tips & Best Practices

- Point the adapter directory directly at the **adapter subfolder** from training, not the parent run directory.
- If validation keeps failing, check that the folder really contains `adapter_config.json` or LoRA weight files.
- Use **Deterministic** preset when quickly verifying whether fine‑tuning did what you expect.
- Use **Creative** when exploring the qualitative behavior of your model.
- For quick smoke tests right after local training, you can still use **Quick Local Inference** in the Training tab; then move to the Inference tab for deeper prompting and chat.

______________________________________________________________________

## Related Topics

- [Training Tab](training-tab.md) – configure and run training jobs.
- [Quick Start Guide](quick-start.md) – overall workflow.
- [Troubleshooting](troubleshooting.md) – includes tips for training and adapter issues.
