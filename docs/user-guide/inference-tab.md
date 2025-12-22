# Inference Tab

The Inference tab is your playground for testing fine-tuned models. Pick a training run, load the adapter, and start chatting with your model to see how it performs. Unlike Quick Local Inference in the Training tab, this dedicated space lets you test against any dataset and offers a full chat view for extended conversations.

![Inference Tab](../../img/new/ff_inference.png)

## Getting Started

Select a base model and a completed training run from the dropdowns. FineFoundry validates the adapter directory automatically—you'll see a spinner while it checks, then either a green "ready" status or a red error if something's wrong.

Once validated, the prompt controls unlock. Type a prompt, pick a preset (Deterministic for predictable outputs, Balanced for general use, Creative for more variety), and click Generate. Your response appears in the output list below.

## Sample Prompts

Unlike the Training tab's Quick Local Inference (which pulls samples from the training dataset), here you can select any saved dataset from your database. This is useful for testing how your model handles data it wasn't trained on.

Pick a dataset from the dropdown, and you'll get 5 random prompts to choose from. Click the refresh button to get different samples. Selecting one fills the prompt box automatically.

## Full Chat View

For deeper testing or demos, click Full Chat View to open a focused chat dialog. It shows your conversation as alternating user/assistant bubbles, with a message composer at the bottom.

![Full Chat View](../../img/new/ff_inference_chat_view.png)

The main tab and Full Chat View share the same conversation history. Messages you generate in either place show up in both. Clear history resets everything.

## Generation Settings

The presets handle most use cases:

- **Deterministic** uses low temperature for consistent, predictable outputs—good for verifying your model learned what you intended
- **Balanced** sits in the middle and works for general testing
- **Creative** uses higher temperature and longer max tokens for more varied, exploratory responses

You can also adjust temperature and max tokens directly with the sliders if you want finer control.

## Under the Hood

Inference runs entirely on your local machine using Hugging Face Transformers, PEFT for loading LoRA adapters, and bitsandbytes for 4-bit quantization when you have a CUDA GPU. No external APIs are called—everything stays local.

## Tips

If adapter validation keeps failing, make sure the training run actually completed and produced an adapter directory with the expected files (`adapter_config.json` and weight files). Use Deterministic mode first to verify your fine-tuning worked as expected, then switch to Creative to explore the model's behavior more broadly.

For quick tests right after training, Quick Local Inference in the Training tab is convenient. Move to this Inference tab when you want a dedicated space for extended testing, different sample datasets, or the full chat experience.

______________________________________________________________________

## Related Guides

Set up and run training in the [Training Tab](training-tab.md). For the overall workflow, see the [Quick Start Guide](quick-start.md). If something's not working, check [Troubleshooting](troubleshooting.md).
