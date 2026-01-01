# Inference Tab

This is where you chat with your trained AI model to see how well it learned! Think of it as a testing playground where you can ask questions and see how your model responds.

![Inference Tab](../../img/new/ff_inference.png)

## How It Works

1. **Select a trained model** — Pick from your completed training runs
1. **Type a message** — Ask it anything!
1. **See the response** — Watch how your AI answers

It's like texting with your trained AI to see if it learned what you wanted.

______________________________________________________________________

## Getting Started

### Step 1: Select Your Model

1. Choose a **base model** from the first dropdown
1. Choose a **training run** from the second dropdown (only completed runs appear)
1. Wait for the green "ready" status

### Step 2: Start Chatting

1. Type your message in the text box
1. Click **"Generate"**
1. Read the AI's response below

Try different questions to see how well your model performs!

______________________________________________________________________

## Response Styles

Choose how your AI should respond:

| Style | Best For |
|-------|----------|
| **Deterministic** | Consistent, predictable answers. Good for testing if training worked. |
| **Balanced** | General use. A mix of consistency and variety. |
| **Creative** | More varied, surprising responses. Fun for exploration. |

**Tip:** Start with "Deterministic" to verify your training worked, then try "Creative" to see what else it can do.

______________________________________________________________________

## Full Chat View

Want a more chat-like experience? Click **"Full Chat View"** to open a dedicated chat window.

![Full Chat View](../../img/new/ff_inference_chat_view.png)

This shows your conversation as message bubbles, just like a messaging app. Great for:

- Extended conversations
- Demos to show others
- Testing multi-turn dialogues

______________________________________________________________________

## Sample Prompts

Not sure what to ask? FineFoundry can suggest prompts from your datasets:

1. Select a dataset from the dropdown
1. Choose from the suggested prompts
1. The prompt auto-fills in your text box

Click the refresh button to get different suggestions.

______________________________________________________________________

## Common Questions

**Why does it say "validating"?**
FineFoundry is checking that your trained model files exist and are complete. This takes a few seconds.

**Why is the first response slow?**
The AI model needs to load into memory the first time. After that, responses are faster.

**The responses don't seem right. What's wrong?**
Your training data might need work. Try:

- Training for more steps
- Collecting more/better data
- Checking data quality in the Analysis tab

**Can I save my conversation?**
Yes! Use the export button to save your chat to a text file.

______________________________________________________________________

## Tips for Better Testing

- **Test with similar prompts** to your training data first
- **Try edge cases** — What happens with unusual questions?
- **Compare styles** — Same prompt, different response styles
- **Use the clear button** to start fresh conversations

______________________________________________________________________

**Next**: [Merge Datasets Tab](merge-tab.md) | **Previous**: [Training Tab](training-tab.md) | [Back to Documentation Index](../README.md)
