# Training Tab

This is where you teach an AI model using your collected data. Think of it like training a pet—you show it examples of what you want, and it learns to respond similarly.

![Training Tab](../../img/new/ff_training.png)

## The Big Picture

Training works in three simple steps:

1. **Pick where to train** — Your computer or cloud GPUs
2. **Choose your data** — Select the dataset you collected earlier
3. **Click Start** — Watch as the AI learns from your examples

When training finishes, you can immediately test your model to see what it learned!

______________________________________________________________________

## Where Should I Train?

You have two options:

### Option 1: Train on Your Computer (Free)

Best if you have a gaming graphics card (GPU) with at least 8GB of memory.

**Pros:**
- Free
- Private—your data stays on your machine
- Good for small experiments

**Cons:**
- Needs a decent GPU
- Can be slow for large datasets

### Option 2: Train in the Cloud (Runpod)

Best if you don't have a GPU or want faster training.

**Pros:**
- Works without a GPU
- Much faster for large jobs
- Powerful hardware available

**Cons:**
- Costs money (typically $0.50-$2 per hour)
- Requires a Runpod account

![Auto Set preset](../../img/new/ff_auto_set.png)

______________________________________________________________________

## How to Train (Step by Step)

### Training on Your Computer

1. **Select "Local"** as your training target
2. **Choose your dataset** from the dropdown (your collected data)
3. **Pick a preset:**
   - **Quick local test** — Fast 2-minute test to make sure everything works
   - **Auto Set** — Automatically configures settings based on your GPU
4. **Click "Start Local Training"**
5. **Watch the logs** — You'll see progress updates as training happens

When it's done, a "Quick Local Inference" panel appears so you can test your model immediately!

### Training on Runpod (Cloud)

1. **Set up Runpod first** (one-time):
   - Go to the Settings tab and add your Runpod API key
   - Back in Training, click "Ensure Infrastructure" to create storage space
2. **Select "Runpod - Pod"** as your training target
3. **Choose your dataset** from the dropdown
4. **Pick a GPU type** (RTX 4090 is a good balance of speed and cost)
5. **Click "Start Training"**

Training on the cloud is faster but costs money. Watch your Runpod dashboard to monitor costs.

______________________________________________________________________

## Beginner vs Expert Mode

**Beginner Mode** (Recommended for new users)
- Simple presets that "just work"
- Automatically configures complex settings
- Harder to make mistakes

**Expert Mode** (For advanced users)
- Full control over all settings
- Lets you fine-tune performance
- Easy to break things if you're not careful

Stick with Beginner mode until you've done several successful training runs!

______________________________________________________________________

## Testing Your Trained Model

After training finishes successfully, a "Quick Local Inference" panel appears. This lets you chat with your newly trained model:

1. **Type a prompt** in the text box
2. **Click "Run Inference"**
3. **See the response** — Does it match what you expected?

Try several prompts to see how well the model learned. There's also a dropdown with sample prompts from your training data.

**Generation styles:**
- **Deterministic** — Consistent, predictable responses
- **Balanced** — Good for general testing
- **Creative** — More varied, sometimes surprising responses

______________________________________________________________________

## Saving Your Settings

Found settings that work well? Save them!

1. Click **"Save current setup"**
2. Give it a descriptive name (like "Reddit bot v1")
3. Your settings are now saved

Next time, just select your saved config from the dropdown and click "Load" to restore everything.

**Bonus:** The app remembers your last used settings and loads them automatically when you restart.

______________________________________________________________________

## Common Problems

**"Out of memory" error**
Your GPU doesn't have enough space. Try:
- Using the "Quick local test" preset (uses less memory)
- Reducing batch size in Expert mode
- Using Runpod instead

**Training seems stuck**
Check the logs—it might just be slow. Training 8B models can take hours depending on your data size and hardware.

**Model gives weird responses**
Your training data might need work. Try:
- Collecting more data
- Using the Analysis tab to check data quality
- Training for more steps

For more help, see the [Troubleshooting Guide](troubleshooting.md).

______________________________________________________________________

## Tips for Better Results

- **Start small** — Do a quick test run before committing to hours of training
- **Check your data first** — Use the Analysis tab to spot problems before training
- **Save working configs** — When something works, save those settings!
- **Watch the logs** — They'll warn you about problems early

______________________________________________________________________

**Next**: [Inference Tab](inference-tab.md) | **Previous**: [Publish Tab](build-publish-tab.md) | [Back to Documentation Index](../README.md)
