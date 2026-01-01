# Settings Tab

This is where you set up your accounts and preferences. You only need to do this once—your settings are saved automatically.

## What Can You Set Up Here?

| Setting | What It's For |
|---------|---------------|
| **Hugging Face** | Sharing datasets and models online |
| **Runpod** | Training on cloud GPUs |
| **Proxy** | Routing traffic through Tor or a VPN |
| **Ollama** | Auto-generating descriptions (optional) |

______________________________________________________________________

## Hugging Face Setup

Needed if you want to share datasets or trained models on Hugging Face.

### How to Set Up

1. **Create a free account** at [huggingface.co](https://huggingface.co/join)
2. **Get your token:**
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Give it a name and select **"Write"** access
   - Copy the token
3. **In FineFoundry:**
   - Paste the token in the Hugging Face field
   - Click "Test" to verify it works
   - Click "Save"

**Don't have an account yet?** That's fine! You can still collect data and train locally without Hugging Face.

______________________________________________________________________

## Runpod Setup

Needed if you want to train on cloud GPUs (faster, but costs money).

### How to Set Up

1. **Create a Runpod account** at [runpod.io](https://runpod.io)
2. **Add payment info** (training costs ~$0.50-$2 per hour)
3. **Get your API key:**
   - Go to [runpod.io/console/user/settings](https://runpod.io/console/user/settings)
   - Copy your API key
4. **In FineFoundry:**
   - Paste the API key
   - Click "Test" to verify
   - Click "Save"

**Don't want to pay?** No problem! You can train on your own computer for free if you have a GPU.

______________________________________________________________________

## Proxy Settings (Optional)

Only needed if you want to route your data collection through Tor or another proxy for privacy.

### How to Set Up Tor

1. **Install Tor** on your computer
2. **Start Tor** and make sure it's running
3. **In FineFoundry:**
   - Enable proxy
   - Enter: `socks5h://127.0.0.1:9050`
   - Click "Save"

**Don't need privacy features?** Leave this off—everything works fine without it.

______________________________________________________________________

## Ollama Setup (Optional)

Ollama can auto-generate nice descriptions for your datasets and models. Completely optional.

### How to Set Up

1. **Install Ollama** from [ollama.ai](https://ollama.ai)
2. **Start Ollama** on your computer
3. **In FineFoundry:**
   - Enable Ollama
   - URL is usually `http://localhost:11434`
   - Pick a model from the dropdown
   - Click "Test" then "Save"

**Don't want this?** Skip it! You can always write descriptions manually.

______________________________________________________________________

## System Check

At the bottom of Settings, there's a "System Check" panel. This runs tests to make sure everything is working correctly.

**When to use it:**
- After first installing FineFoundry
- If something seems broken
- When reporting bugs (attach the log)

Click "Run System Check" and wait for results. Green = good, red = problem.

______________________________________________________________________

## Tips

- **Test before saving** — Always click "Test" after entering credentials
- **Save your work** — Click "Save" to keep your settings
- **One-time setup** — You usually only need to do this once

______________________________________________________________________

**Previous**: [Analysis Tab](analysis-tab.md) | [Back to Documentation Index](../README.md)
