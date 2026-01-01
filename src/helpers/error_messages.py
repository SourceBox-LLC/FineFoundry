"""User-friendly error message helpers.

Translates technical errors into plain language that beginners can understand.
"""


def friendly_error(error: Exception, context: str = "") -> str:
    """Convert a technical error into a user-friendly message.

    Args:
        error: The exception that occurred
        context: Optional context about what operation was being attempted

    Returns:
        A user-friendly error message
    """
    msg = str(error).lower()

    # Network errors
    if "connection" in msg or "timeout" in msg or "network" in msg:
        return "Couldn't connect to the internet. Check your connection and try again."

    if "ssl" in msg or "certificate" in msg:
        return "Secure connection failed. Try disabling your VPN or proxy, or check your network settings."

    if "403" in msg or "forbidden" in msg:
        return "Access denied. You may need to check your credentials or permissions."

    if "401" in msg or "unauthorized" in msg:
        return "Authentication failed. Check that your API key or token is correct and not expired."

    if "404" in msg or "not found" in msg:
        return "The requested resource wasn't found. Check the URL or path and try again."

    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return "Too many requests. Wait a minute and try again."

    # File/storage errors
    if "permission denied" in msg or "access denied" in msg:
        return "Permission denied. Try running as administrator or check folder permissions."

    if "no space" in msg or "disk full" in msg:
        return "Not enough disk space. Free up some space and try again."

    if "file name too long" in msg:
        return "The file path is too long. Try using a shorter name or different location."

    if "no such file" in msg or "does not exist" in msg:
        return "File or folder not found. Check the path and try again."

    # GPU/CUDA errors
    if "cuda" in msg and ("out of memory" in msg or "oom" in msg):
        return "GPU ran out of memory. Try reducing batch size or using a smaller model."

    if "cuda" in msg and "not available" in msg:
        return "No GPU detected. Make sure CUDA drivers are installed, or use CPU mode."

    if "cuda" in msg:
        return f"GPU error: {str(error)[:100]}. Try restarting the app or your computer."

    # Model/training errors
    if "model" in msg and "not found" in msg:
        return "Model not found. Check the model name or path."

    if "tokenizer" in msg:
        return "Problem loading the model's tokenizer. The model may be incompatible."

    if "checkpoint" in msg:
        return "Problem with checkpoint. It may be corrupted or from an incompatible version."

    # Database errors
    if "database" in msg or "sqlite" in msg:
        return "Database error. Try restarting the app. If the problem persists, the database may be corrupted."

    # JSON errors
    if "json" in msg or "decode" in msg:
        return "Invalid data format. The file may be corrupted or in the wrong format."

    # API-specific errors
    if "hugging" in msg or "hf_" in msg:
        return f"Hugging Face error: Check your token has write permissions. Details: {str(error)[:80]}"

    if "runpod" in msg:
        return f"RunPod error: Check your API key and account credits. Details: {str(error)[:80]}"

    # Generic fallback with context
    if context:
        return f"{context} failed: {str(error)[:100]}"

    # Last resort - clean up the error message
    clean_msg = str(error)
    if len(clean_msg) > 150:
        clean_msg = clean_msg[:150] + "..."
    return f"Something went wrong: {clean_msg}"


def friendly_scrape_error(error: Exception, source: str = "") -> str:
    """Specialized error messages for scraping operations."""
    msg = str(error).lower()

    if "rate limit" in msg or "429" in msg:
        return f"The {source or 'website'} is limiting requests. Wait a few minutes and try again with a longer delay."

    if "403" in msg or "forbidden" in msg:
        return f"Access to {source or 'the website'} was blocked. Try using a proxy or VPN."

    if "timeout" in msg:
        return f"Request timed out. The {source or 'website'} may be slow. Try again or increase the delay."

    if "no threads" in msg or "no posts" in msg or "empty" in msg:
        return f"No data found on {source or 'the website'}. Try different settings or a different URL."

    return friendly_error(error, f"Scraping {source}" if source else "Scraping")


def friendly_training_error(error: Exception) -> str:
    """Specialized error messages for training operations."""
    msg = str(error).lower()

    if "out of memory" in msg or "oom" in msg:
        return "GPU ran out of memory. Try: (1) Reduce batch size, (2) Increase gradient accumulation, (3) Use a smaller model."

    if "nan" in msg or "diverged" in msg:
        return "Training became unstable. Try: (1) Lower learning rate, (2) Use gradient clipping, (3) Check your data for issues."

    if "dataset" in msg and ("empty" in msg or "no data" in msg):
        return "No training data found. Make sure you've selected a dataset with data in it."

    if "killed" in msg or "137" in msg:
        return "Training was killed (likely out of memory). Reduce batch size or use cloud GPUs."

    return friendly_error(error, "Training")


def friendly_inference_error(error: Exception) -> str:
    """Specialized error messages for inference operations."""
    msg = str(error).lower()

    if "adapter" in msg or "lora" in msg:
        return "Couldn't load the trained adapter. Make sure training completed successfully."

    if "tokenizer" in msg:
        return "Problem with the model's tokenizer. The base model may have changed."

    if "out of memory" in msg:
        return "Not enough GPU memory to run inference. Close other programs or try a smaller model."

    return friendly_error(error, "Running inference")
