
# Returns the configuration for the chosen model
# chosen model - one of the options:
# "gpt2-small (124M)"
# "gpt2-medium (355M)"
# "gpt2-large (774M)"
# "gpt2-xl (1558M)"
# returns a dictionary with the model configuration, and the model size
def GetModelConfig(chosen_model):
    BASE_CONFIG = {
        "vocab_size": 22,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }



    BASE_CONFIG.update(model_configs[chosen_model])
    model_size = chosen_model.split(" ")[-1].lstrip("(").rstrip(")")

    return BASE_CONFIG, model_size