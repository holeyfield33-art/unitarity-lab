# sanitize_config.py
import json
import os


def sanitize():
    config_path = "./merged_model/config.json"
    if not os.path.exists(config_path):
        print(f"[Sanitizer] ERROR: {config_path} not found. Run merge_adapters.py first.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 1. Remove residual quantization configs left over from training.
    keys_to_remove = ["quantization_config", "rope_scaling", "use_dynamic_ntk"]
    removed_keys = []

    for key in keys_to_remove:
        if key in config:
            config.pop(key)
            removed_keys.append(key)

    # 2. Reset tie_word_embeddings to avoid allocation misalignments in mlc_llm.
    config["tie_word_embeddings"] = False

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[Sanitizer] Cleaned config.json. Stripped keys: {removed_keys}")


if __name__ == "__main__":
    sanitize()
