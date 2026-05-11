#!/usr/bin/env python3
"""
model_manager.py — add or remove models from truss-native.

Usage:
    python3 model_manager.py add
    python3 model_manager.py remove
"""

import argparse
import os
import shutil
import sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
LLAMA_DIR    = os.path.join(MODELS_DIR, "llama")
MLX_DIR      = os.path.join(MODELS_DIR, "mlx")
YAML_PATH    = os.path.join(MODELS_DIR, "models.yaml")

def _load_yaml_text():
    with open(YAML_PATH, "r") as f:
        return f.read()

def _parse_models_block(text):
    """Return list of dicts parsed from the 'models:' block — best effort."""
    try:
        import yaml
        data = yaml.safe_load(text)
        return data.get("models") or []
    except ImportError:
        pass
    # Fallback: very small hand-rolled parser (id / type / chat_format / path lines)
    models = []
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- id:"):
            if current:
                models.append(current)
            current = {"id": stripped[len("- id:"):].strip()}
        elif current is not None:
            for key in ("type", "chat_format", "path", "flash_attention"):
                prefix = key + ":"
                if stripped.startswith(prefix):
                    val = stripped[len(prefix):].strip()
                    current[key] = val
    if current:
        models.append(current)
    return models

def _append_model_entry(spec: dict):
    """Append a new model entry to models.yaml, preserving existing content."""
    fa_line = ""
    if spec["type"] == "llama":
        fa_line = f"\n    flash_attention: true"

    entry = (
        f"\n  - id: {spec['id']}\n"
        f"    type: {spec['type']}\n"
        f"    chat_format: {spec['chat_format']}\n"
        f"    path: {spec['path']}"
        f"{fa_line}\n"
    )

    with open(YAML_PATH, "a") as f:
        f.write(entry)

def _remove_model_entry(model_id: str):
    """Remove the YAML block for a given id, leaving everything else intact."""
    with open(YAML_PATH, "r") as f:
        lines = f.readlines()

    out = []
    skip = False
    for line in lines:
        stripped = line.strip()
        # Detect start of the entry we want to remove
        if stripped == f"- id: {model_id}":
            # Remove the blank line we may have written before this entry
            if out and out[-1].strip() == "":
                out.pop()
            skip = True
            continue
        # Detect start of next entry — stop skipping
        if skip and stripped.startswith("- id:"):
            skip = False
        if not skip:
            out.append(line)

    with open(YAML_PATH, "w") as f:
        f.writelines(out)

def _prompt_choice(prompt: str, choices: list[str]) -> str:
    numbered = "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(choices))
    while True:
        print(f"\n{prompt}\n{numbered}")
        raw = input("  Choice: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print("  Invalid choice, please try again.")

def _prompt_text(prompt: str, validator=None) -> str:
    while True:
        val = input(f"\n{prompt}: ").strip()
        if not val:
            print("  Cannot be empty, please try again.")
            continue
        if validator and not validator(val):
            continue
        return val

def _ensure_huggingface_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("\nThe 'huggingface_hub' package is required for downloading models.")
        ans = input("Install it now with pip? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        else:
            sys.exit("Aborted: huggingface_hub not installed.")

def _download_gguf(repo_id: str, dest_dir: str) -> str:
    """
    Download all GGUF files from a HF repo into dest_dir.
    Returns the path of the first (or only) GGUF file found.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    os.makedirs(dest_dir, exist_ok=True)

    gguf_files = [f for f in list_repo_files(repo_id) if f.endswith(".gguf")]
    if not gguf_files:
        sys.exit(f"Error: no .gguf files found in repo '{repo_id}'.")

    if len(gguf_files) > 1:
        print(f"\nMultiple GGUF files found in '{repo_id}':")
        chosen = _prompt_choice("Which file do you want to download?", gguf_files)
        gguf_files = [chosen]

    filename = gguf_files[0]
    print(f"\nDownloading '{filename}' from '{repo_id}'...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
    )
    # hf_hub_download may nest into a cache subdir — copy to dest_dir root if needed
    final_path = os.path.join(dest_dir, os.path.basename(filename))
    if os.path.abspath(local_path) != os.path.abspath(final_path):
        shutil.copy2(local_path, final_path)
    return final_path

def _download_mlx(repo_id: str, dest_dir: str):
    """Snapshot-download an entire MLX model repo into dest_dir."""
    from huggingface_hub import snapshot_download

    print(f"\nDownloading MLX model '{repo_id}' into '{dest_dir}'...")
    snapshot_download(repo_id=repo_id, local_dir=dest_dir)

def cmd_add():
    existing = _parse_models_block(_load_yaml_text())
    existing_ids = {m.get("id", "") for m in existing}

    def validate_id(val):
        if val in existing_ids:
            print(f"  ID '{val}' already exists in models.yaml — choose a different one.")
            return False
        if " " in val:
            print("  ID must not contain spaces.")
            return False
        return True

    model_id = _prompt_text("Enter a unique model id (e.g. llama-3-small)", validator=validate_id)

    model_type = _prompt_choice("Model format?", ["llama (GGUF via llama.cpp)", "mlx (safetensors via mlx-lm)"])
    model_type = "llama" if model_type.startswith("llama") else "mlx"

    chat_format = _prompt_choice("Chat format / prompt template?", ["llama3", "qwen"])

    hf_repo = _prompt_text(
        "Paste the HuggingFace repo id (e.g. 'lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF')"
    )

    _ensure_huggingface_hub()

    if model_type == "llama":
        dir_name = hf_repo.split("/")[-1].lower()
        dest_dir = os.path.join(LLAMA_DIR, dir_name)
        gguf_path = _download_gguf(hf_repo, dest_dir)
        rel_path = "../models/llama/" + dir_name + "/" + os.path.basename(gguf_path)
    else:
        dir_name = hf_repo.split("/")[-1].lower()
        dest_dir = os.path.join(MLX_DIR, dir_name)
        _download_mlx(hf_repo, dest_dir)
        rel_path = "../models/mlx/" + dir_name

    spec = {
        "id":          model_id,
        "type":        model_type,
        "chat_format": chat_format,
        "path":        rel_path,
    }

    _append_model_entry(spec)

    print(f"\nModel '{model_id}' added successfully.")
    print(f"  Path: {rel_path}")
    print(f"  Entry written to models/models.yaml")

def cmd_remove():
    models = _parse_models_block(_load_yaml_text())
    if not models:
        sys.exit("No models registered in models.yaml.")

    print("\nRegistered models:")
    for m in models:
        print(f"  [{m.get('id')}]  type={m.get('type')}  path={m.get('path')}")

    target_id = _prompt_text("Enter the id of the model to remove")

    match = next((m for m in models if m.get("id") == target_id), None)
    if not match:
        sys.exit(f"Error: no model with id '{target_id}' found in models.yaml.")

    model_path = match.get("path", "")

    # Confirm
    print(f"\nAbout to remove:")
    print(f"  id:   {target_id}")
    print(f"  path: {model_path}")
    ans = input("\nAlso delete the model files from disk? [y/N]: ").strip().lower()
    delete_files = ans in ("y", "yes")

    _remove_model_entry(target_id)
    print(f"\nRemoved '{target_id}' from models/models.yaml.")

    if delete_files:
        # Resolve path relative to build/ (where the server runs), but here we
        # resolve relative to the project root's parent — same effect.
        # model_path starts with "../models/..." so strip the leading "../"
        if model_path.startswith("../"):
            abs_path = os.path.join(SCRIPT_DIR, model_path[len("../"):])
        else:
            abs_path = os.path.join(SCRIPT_DIR, model_path)

        abs_path = os.path.normpath(abs_path)

        if os.path.isfile(abs_path):
            os.remove(abs_path)
            print(f"Deleted file: {abs_path}")
            # Remove the parent directory if now empty
            parent = os.path.dirname(abs_path)
            if os.path.isdir(parent) and not os.listdir(parent):
                shutil.rmtree(parent)
                print(f"Removed empty directory: {parent}")
        elif os.path.isdir(abs_path):
            shutil.rmtree(abs_path)
            print(f"Deleted directory: {abs_path}")
        else:
            print(f"Warning: path not found on disk, skipping file deletion: {abs_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Manage truss-native models (add / remove).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 model_manager.py add\n"
            "  python3 model_manager.py remove\n"
        ),
    )
    parser.add_argument("action", choices=["add", "remove"], help="Action to perform")
    args = parser.parse_args()

    if args.action == "add":
        cmd_add()
    else:
        cmd_remove()

if __name__ == "__main__":
    main()
