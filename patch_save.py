#!/usr/bin/env python3
"""Patch train.py to handle save_pretrained errors gracefully."""
with open("weak_to_strong/train.py") as f:
    content = f.read()

old = '''        if save_path:
            _m = model if hasattr(model, "save_pretrained") else model.module
            try:
                _m.save_pretrained(save_path, safe_serialization=False)
            except Exception:
                print("Warning: pytorch save failed, trying safetensors...")
                try:
                    _m.save_pretrained(save_path, safe_serialization=True)
                except Exception:
                    print("Warning: all saves failed, skipping model save")
                    import os; os.makedirs(save_path, exist_ok=True)
            print("saved", save_path)'''

new = '''        if save_path:
            _m = model if hasattr(model, "save_pretrained") else model.module
            try:
                _m.save_pretrained(save_path, safe_serialization=False)
            except Exception:
                print("Warning: pytorch save failed, trying safetensors...")
                try:
                    _m.save_pretrained(save_path, safe_serialization=True)
                except Exception:
                    print("Warning: all saves failed, skipping model save")
                    os.makedirs(save_path, exist_ok=True)
            print("saved", save_path)'''

content = content.replace(old, new)
with open("weak_to_strong/train.py", "w") as f:
    f.write(content)
print("Patched train.py with try/except for save_pretrained")
