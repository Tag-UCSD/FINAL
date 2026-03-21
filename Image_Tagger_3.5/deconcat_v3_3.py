import os
from pathlib import Path

def deconcat(filename="Image_Tagger_v3.3.0_Grand_Jury_Fixes.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_file = None
    buffer = []
    
    print("🚀 Applying v3.3.0 Grand Jury Fixes...")

    for line in lines:
        if line.startswith("----- FILE PATH:"):
            if current_file and buffer:
                write_file(current_file, buffer)
                buffer = []
            rel_path = line.split("----- FILE PATH:")[1].strip()
            current_file = Path(rel_path)
        
        elif line.startswith("----- CONTENT START -----"):
            continue
        elif line.startswith("----- CONTENT END -----"):
            continue
        else:
            if current_file:
                buffer.append(line)

    if current_file and buffer:
        write_file(current_file, buffer)

def write_file(path: Path, content: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(content)
    print(f"✅ Updated: {path}")

if __name__ == "__main__":
    deconcat()
