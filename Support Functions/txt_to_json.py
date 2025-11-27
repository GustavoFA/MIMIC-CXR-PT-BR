import os
import json
import random
from tqdm import tqdm

# === Base paths ===
dirName = os.path.dirname(os.path.abspath(__file__))
dirFiles = os.path.join(dirName, 'small_mimic_cxr', 'mimic_small', 'files')

all_data = {}

# === Limit number of files (0 = no limit) ===
limit_data = 30  # change to 10 or any number if you want to limit processing

# === Collect all .txt files ===
txt_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(dirFiles)
    for file in files
    if file.lower().endswith('.txt')
]

# Apply limit if set
if limit_data > 0:
    # txt_files = txt_files[:limit_data]
    txt_files = random.sample(txt_files, limit_data)

# === Process files with progress bar ===
for file_path in tqdm(txt_files, desc="Processing .txt files", unit="file"):
    try:
        # Try reading with UTF-8 encoding
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except UnicodeDecodeError:
            # Fallback to Latin-1 if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read().strip()

        file_name = os.path.basename(file_path)
        key_name = os.path.splitext(file_name)[0]

        # Save file content under the file name (without extension)
        all_data[key_name] = {
            'english': text
        }

    except Exception as e:
        print(f'[ERROR] Failed to read {file_path} - {e}')

# === Output JSON path ===
output_path = os.path.join(dirName, 'medical_texts.json')

# === Save as JSON ===
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

# === Final log ===
print(f"\nJSON file successfully created: {output_path}")
print(f"Total .txt files found: {len(txt_files)}")
print(f"Total texts saved: {len(all_data)}")