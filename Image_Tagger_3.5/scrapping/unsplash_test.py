import os
import time
import csv
import requests
import re
from urllib.parse import urlparse

# ==========================================
# 1. CONFIGURATION SECTION
# ==========================================

# Please enter your Unsplash Access Key here (registration on website)
UNSPLASH_ACCESS_KEY = "hPHytSSIUFWduspiP27CrHMCw0I-OHHoL93CBC9Ke0U"

# Target Keywords 
ARCH_KEYWORDS = [
    "Vaulted Ceiling", "Coffered Ceiling", "Exposed Beams",
    "Archway", "Colonnade", "Atrium", "Mezzanine",
    "Clerestory Windows", "Skylight", "Glass Curtain Wall",
    "Concrete Wall", "Brick Interior", "Marble Floor",
    "Spiral Staircase", "Grand Staircase",
    "Recessed Lighting", "Chandelier", "Natural Light",
    "Corridor", "Hallway", "Foyer", "Vestibule"
]

# Filter Blacklist
EXCLUDED_TERMS = [
    # 1. Exterior/Nature 
    "exterior", "facade", "outside", "street", "city", "urban",
    "aerial", "drone", "forest", "landscape", "mountain", "sea", "beach",
    "garden", "park", "courtyard", "terrace", "patio",

    # 2. People 
    "people", "person", "man", "woman", "portrait", "model", "girl", "boy",
    "group", "team", "crowd", "meeting", "wedding", "family", "couple", "fashion",

    # 3. Misc
    "text", "mockup", "sign", "poster", "screenshot", "rendering"
]

# Output Settings
TARGET_FOLDER = "Dataset_Unsplash_Interior"
METADATA_FILE = "dataset_unsplash_captions.csv"

TARGET_PER_CLASS = 30   # Target number per class
MIN_RESOLUTION = 1200   # Higher definition

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def is_valid_content(description, alt_description, tags):
    """
    Check if the description and tags contain any blacklisted terms.
    """
    # Combine all text fields
    full_text = (str(description) + " " + str(alt_description) + " " + " ".join(tags)).lower()

    for term in EXCLUDED_TERMS:
        if term in full_text:
            return False, term
    return True, ""

def download_image(url, folder, image_id):
    """
    Download image. Unsplash requires triggering a 'download_location' (simplified here, downloading URL directly).
    """
    try:
        local_filename = f"{image_id}.jpg"
        local_path = os.path.join(folder, local_filename)

        if os.path.exists(local_path):
            return local_path # Already exists

        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            return local_path
    except Exception as e:
        print(f"    Download Error: {e}")
        return None
    return None

def search_unsplash(keyword, page=1):
    """
    Call Unsplash Search API
    """
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": f"{keyword} interior", # Add 'interior' to improve precision
        "page": page,
        "per_page": 30, # Max 30 per page
        "orientation": "landscape", 
        "client_id": UNSPLASH_ACCESS_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"  API Error ({response.status_code}): {response.text}")
        return None
    return response.json()

# ==========================================
# 3. MAIN PROGRAM
# ==========================================

def main():
    if UNSPLASH_ACCESS_KEY == "YOUR_ACCESS_KEY_HERE":
        print("Error: Please enter your Unsplash Access Key in line 13")
        return

    os.makedirs(TARGET_FOLDER, exist_ok=True)

    # Record downloaded IDs to prevent duplicates
    downloaded_ids = set()

    # Check if CSV exists, if so, load downloaded IDs
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract ID from filename 
                fname = row.get("filename", "")
                img_id = os.path.splitext(fname)[0]
                downloaded_ids.add(img_id)

    # Open CSV for writing
    with open(METADATA_FILE, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filename", "keyword", "full_caption", "description", "alt_description", "tags", "image_url", "photographer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If file is empty, write header
        if not os.path.exists(METADATA_FILE) or os.stat(METADATA_FILE).st_size == 0:
            writer.writeheader()

        for keyword in ARCH_KEYWORDS:
            print(f"\n=== Unsplash: Processing '{keyword}' ===")

            saved_count = 0
            page = 1

            while saved_count < TARGET_PER_CLASS:
                print(f"  Searching Page {page}...")
                data = search_unsplash(keyword, page)

                if not data or not data.get('results'):
                    print("  No more results.")
                    break

                results = data['results']

                for img_data in results:
                    if saved_count >= TARGET_PER_CLASS:
                        break

                    img_id = img_data['id']

                    # --- Filter 1: Deduplication ---
                    if img_id in downloaded_ids:
                        continue

                    # --- Filter 2: Resolution ---
                    width = img_data['width']
                    height = img_data['height']
                    if width < MIN_RESOLUTION or height < MIN_RESOLUTION:
                        continue

                    # --- Filter 3: Content ---
                    desc = img_data.get('description') or ""
                    alt_desc = img_data.get('alt_description') or ""
                    # Extract Tag 
                    tags_list = [t['title'] for t in img_data.get('tags', [])]

                    is_valid, bad_term = is_valid_content(desc, alt_desc, tags_list)

                    if not is_valid:
                        continue

                    # --- 4. Construct Rich Caption ---
                    full_caption_parts = [f"An interior architectural photo of {keyword}."]

                    if alt_desc:
                        full_caption_parts.append(f"{alt_desc.capitalize()}.")

                    if desc:
                        full_caption_parts.append(f"Photographer description: {desc}.")

                    if tags_list:
                        top_tags = ", ".join(tags_list[:5]) # Take only top 5 tags
                        full_caption_parts.append(f"Tags: {top_tags}.")

                    full_caption = " ".join(full_caption_parts)
                    full_caption = " ".join(full_caption.split())

                    # --- 5. Download ---
                    # If raw needed, use img_data['urls']['raw']
                    download_url = img_data['urls']['regular']
                    photographer = img_data['user']['name']

                    print(f"    [DOWN] ({saved_count+1}/{TARGET_PER_CLASS}) ID:{img_id} by {photographer}...")
                    local_path = download_image(download_url, TARGET_FOLDER, img_id)

                    if local_path:
                        writer.writerow({
                            "filename": os.path.basename(local_path),
                            "keyword": keyword,
                            "full_caption": full_caption,
                            "description": desc,
                            "alt_description": alt_desc,
                            "tags": ", ".join(tags_list),
                            "image_url": download_url,
                            "photographer": photographer
                        })
                        saved_count += 1
                        downloaded_ids.add(img_id)

                # Next page
                page += 1
                # Important: Unsplash Demo API limit is 50 requests/hour.
                # Each search consumes 1 request. Sleeping briefly here, though mainly limited by hourly total.
                time.sleep(1)

            print(f"  >>> Finished '{keyword}': Collected {saved_count} images.")

if __name__ == "__main__":
    main()
