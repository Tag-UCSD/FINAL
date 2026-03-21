import os
import time
import csv
import requests
from urllib.parse import unquote
import re
from difflib import SequenceMatcher  # Library for comparing text similarity

# ==========================================
# CONFIGURATION SECTION
# ==========================================

# Wikimedia Commons API Endpoint
API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"

# User-Agent Header (Required by Wikimedia policy).
# Please replace 'your_email@example.com' with your actual contact info.
HEADERS = {
    "User-Agent": "InteriorArchScraper/5.0 (your_email@example.com)" 
}

# List of target architectural elements to search for.
ARCH_KEYWORDS = [
    "Vaulted Ceiling", "Coffered Ceiling", "Exposed Beams", 
    "Archway", "Colonnade", "Atrium", "Mezzanine", 
    "Clerestory Windows", "Skylight", "Glass Curtain Wall", 
    "Concrete Wall", "Brick Interior", "Marble Floor", 
    "Spiral Staircase", "Grand Staircase", 
    "Recessed Lighting", "Chandelier", "Natural Light", 
    "Corridor", "Hallway", "Foyer", "Vestibule"
]

# "Blacklist" of terms. If any of these appear in the title, description, or categories,
# the image will be skipped. This filters out drawings, exteriors, and people.
EXCLUDED_TERMS = [
    # 1. Non-photographic material (drawings, plans, scans)
    "drawing", "sketch", "plan", "map", "diagram", "section", "elevation", 
    "plate", "scan", "book", "page", "text", "document", "poster", "icon", "logo",
    "construction", "demolition", "ruins", "renovation",

    # 2. Exterior or non-interior contexts
    "exterior", "facade", "outside", "external", 
    "street", "road", "city", "town", "village", "urban",
    "garden", "park", "forest", "tree", "landscape", "nature",
    "aerial", "bird's eye", "view from",
    "courtyard", "patio", "terrace",

    # 3. People and faces (Privacy & Relevance filter)
    "people", "person", "man", "woman", "men", "women", "child", "children",
    "portrait", "group", "crowd", "meeting", "conference", "ceremony",
    "standing", "sitting", "posing", "family", "wedding", "tourist",
    "politician", "actor", "author", "artist",

    # 4. Close-ups and low-context details (Texture/Macro)
    "close-up", "closeup", "detail", "details", "texture", "pattern",
    "material","fragment", "piece", "part of", "inscription", "plaque",
    "sign", "writing", "macro", "object", "artifact"
]

# Output settings
TARGET_FOLDER = "Dataset_Final_Dedup"
METADATA_FILE = "dataset_dedup_captions.csv"

# Scraping Constraints
TARGET_PER_CLASS = 30       # Goal: Collect 30 valid images per keyword
MIN_RESOLUTION = 800        # Minimum width or height in pixels
MAX_SEARCH_DEPTH = 500      # Check up to 500 search results per keyword
SIMILARITY_THRESHOLD = 0.8  # Duplicate threshold (0.8 = 80% similarity in title)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def clean_html(raw_html):
    """
    Removes HTML tags from a string.
    Wikimedia descriptions often contain <a> or <i> tags.
    """
    if not raw_html:
        return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    # Remove extra whitespace
    return " ".join(cleantext.split())

def is_valid_content(title, categories, description):
    """
    Checks if the image metadata contains any 'forbidden' terms.
    Returns (False, bad_term) if found, otherwise (True, "").
    """
    # Combine all text fields and convert to lowercase for checking
    full_text = (str(title) + " " + str(categories) + " " + str(description)).lower()
    
    for term in EXCLUDED_TERMS:
        if term in full_text:
            return False, term
    return True, ""

def get_image_data_batch(file_titles):
    """
    Fetches detailed metadata (URL, dimensions, description) for a list of files.
    Uses batching (chunking) to avoid URL length errors.
    """
    if not file_titles:
        return {}
    
    all_pages = {}
    chunk_size = 10  # Request 10 images at a time
    
    for i in range(0, len(file_titles), chunk_size):
        chunk = file_titles[i:i + chunk_size]
        titles_string = "|".join(chunk)
        
        params = {
            "action": "query",
            "format": "json",
            "titles": titles_string,
            "prop": "imageinfo",
            # We specifically ask for URL, Extended Metadata (desc/cats), and Size
            "iiprop": "url|extmetadata|size"
        }
        
        try:
            response = requests.get(API_ENDPOINT, params=params, headers=HEADERS)
            data = response.json()
            # Merge this batch's results into the main dictionary
            all_pages.update(data.get("query", {}).get("pages", {}))
        except Exception as e:
            print(f"  Metadata batch failed: {e}")
            
    return all_pages

def download_image(url, folder):
    """
    Downloads an image from a URL to the target folder.
    """
    try:
        # Decode URL-encoded filename 
        filename = unquote(os.path.basename(url))
        
        # Truncate filename if it's too long (OS filesystem limit safety)
        if len(filename) > 80:
            name, ext = os.path.splitext(filename)
            filename = name[:80] + ext
            
        local_path = os.path.join(folder, filename)
        
        # If file already exists, skip download
        if os.path.exists(local_path):
            return local_path

        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            return local_path
    except Exception:
        return None
    return None

def is_duplicate_title(new_title, existing_titles):
    """
    Check if the 'new_title' is too similar to any title we have already downloaded
    for this specific keyword. This helps avoid downloading burst shots (e.g., View01, View02).
    """
    # Helper: simplify title by lowercasing and removing numbers
    # Example: "Church_Interior_01.jpg" -> "church_interior_.jpg"
    def simplify(t):
        t = t.lower().replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        return re.sub(r'\d+', '', t) 

    simple_new = simplify(new_title)
    
    for existing in existing_titles:
        simple_existing = simplify(existing)
        
        # Calculate similarity ratio (0.0 to 1.0)
        similarity = SequenceMatcher(None, simple_new, simple_existing).ratio()
        
        # If titles are too similar (>80%) or identical after removing numbers, reject it
        if similarity > SIMILARITY_THRESHOLD or simple_new == simple_existing:
            return True
            
    return False

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================

def main():
    # Create the target folder if it doesn't exist
    os.makedirs(TARGET_FOLDER, exist_ok=True)
    
    # Open CSV file to record metadata
    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as csvfile:
        # 'full_caption' is our constructed training text
        fieldnames = ["filename", "keyword", "full_caption", "source_title", "source_desc", "categories", "image_url"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Loop through each architectural keyword
        for keyword in ARCH_KEYWORDS:
            print(f"\n=== Processing Keyword: {keyword} ===")
            print(f"Target: {TARGET_PER_CLASS} images | Dedup Threshold: {SIMILARITY_THRESHOLD}")
            
            saved_count = 0
            search_offset = 0
            
            # Append "interior" to the keyword to filter out generic/exterior results immediately
            search_term = f"{keyword} interior"
            
            # List to track titles downloaded for THIS keyword (for de-duplication)
            downloaded_titles_this_keyword = []
            
            # Pagination Loop: Keep searching until we meet the target or hit the depth limit
            while saved_count < TARGET_PER_CLASS and search_offset < MAX_SEARCH_DEPTH:
                
                # 1. Perform Search
                search_params = {
                    "action": "query", "format": "json", "list": "search",
                    "srsearch": search_term, 
                    "srnamespace": 6, # Namespace 6 = Files
                    "srlimit": 50,    # Fetch 50 candidates per page
                    "sroffset": search_offset 
                }
                
                try:
                    search_resp = requests.get(API_ENDPOINT, params=search_params, headers=HEADERS)
                    search_data = search_resp.json()
                    results = search_data.get("query", {}).get("search", [])
                except Exception as e:
                    print(f"  Search API Error: {e}")
                    break

                if not results:
                    print("  No more results available.")
                    break

                # Update offset for the next loop iteration
                if "continue" in search_data:
                    search_offset = search_data["continue"]["sroffset"]
                else:
                    # No more pages, force loop exit after this batch
                    search_offset = MAX_SEARCH_DEPTH + 1

                # Filter for image files only
                titles = [r["title"] for r in results if r["title"].lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # 2. Get Detailed Metadata
                pages = get_image_data_batch(titles)
                
                # 3. Process Each Image
                for page_data in pages.values():
                    # Stop immediately if we reached the target for this keyword
                    if saved_count >= TARGET_PER_CLASS: 
                        break
                    
                    if "imageinfo" not in page_data: 
                        continue
                    
                    info = page_data["imageinfo"][0]
                    title = page_data.get("title", "")
                    
                    # --- Filter A: Resolution ---
                    width = info.get("width", 0)
                    height = info.get("height", 0)
                    if width < MIN_RESOLUTION and height < MIN_RESOLUTION: 
                        # Skip images that are too small
                        continue

                    # --- Filter B: De-Duplication (Check title similarity) ---
                    if is_duplicate_title(title, downloaded_titles_this_keyword):
                        # print(f"  [DEDUP] Skipped similar image: {title[:30]}...")
                        continue

                    # --- Filter C: Content (Check Excluded Terms) ---
                    ext_meta = info.get("extmetadata", {})
                    
                    # Extract raw metadata text
                    raw_desc = ext_meta.get("ImageDescription", {}).get("value", "")
                    clean_desc = clean_html(raw_desc)
                    
                    raw_cats = ext_meta.get("Categories", {}).get("value", "")
                    clean_cats = clean_html(raw_cats)
                    
                    # Check against blacklist
                    is_valid, bad_term = is_valid_content(title, clean_cats, clean_desc)
                    if not is_valid: 
                        # Skip if it contains forbidden words
                        continue
                    
                    # --- 4. Construct Rich Caption ---
                    # We combine the prompt + title + description + categories
                    full_caption_parts = []
                    full_caption_parts.append(f"An interior architectural photo of {keyword}.")
                    
                    if len(title) > 5: 
                        full_caption_parts.append(f"Title: {title.replace('File:', '')}.")
                    if len(clean_desc) > 5: 
                        full_caption_parts.append(f"Description: {clean_desc}")
                    if len(clean_cats) > 5: 
                        full_caption_parts.append(f"Categories: {clean_cats}")
                    
                    full_caption = " ".join(full_caption_parts)
                    
                    # --- 5. Download ---
                    print(f"  [DOWN] ({saved_count+1}/{TARGET_PER_CLASS}) {title[:30]}...")
                    local_path = download_image(info.get("url"), TARGET_FOLDER)
                    
                    if local_path:
                        writer.writerow({
                            "filename": os.path.basename(local_path),
                            "keyword": keyword,
                            "full_caption": full_caption,
                            "source_title": title,
                            "source_desc": clean_desc,
                            "categories": clean_cats,
                            "image_url": info.get("url")
                        })
                        saved_count += 1
                        
                        # Add this title to the list so we don't download similar ones later
                        downloaded_titles_this_keyword.append(title)
                        
                        # Small pause to be polite to the server
                        time.sleep(0.1)
            
            print(f"  >>> Finished '{keyword}': Collected {saved_count} images.")

if __name__ == "__main__":
    main()
