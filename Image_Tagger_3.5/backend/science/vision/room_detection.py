"""
Room Type Tagging Pipeline using Places365 + Coarse Mapping.

This module provides stable room type classification for interior images using:
1. Places365 pretrained classifier for fine-grained scene recognition
2. Coarse taxonomy mapping for reliable downstream reasoning
3. Optional object-based consistency checks to improve accuracy

Architecture:
- Primary: Places365 CNN classifier (ResNet50 pretrained on 365 scene categories)
- Stabilizer: Coarse room taxonomy mapping (13 categories)
- Enhancer: Object detection heuristics for confidence adjustment
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from backend.science.core import AnalysisFrame

# Lazy load models
PLACES_MODEL = None
PLACES_LABELS = None
PLACES_TRANSFORM = None

logger = logging.getLogger("v3.science.room_detection")

# ============================================================================
# COARSE ROOM TAXONOMY
# Maps Places365 fine labels to coarse categories for stable predictions
# ============================================================================

COARSE_CATEGORIES = [
    "bathroom",
    "bedroom", 
    "corridor",
    "dining",
    "kitchen",
    "living",
    "office",
    "lobby",
    "retail",
    "classroom",
    "industrial",
    "outdoor_adjacent",
    "other",
]

# Mapping from Places365 labels to coarse categories.
# Labels are stored WITHOUT _indoor/_outdoor/_interior/_exterior suffixes.
# The matching logic strips these suffixes before lookup, so "pub_indoor"
# matches "pub", "library_indoor" matches "library", etc.
PLACES365_TO_COARSE = {
    # Bathroom
    "bathroom": "bathroom",
    "shower": "bathroom",
    "sauna": "bathroom",
    "locker_room": "bathroom",
    "dressing_room": "bathroom",

    # Bedroom
    "bedroom": "bedroom",
    "hotel_room": "bedroom",
    "youth_hostel": "bedroom",
    "nursery": "bedroom",
    "childs_room": "bedroom",
    "dormitory": "bedroom",
    "dorm_room": "bedroom",
    "berth": "bedroom",

    # Corridor
    "corridor": "corridor",
    "hallway": "corridor",
    "staircase": "corridor",
    "elevator_lobby": "corridor",
    "elevator_door": "corridor",
    "elevator": "corridor",
    "entrance_hall": "corridor",
    "fire_escape": "corridor",
    "escalator": "corridor",
    "doorway": "corridor",
    "mezzanine": "corridor",

    # Dining
    "dining_room": "dining",
    "banquet_hall": "dining",
    "cafeteria": "dining",
    "food_court": "dining",
    "restaurant": "dining",
    "restaurant_kitchen": "dining",
    "restaurant_patio": "dining",
    "dining_hall": "dining",
    "fastfood_restaurant": "dining",
    "pizzeria": "dining",
    "sushi_bar": "dining",
    "coffee_shop": "dining",
    "bar": "dining",
    "pub": "dining",
    "beer_hall": "dining",
    "delicatessen": "dining",
    "ice_cream_parlor": "dining",
    "wet_bar": "dining",
    "wine_cellar": "dining",
    "discotheque": "dining",

    # Kitchen
    "kitchen": "kitchen",
    "kitchenette": "kitchen",
    "galley": "kitchen",
    "bakery": "kitchen",
    "butchers_shop": "kitchen",
    "pantry": "kitchen",

    # Living
    "living_room": "living",
    "home_theater": "living",
    "television_room": "living",
    "recreation_room": "living",
    "game_room": "living",
    "playroom": "living",
    "attic": "living",
    "basement": "living",
    "loft": "living",
    "parlor": "living",
    "waiting_room": "living",
    "closet": "living",
    "bow_window": "living",
    "alcove": "living",
    "ball_pit": "living",
    "mansion": "living",
    "manufactured_home": "living",
    "house": "living",
    "chalet": "living",
    "cottage": "living",
    "cabin": "living",
    "igloo": "living",

    # Office
    "office": "office",
    "home_office": "office",
    "office_building": "office",
    "office_cubicles": "office",
    "conference_room": "office",
    "conference_center": "office",
    "cubicle": "office",
    "computer_room": "office",
    "server_room": "office",
    "reception": "office",
    "clean_room": "office",
    "television_studio": "office",
    "legislative_chamber": "office",
    "embassy": "office",
    "courthouse": "office",
    "veterinarians_office": "office",
    "hospital": "office",
    "hospital_room": "office",
    "operating_room": "office",
    "nursing_home": "office",

    # Lobby
    "lobby": "lobby",
    "hotel_lobby": "lobby",
    "apartment_building_lobby": "lobby",
    "foyer": "lobby",
    "vestibule": "lobby",
    "atrium_public": "lobby",
    "ballroom": "lobby",
    "throne_room": "lobby",
    "palace": "lobby",
    "castle": "lobby",

    # Retail
    "shopping_mall": "retail",
    "shoe_shop": "retail",
    "clothing_store": "retail",
    "bookstore": "retail",
    "jewelry_shop": "retail",
    "gift_shop": "retail",
    "toyshop": "retail",
    "pharmacy": "retail",
    "drugstore": "retail",
    "supermarket": "retail",
    "grocery_store": "retail",
    "market": "retail",
    "bazaar": "retail",
    "florist_shop": "retail",
    "beauty_salon": "retail",
    "barbershop": "retail",
    "candy_store": "retail",
    "department_store": "retail",
    "fabric_store": "retail",
    "flea_market": "retail",
    "general_store": "retail",
    "hardware_store": "retail",
    "pet_shop": "retail",
    "shopfront": "retail",
    "auto_showroom": "retail",

    # Classroom / Cultural
    "classroom": "classroom",
    "kindergarden_classroom": "classroom",
    "lecture_room": "classroom",
    "auditorium": "classroom",
    "library": "classroom",
    "reading_room": "classroom",
    "chemistry_lab": "classroom",
    "biology_laboratory": "classroom",
    "physics_laboratory": "classroom",
    "art_studio": "classroom",
    "art_school": "classroom",
    "art_gallery": "classroom",
    "music_studio": "classroom",
    "gymnasium": "classroom",
    "martial_arts_gym": "classroom",
    "schoolhouse": "classroom",
    "museum": "classroom",
    "natural_history_museum": "classroom",
    "science_museum": "classroom",
    "archive": "classroom",
    "church": "classroom",
    "mosque": "classroom",
    "synagogue": "classroom",
    "temple_asia": "classroom",
    "catacomb": "classroom",
    "burial_chamber": "classroom",

    # Industrial
    "warehouse": "industrial",
    "factory": "industrial",
    "assembly_line": "industrial",
    "auto_factory": "industrial",
    "machine_shop": "industrial",
    "repair_shop": "industrial",
    "workshop": "industrial",
    "laboratory": "industrial",
    "laundromat": "industrial",
    "utility_room": "industrial",
    "boiler_room": "industrial",
    "engine_room": "industrial",
    "garage": "industrial",
    "parking_garage": "industrial",
    "loading_dock": "industrial",
    "industrial_area": "industrial",
    "storage_room": "industrial",
    "shed": "industrial",
    "stable": "industrial",
    "hangar": "industrial",
    "construction_site": "industrial",
    "oilrig": "industrial",
    "landfill": "industrial",
    "junkyard": "industrial",

    # Outdoor Adjacent
    "patio": "outdoor_adjacent",
    "veranda": "outdoor_adjacent",
    "balcony": "outdoor_adjacent",
    "porch": "outdoor_adjacent",
    "deck": "outdoor_adjacent",
    "terrace": "outdoor_adjacent",
    "courtyard": "outdoor_adjacent",
    "gazebo": "outdoor_adjacent",
    "greenhouse": "outdoor_adjacent",
    "sunroom": "outdoor_adjacent",
    "conservatory": "outdoor_adjacent",
    "solarium": "outdoor_adjacent",
    "pool": "outdoor_adjacent",
    "swimming_pool": "outdoor_adjacent",
    "swimming_hole": "outdoor_adjacent",
    "jacuzzi": "outdoor_adjacent",
    "spa": "outdoor_adjacent",
    "water_park": "outdoor_adjacent",
    "pavilion": "outdoor_adjacent",
    "roof_garden": "outdoor_adjacent",
    "botanical_garden": "outdoor_adjacent",
    "formal_garden": "outdoor_adjacent",
    "japanese_garden": "outdoor_adjacent",
    "zen_garden": "outdoor_adjacent",
    "topiary_garden": "outdoor_adjacent",
    "vegetable_garden": "outdoor_adjacent",
    "yard": "outdoor_adjacent",
    "lawn": "outdoor_adjacent",
    "driveway": "outdoor_adjacent",
    "inn": "outdoor_adjacent",
    "motel": "outdoor_adjacent",
    "hotel": "outdoor_adjacent",
    "beach_house": "outdoor_adjacent",
}

# Suffixes to strip from Places365 labels before coarse mapping lookup
_LABEL_SUFFIXES_TO_STRIP = [
    "_indoor", "_outdoor", "_interior", "_exterior",
]


class RoomDetectionAnalyzer:
    """
    Room Type Tagging using Places365 + Coarse Mapping.
    
    Provides stable room type predictions by:
    1. Running Places365 classifier for fine-grained scene recognition
    2. Mapping fine labels to coarse taxonomy (13 categories)
    3. Optionally adjusting confidence based on detected objects
    
    Attributes computed:
    - room.type_coarse: Primary coarse room category
    - room.type_coarse_confidence: Confidence for coarse category
    - room.type_fine: Top fine-grained Places365 label
    - room.type_fine_confidence: Confidence for fine label
    """
    
    @staticmethod
    def load_model():
        """
        Load the Places365 model (lazy loading).
        
        Uses ResNet50 pretrained on Places365 dataset.
        Requires: pip install torch torchvision
        """
        global PLACES_MODEL, PLACES_LABELS, PLACES_TRANSFORM
        
        if PLACES_MODEL is None:
            try:
                import torch
                import torch.nn as nn
                from torchvision import models, transforms
                
                logger.info("Loading Places365 model for room detection...")
                
                # Load ResNet50 architecture
                model = models.resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 365)
                
                # Try to load pretrained weights
                try:
                    # Download Places365 weights if not cached
                    import urllib.request
                    import os
                    
                    weights_path = os.path.join(
                        os.path.dirname(__file__), 
                        "weights", 
                        "resnet50_places365.pth.tar"
                    )
                    
                    if not os.path.exists(weights_path):
                        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                        url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
                        logger.info(f"Downloading Places365 weights from {url}...")
                        urllib.request.urlretrieve(url, weights_path)
                    
                    checkpoint = torch.load(weights_path, map_location='cpu')
                    state_dict = {k.replace('module.', ''): v 
                                  for k, v in checkpoint['state_dict'].items()}
                    model.load_state_dict(state_dict)
                    logger.info("Places365 weights loaded successfully")
                    
                except Exception as e:
                    logger.warning(f"Could not load Places365 weights: {e}")
                    logger.warning("Using random initialization - predictions will be unreliable")
                
                model.eval()
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    model = model.cuda()
                    logger.info("Places365 model loaded on GPU")
                else:
                    logger.info("Places365 model loaded on CPU")
                
                PLACES_MODEL = model
                
                # Load category labels
                PLACES_LABELS = RoomDetectionAnalyzer._load_places365_labels()
                
                # Define image transforms
                PLACES_TRANSFORM = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
                
            except ImportError as e:
                logger.error(f"Failed to load Places365: {e}. Install with: pip install torch torchvision")
                raise
    
    @staticmethod
    def _load_places365_labels() -> List[str]:
        """Load Places365 category labels."""
        # Places365 category names (subset - full list has 365)
        # These are the most common interior-related categories
        labels = [
            "airfield", "airplane_cabin", "airport_terminal", "alcove", "alley",
            "amphitheater", "amusement_arcade", "amusement_park", "apartment_building_outdoor",
            "aquarium", "aqueduct", "arcade", "arch", "archive", "arena_hockey",
            "arena_performance", "arena_rodeo", "army_base", "art_gallery", "art_school",
            "art_studio", "assembly_line", "athletic_field_outdoor", "atrium_public",
            "attic", "auditorium", "auto_factory", "auto_showroom", "badlands",
            "bakery", "balcony_exterior", "balcony_interior", "ball_pit", "ballroom",
            "bamboo_forest", "bank_vault", "banquet_hall", "bar", "barn", "barndoor",
            "baseball_field", "basement", "basketball_court_indoor", "bathroom",
            "bazaar_indoor", "bazaar_outdoor", "beach", "beach_house", "beauty_salon",
            "bedroom", "berth", "biology_laboratory", "boardwalk", "boat_deck",
            "boathouse", "bookstore", "booth_indoor", "botanical_garden", "bow_window_indoor",
            "bowling_alley", "boxing_ring", "bridge", "building_facade", "bullring",
            "burial_chamber", "bus_interior", "bus_station_indoor", "butchers_shop",
            "butte", "cabin_outdoor", "cafeteria", "campsite", "campus", "canal_natural",
            "canal_urban", "candy_store", "canyon", "car_interior", "carrousel",
            "casino_indoor", "castle", "catacomb", "cemetery", "chalet", "chemistry_lab",
            "childs_room", "church_indoor", "church_outdoor", "classroom", "clean_room",
            "cliff", "closet", "clothing_store", "coast", "cockpit", "coffee_shop",
            "computer_room", "conference_center", "conference_room", "construction_site",
            "corn_field", "corral", "corridor", "cottage", "courthouse", "courtyard",
            "creek", "crevasse", "crosswalk", "dam", "delicatessen", "department_store",
            "desert_road", "desert_sand", "desert_vegetation", "dining_hall", "dining_room",
            "discotheque", "doorway_outdoor", "dorm_room", "downtown", "dressing_room",
            "driveway", "drugstore", "elevator_door", "elevator_interior", "elevator_lobby",
            "embassy", "engine_room", "entrance_hall", "escalator_indoor", "excavation",
            "fabric_store", "farm", "fastfood_restaurant", "field_cultivated", "field_wild",
            "fire_escape", "fire_station", "fishpond", "flea_market_indoor", "florist_shop_indoor",
            "food_court", "football_field", "forest_broadleaf", "forest_needleleaf",
            "forest_path", "forest_road", "formal_garden", "fountain", "galley", "garage_indoor",
            "garage_outdoor", "gas_station", "gazebo_exterior", "general_store_indoor",
            "general_store_outdoor", "gift_shop", "glacier", "golf_course", "greenhouse_indoor",
            "greenhouse_outdoor", "grotto", "gymnasium_indoor", "hangar_indoor", "hangar_outdoor",
            "harbor", "hardware_store", "hayfield", "heliport", "highway", "home_office",
            "home_theater", "hospital", "hospital_room", "hot_spring", "hotel_outdoor",
            "hotel_room", "house", "hunting_lodge_outdoor", "ice_cream_parlor", "ice_floe",
            "ice_shelf", "ice_skating_rink_indoor", "ice_skating_rink_outdoor", "iceberg",
            "igloo", "industrial_area", "inn_outdoor", "islet", "jacuzzi_indoor",
            "jail_cell", "japanese_garden", "jewelry_shop", "junkyard", "kasbah",
            "kennel_outdoor", "kindergarden_classroom", "kitchen", "lagoon", "lake_natural",
            "landfill", "landing_deck", "laundromat", "lawn", "lecture_room", "legislative_chamber",
            "library_indoor", "library_outdoor", "lighthouse", "living_room", "loading_dock",
            "lobby", "lock_chamber", "locker_room", "mansion", "manufactured_home",
            "market_indoor", "market_outdoor", "marsh", "martial_arts_gym", "mausoleum",
            "medina", "mezzanine", "moat_water", "mosque_outdoor", "motel", "mountain",
            "mountain_path", "mountain_snowy", "movie_theater_indoor", "museum_indoor",
            "museum_outdoor", "music_studio", "natural_history_museum", "nursery",
            "nursing_home", "oast_house", "ocean", "office", "office_building",
            "office_cubicles", "oilrig", "operating_room", "orchard", "orchestra_pit",
            "pagoda", "palace", "pantry", "park", "parking_garage_indoor",
            "parking_garage_outdoor", "parking_lot", "pasture", "patio", "pavilion",
            "pet_shop", "pharmacy", "phone_booth", "physics_laboratory", "picnic_area",
            "pier", "pizzeria", "playground", "playroom", "plaza", "pond",
            "porch", "promenade", "pub_indoor", "racecourse", "raceway", "raft",
            "railroad_track", "rainforest", "reception", "recreation_room", "repair_shop",
            "residential_neighborhood", "restaurant", "restaurant_kitchen", "restaurant_patio",
            "rice_paddy", "river", "rock_arch", "roof_garden", "rope_bridge", "ruin",
            "runway", "sandbox", "sauna", "schoolhouse", "science_museum", "server_room",
            "shed", "shoe_shop", "shopfront", "shopping_mall_indoor", "shower",
            "ski_resort", "ski_slope", "sky", "skyscraper", "slum", "snowfield",
            "soccer_field", "stable", "stadium_baseball", "stadium_football", "stadium_soccer",
            "stage_indoor", "stage_outdoor", "staircase", "storage_room", "street",
            "subway_station_platform", "supermarket", "sushi_bar", "swamp", "swimming_hole",
            "swimming_pool_indoor", "swimming_pool_outdoor", "synagogue_outdoor", "television_room",
            "television_studio", "temple_asia", "throne_room", "ticket_booth", "topiary_garden",
            "tower", "toyshop", "track_outdoor", "train_interior", "train_station_platform",
            "tree_farm", "tree_house", "trench", "tundra", "underwater_ocean_deep",
            "utility_room", "valley", "vegetable_garden", "veterinarians_office",
            "viaduct", "village", "vineyard", "volcano", "volleyball_court_outdoor",
            "waiting_room", "water_park", "water_tower", "waterfall", "watering_hole",
            "wave", "wet_bar", "wheat_field", "wind_farm", "windmill", "wine_cellar",
            "wrestling_ring_indoor", "yard", "youth_hostel", "zen_garden"
        ]
        return labels
    
    @staticmethod
    def _classify_places365(
        image: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Classify an image using Places365.
        
        Args:
            image: RGB numpy array (H, W, 3)
            top_k: Number of top predictions to return
            
        Returns:
            List of (label, probability) tuples sorted by probability descending
        """
        import torch
        from PIL import Image as PILImage
        
        # Convert numpy array to PIL Image
        pil_image = PILImage.fromarray(image)
        
        # Apply transforms
        input_tensor = PLACES_TRANSFORM(pil_image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Move to GPU if model is on GPU
        if next(PLACES_MODEL.parameters()).is_cuda:
            input_batch = input_batch.cuda()
        
        # Get predictions
        with torch.no_grad():
            logits = PLACES_MODEL(input_batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], top_k)
        
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            if idx < len(PLACES_LABELS):
                label = PLACES_LABELS[idx]
                results.append((label, float(prob)))
        
        return results
    
    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize a Places365 label for coarse mapping lookup.

        Strips _indoor/_outdoor/_interior/_exterior suffixes so that e.g.
        "pub_indoor" matches the mapping key "pub".
        """
        clean = label.lower().replace(" ", "_").replace("-", "_")
        for suffix in _LABEL_SUFFIXES_TO_STRIP:
            if clean.endswith(suffix):
                clean = clean[: -len(suffix)]
                break
        return clean

    @staticmethod
    def _map_to_coarse(
        fine_predictions: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """
        Map fine-grained Places365 predictions to coarse taxonomy.

        Aggregates probabilities from fine labels that map to the same
        coarse category.  Labels are normalized (suffix-stripped) before
        lookup so that "library_indoor" matches "library", etc.

        Args:
            fine_predictions: List of (fine_label, probability) from Places365

        Returns:
            Dictionary of coarse_category -> aggregated_probability
        """
        coarse_probs = {cat: 0.0 for cat in COARSE_CATEGORIES}

        for label, prob in fine_predictions:
            normalized = RoomDetectionAnalyzer._normalize_label(label)

            # Try exact match first, then try without suffix
            coarse = PLACES365_TO_COARSE.get(normalized)

            if coarse is None:
                # Last resort: check if any mapping key is a substring
                # e.g. "hunting_lodge_outdoor" -> try "hunting_lodge" -> not found -> "other"
                coarse = "other"

            coarse_probs[coarse] += prob

        # Normalize probabilities
        total = sum(coarse_probs.values())
        if total > 0:
            coarse_probs = {k: v / total for k, v in coarse_probs.items()}

        return coarse_probs
    
    @staticmethod
    def _apply_object_consistency(
        coarse_probs: Dict[str, float],
        frame: AnalysisFrame
    ) -> Dict[str, float]:
        """
        Adjust coarse probabilities based on detected objects.
        
        Uses object detection results (if available) to improve confidence:
        - Bathroom: boost if toilet/sink detected, reduce if not
        - Kitchen: boost if oven/refrigerator detected
        - Office: boost if chair/laptop/monitor detected
        
        Args:
            coarse_probs: Current coarse category probabilities
            frame: AnalysisFrame with potential object detection metadata
            
        Returns:
            Adjusted coarse probabilities
        """
        # Check if object detection was run
        seg_data = frame.metadata.get("segmentation_masks", [])
        if not seg_data:
            return coarse_probs
        
        # Extract detected classes
        detected_classes = set()
        for class_name, _, _, _ in seg_data:
            detected_classes.add(class_name.lower())
        
        # Adjustment factors
        adjusted = coarse_probs.copy()
        
        # Bathroom consistency
        bathroom_objects = {"toilet", "sink"}
        if adjusted["bathroom"] > 0.1:
            if bathroom_objects & detected_classes:
                adjusted["bathroom"] *= 1.3  # Boost
            else:
                adjusted["bathroom"] *= 0.7  # Reduce
        
        # Kitchen consistency
        kitchen_objects = {"oven", "refrigerator", "microwave", "toaster"}
        if adjusted["kitchen"] > 0.1:
            if kitchen_objects & detected_classes:
                adjusted["kitchen"] *= 1.3
            else:
                adjusted["kitchen"] *= 0.8
        
        # Office consistency
        office_objects = {"laptop", "keyboard", "mouse", "monitor", "computer"}
        if adjusted["office"] > 0.1:
            if office_objects & detected_classes:
                adjusted["office"] *= 1.2
        
        # Living room consistency
        living_objects = {"couch", "tv", "remote"}
        if adjusted["living"] > 0.1:
            if living_objects & detected_classes:
                adjusted["living"] *= 1.2
        
        # Dining consistency
        dining_objects = {"dining table", "chair", "wine glass", "fork", "knife"}
        if adjusted["dining"] > 0.1:
            if dining_objects & detected_classes:
                adjusted["dining"] *= 1.2
        
        # Bedroom consistency
        bedroom_objects = {"bed"}
        if adjusted["bedroom"] > 0.1:
            if bedroom_objects & detected_classes:
                adjusted["bedroom"] *= 1.3
            else:
                adjusted["bedroom"] *= 0.8
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    @staticmethod
    def analyze(
        frame: AnalysisFrame,
        top_k: int = 5,
        apply_object_consistency: bool = True
    ) -> Dict[str, Any]:
        """
        Run room type classification on the image.
        
        Args:
            frame: AnalysisFrame containing the image to analyze
            top_k: Number of top fine-grained predictions to return
            apply_object_consistency: Whether to adjust based on detected objects
            
        Returns:
            Dictionary containing:
            - room_type_fine: List of top-k fine predictions [(label, prob), ...]
            - room_type_coarse: Dict of coarse_category -> probability
            - top_coarse: (label, probability) for best coarse prediction
            - top_fine: (label, probability) for best fine prediction
        """
        RoomDetectionAnalyzer.load_model()
        
        # Step 1: Run Places365 classifier — get ALL predictions in one pass
        # We need all 365 for accurate coarse mapping (using only top-k
        # leaves most probability mass in "other").
        all_predictions = RoomDetectionAnalyzer._classify_places365(
            frame.original_image,
            top_k=365
        )
        fine_predictions = all_predictions[:top_k]  # Top-k for display / tags
        
        # Step 2: Map ALL 365 predictions to coarse taxonomy
        coarse_probs = RoomDetectionAnalyzer._map_to_coarse(all_predictions)
        
        # Step 3: Optional object consistency check
        if apply_object_consistency:
            coarse_probs = RoomDetectionAnalyzer._apply_object_consistency(
                coarse_probs, frame
            )
        
        # Get top predictions
        top_fine = fine_predictions[0] if fine_predictions else ("unknown", 0.0)
        top_coarse = max(coarse_probs.items(), key=lambda x: x[1])
        
        # Store in frame attributes
        coarse_idx = COARSE_CATEGORIES.index(top_coarse[0]) if top_coarse[0] in COARSE_CATEGORIES else -1
        frame.add_attribute("room.type_coarse", coarse_idx)
        frame.add_attribute("room.type_coarse_confidence", float(top_coarse[1]))
        frame.add_attribute("room.type_fine_confidence", float(top_fine[1]))
        
        # Generate tags with confidence for image attachment
        # Format: "room_type:category (XX%)" for easy parsing and display
        coarse_conf_pct = int(top_coarse[1] * 100)
        fine_conf_pct = int(top_fine[1] * 100)
        
        tags = [
            f"room:{top_coarse[0]} ({coarse_conf_pct}%)",  # Primary tag with coarse type
            f"room_fine:{top_fine[0]} ({fine_conf_pct}%)",  # Fine-grained tag
        ]
        
        # Add secondary predictions if confidence > 15%
        for cat, prob in sorted(coarse_probs.items(), key=lambda x: x[1], reverse=True)[1:4]:
            if prob > 0.15:
                tags.append(f"room_alt:{cat} ({int(prob * 100)}%)")
        
        # Store detailed results in metadata
        result = {
            "room_type_fine": [(label, float(prob)) for label, prob in fine_predictions],
            "room_type_coarse": {k: float(v) for k, v in coarse_probs.items()},
            "top_coarse": {"label": top_coarse[0], "probability": float(top_coarse[1])},
            "top_fine": {"label": top_fine[0], "probability": float(top_fine[1])},
            "tags": tags,  # Tags ready to attach to image
        }
        
        frame.metadata["room_detection"] = result
        frame.metadata["room_tags"] = tags  # Easy access to just the tags
        
        logger.info(
            f"Room detection: {top_coarse[0]} ({top_coarse[1]:.1%}) "
            f"[fine: {top_fine[0]} ({top_fine[1]:.1%})]"
        )
        
        return result


    @staticmethod
    def get_tags(frame: AnalysisFrame) -> List[str]:
        """
        Get the room detection tags from a frame that has been analyzed.
        
        Args:
            frame: AnalysisFrame that has been processed by analyze()
            
        Returns:
            List of tag strings, e.g. ["room:kitchen (67%)", "room_fine:kitchen (45%)"]
        """
        return frame.metadata.get("room_tags", [])
    
    @staticmethod
    def update_image_tags(
        db_session,
        image_id: int,
        tags: List[str],
        replace_room_tags: bool = True
    ) -> bool:
        """
        Update an image's meta_data tags with room detection results.
        
        Args:
            db_session: SQLAlchemy database session
            image_id: ID of the image to update
            tags: List of room tags to add
            replace_room_tags: If True, removes existing room:* tags before adding new ones
            
        Returns:
            True if successful, False otherwise
        """
        from backend.models.assets import Image
        
        try:
            image = db_session.query(Image).filter(Image.id == image_id).first()
            if not image:
                logger.warning(f"Image {image_id} not found for tag update")
                return False
            
            # Get existing meta_data or create new
            meta_data = image.meta_data or {}
            existing_tags = meta_data.get("tags", [])
            
            if replace_room_tags:
                # Remove existing room tags
                existing_tags = [
                    t for t in existing_tags 
                    if not t.startswith("room:") 
                    and not t.startswith("room_fine:") 
                    and not t.startswith("room_alt:")
                ]
            
            # Add new room tags
            existing_tags.extend(tags)
            
            # Update meta_data
            meta_data["tags"] = existing_tags
            image.meta_data = meta_data
            
            db_session.commit()
            logger.info(f"Updated image {image_id} with room tags: {tags}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update image {image_id} tags: {e}")
            db_session.rollback()
            return False


def detect_room_type(
    image: np.ndarray,
    top_k: int = 5,
    apply_object_consistency: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to detect room type from a single image.
    
    Args:
        image: RGB numpy array (H, W, 3)
        top_k: Number of top predictions to return
        apply_object_consistency: Whether to use object detection for adjustment
        
    Returns:
        Dictionary with room type classifications including 'tags' key
    """
    frame = AnalysisFrame(image_id=-1, original_image=image)
    return RoomDetectionAnalyzer.analyze(
        frame, 
        top_k=top_k,
        apply_object_consistency=apply_object_consistency
    )
