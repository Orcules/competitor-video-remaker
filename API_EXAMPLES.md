# API Call Examples

This file shows example API calls for the product detection system.

## OpenAI Vision API - Product Detection

### Request Structure

```python
# Prepare images
images = []
for frame_path in frame_paths:
    with open(frame_path, 'rb') as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
        images.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        })

# API call
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are an expert product analyst specializing in visual product detection."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze these frames for a PRODUCT (physical commercial item).

Provide detailed description including:
- Product type, brand, exact colors, materials
- Packaging details, distinctive features
- Frame-by-frame visibility scores (0-1)
- Best frame index for reference

Return JSON format:
{
    "has_product": boolean,
    "product_detected": string,
    "product_description": string,
    "product_details": {
        "type": string,
        "brand": string,
        "colors": [string],
        "materials": [string]
    },
    "best_frame_index": number,
    "overall_confidence": number
}"""
                }
            ] + images
        }
    ],
    max_tokens=2000,
    temperature=0.2,
    response_format={"type": "json_object"}
)

result = json.loads(response.choices[0].message.content)
```

### Example Response

```json
{
    "has_product": true,
    "product_detected": "bottle",
    "product_description": "Red glass Coca-Cola bottle, 500ml capacity, iconic contoured shape. White script logo on red label wrapping around center. Red plastic screw cap. Glass has slight condensation. Classic 1915 contour design, measuring approximately 8 inches tall.",
    "product_details": {
        "type": "beverage bottle",
        "brand": "Coca-Cola",
        "colors": ["cherry red", "white", "transparent glass"],
        "materials": ["glass", "plastic cap", "paper label"],
        "packaging": "red plastic screw cap, paper label with white logo",
        "distinctive_features": [
            "contoured glass shape",
            "white script Coca-Cola logo",
            "red color scheme",
            "classic design"
        ]
    },
    "frame_analysis": [
        {
            "frame_index": 0,
            "has_product": true,
            "visibility_score": 0.75,
            "suitability_score": 0.70,
            "product_position": "right side"
        },
        {
            "frame_index": 1,
            "has_product": true,
            "visibility_score": 0.92,
            "suitability_score": 0.88,
            "product_position": "center"
        },
        {
            "frame_index": 2,
            "has_product": true,
            "visibility_score": 0.85,
            "suitability_score": 0.82,
            "product_position": "center-left"
        }
    ],
    "best_frame_index": 1,
    "overall_confidence": 0.92
}
```

## OpenAI API - Prompt Enhancement

### Request Structure

```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You maintain product accuracy while adapting scenes to new contexts."
        },
        {
            "role": "user",
            "content": f"""Adapt this scene keeping the product IDENTICAL:

ORIGINAL SCENE:
{original_prompt}

PRODUCT (MUST STAY EXACT):
{product_description}

NEW CONTEXT:
{article_text}

Generate enhanced prompt that:
1. Describes exact product from description
2. Places it in new context
3. Changes environment, people, mood
4. Keeps product prominent and visible

Product must be pixel-perfect match."""
        }
    ],
    max_tokens=1000,
    temperature=0.3
)

enhanced = response.choices[0].message.content.strip()
```

### Example Input/Output

**Input:**
```
Original: "Refreshing beverage on sunny beach, person holding drink"
Product: "Red glass Coca-Cola bottle, 500ml, white script logo, contoured shape"
Context: "Israeli study shows hydration improves productivity by 35%"
```

**Output:**
```
Modern Israeli office setting during afternoon work break. A professional wearing business casual attire holds a red glass Coca-Cola bottle - the classic 500ml size with distinctive contoured shape and white script logo on red label, red plastic screw cap intact. The bottle shows slight condensation on the glass surface. 

The office has contemporary design with white walls, large windows showing Tel Aviv skyline, ergonomic desk setup with dual monitors. Natural lighting creates soft shadows. The person is taking a refreshing break, bottle positioned prominently in center frame, clearly visible against the clean office background. The Coca-Cola branding and iconic bottle design are unmistakable. 

The scene conveys a moment of rejuvenation during a productive workday, aligning with research on hydration and performance. Office environment is modern Israeli tech company style, with Hebrew text visible on wall poster in background.
```

## Nano Banana API - Image Generation with Reference

### Request Structure

```python
payload = {
    "prompt": enhanced_prompt,
    "aspect_ratio": "9:16",
    "reference_image": reference_image_url,  # NEW
    "reference_description": product_description,  # NEW
    # ... other parameters
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.post(
    "https://api.kie.ai/nano-banana/generate",
    json=payload,
    headers=headers
)
```

### Full Example

```python
# After product detection and enhancement:
if product_info["has_product"]:
    
    # Upload reference frame
    reference_url = upload_to_cloud(
        frame_paths[product_info["best_frame_index"]]
    )
    
    # Enhance prompt
    enhanced = enhance_prompt_with_product(
        original_prompt=scene_prompt,
        product_description=product_info["product_description"],
        article_text=article_content
    )
    
    # Generate image with reference
    image_url = generate_image_nano_banana(
        prompt=enhanced,
        reference_image_url=reference_url,
        reference_description=product_info["product_description"]
    )
else:
    # Standard generation without reference
    image_url = generate_image_nano_banana(
        prompt=scene_prompt
    )
```

## Error Handling Examples

### Product Detection Error

```python
try:
    product_info = detect_product_in_frames(frames, min_confidence=0.7)
except openai.APIError as e:
    logger.error(f"OpenAI API error: {e}")
    product_info = {"has_product": False, "error": str(e)}
except Exception as e:
    logger.error(f"Product detection failed: {e}")
    product_info = {"has_product": False, "error": str(e)}

# Always continue pipeline even if detection fails
if not product_info.get("has_product"):
    logger.info("Continuing without product detection")
```

### Reference Upload Error

```python
reference_url = upload_product_reference(best_frame_path)

if not reference_url:
    logger.warning("Reference upload failed, using enhanced prompt without reference")
    # Continue with enhanced prompt but no reference image
    image_url = generate_image_nano_banana(
        prompt=enhanced_prompt,
        reference_image_url=None,  # No reference
        reference_description=None
    )
```

### API Rate Limiting

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def detect_product_with_retry(frames):
    return detect_product_in_frames(frames)

# Use it:
try:
    product_info = detect_product_with_retry(detection_frames)
except Exception as e:
    logger.error(f"Failed after 3 retries: {e}")
    product_info = {"has_product": False}
```

## Caching Examples

### Cache Product Detection Results

```python
import hashlib
import json
from pathlib import Path

def get_video_hash(video_path):
    """Generate hash of video file for cache key."""
    hasher = hashlib.md5()
    with open(video_path, 'rb') as f:
        # Read first 1MB for hash (faster than full file)
        hasher.update(f.read(1024 * 1024))
    return hasher.hexdigest()

def cache_product_detection(video_path, product_info):
    """Save product detection to cache."""
    cache_dir = Path("cache/product_detection")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    video_hash = get_video_hash(video_path)
    cache_file = cache_dir / f"{video_hash}.json"
    
    with open(cache_file, 'w') as f:
        json.dump(product_info, f)
    
    logger.info(f"Cached product detection for {video_path}")

def get_cached_product_detection(video_path):
    """Retrieve cached product detection."""
    cache_dir = Path("cache/product_detection")
    video_hash = get_video_hash(video_path)
    cache_file = cache_dir / f"{video_hash}.json"
    
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            product_info = json.load(f)
        logger.info(f"Using cached product detection for {video_path}")
        return product_info
    
    return None

# Usage:
product_info = get_cached_product_detection(video_path)
if not product_info:
    product_info = detect_product_in_frames(frames)
    cache_product_detection(video_path, product_info)
```

## Logging Examples

### Structured Logging

```python
import logging

logger = logging.getLogger(__name__)

# Product detection
logger.info(f"🔍 [PRODUCT] Analyzing {len(frames)} frames for product detection")

if product_info["has_product"]:
    logger.info(f"✅ [PRODUCT] Detected: {product_info['product_detected']}")
    logger.info(f"   Brand: {product_info['product_details']['brand']}")
    logger.info(f"   Confidence: {product_info['overall_confidence']:.2f}")
    logger.info(f"   Best frame: {product_info['best_frame_index']}")
else:
    logger.info("ℹ️ [PRODUCT] No product detected, continuing with standard flow")

# Reference upload
logger.info(f"📤 [PRODUCT] Uploading reference frame to cloud storage")
if reference_url:
    logger.info(f"✅ [PRODUCT] Reference uploaded: {reference_url}")
else:
    logger.warning(f"⚠️ [PRODUCT] Reference upload failed")

# Prompt enhancement
logger.info(f"🎨 [PRODUCT] Enhancing prompt with product details")
logger.debug(f"   Original prompt: {original_prompt[:100]}...")
logger.debug(f"   Enhanced prompt: {enhanced_prompt[:100]}...")

# Image generation
logger.info(f"📸 [PRODUCT] Generating image with product reference")
```

## Testing Examples

### Unit Test - Product Detection

```python
def test_product_detection_with_product():
    """Test detection with frames containing a product."""
    
    # Arrange
    frame_paths = [
        "test_data/coca_cola_frame_1.jpg",
        "test_data/coca_cola_frame_2.jpg",
        "test_data/coca_cola_frame_3.jpg"
    ]
    
    # Act
    result = detect_product_in_frames(frame_paths, min_confidence=0.7)
    
    # Assert
    assert result["has_product"] == True
    assert result["overall_confidence"] >= 0.7
    assert "bottle" in result["product_detected"].lower()
    assert result["best_frame_index"] in [0, 1, 2]

def test_product_detection_no_product():
    """Test detection with frames containing no product."""
    
    # Arrange
    frame_paths = [
        "test_data/landscape_1.jpg",
        "test_data/landscape_2.jpg"
    ]
    
    # Act
    result = detect_product_in_frames(frame_paths, min_confidence=0.7)
    
    # Assert
    assert result["has_product"] == False
```

### Integration Test - Full Pipeline

```python
def test_full_pipeline_with_product():
    """Test complete pipeline with product detection."""
    
    # Arrange
    video_path = "test_data/coca_cola_ad.mp4"
    article = "New Israeli study on hydration benefits"
    
    # Act
    result = process_video_with_product_detection(
        video_path=video_path,
        article_text=article,
        language="he"
    )
    
    # Assert
    assert result["success"] == True
    assert result["product_detected"] == True
    assert result["product_reference_url"] is not None
    assert result["final_video_url"] is not None
    
    # Verify product in final frame
    final_frame = extract_frame_from_video(result["final_video_url"], 5.0)
    verify_result = verify_product_in_frame(
        final_frame,
        result["product_description"]
    )
    assert verify_result["accuracy_score"] > 0.8
```

## Complete Workflow Example

```python
def process_video_with_product_detection(
    video_path: str,
    article_text: str,
    language: str = "en"
) -> dict:
    """
    Complete workflow example showing all integration points.
    """
    
    result = {"success": False}
    
    try:
        # Step 1: Detect scenes (your existing code)
        logger.info("Step 1: Detecting scenes")
        scenes = detect_scenes_pyscenedetect(video_path)
        
        # Step 2: Extract frames from first scene
        logger.info("Step 2: Extracting frames for product detection")
        first_scene_start, first_scene_end = scenes[0]
        detection_frames = extract_frames_evenly(
            video_path,
            first_scene_start,
            first_scene_end,
            num_frames=5
        )
        
        # Step 3: Detect product
        logger.info("Step 3: Detecting product")
        product_info = detect_product_in_frames(
            detection_frames,
            min_confidence=0.7
        )
        
        # Step 4: Upload reference if product detected
        product_reference_url = None
        if product_info["has_product"]:
            logger.info("Step 4: Uploading product reference")
            best_frame = detection_frames[product_info["best_frame_index"]]
            product_reference_url = upload_product_reference(best_frame)
            
            result["product_detected"] = True
            result["product_reference_url"] = product_reference_url
            result["product_description"] = product_info["product_description"]
        
        # Step 5: Process each scene
        logger.info("Step 5: Processing scenes")
        scene_videos = []
        
        for scene_num, (start, end) in enumerate(scenes, 1):
            
            # Extract frames and analyze (your existing code)
            scene_frames = extract_scene_frames(video_path, start, end)
            scene_prompt = analyze_scene_frames(scene_frames)
            
            # Enhance prompt if product detected
            if product_info["has_product"]:
                enhanced_prompt = enhance_prompt_with_product(
                    scene_prompt["first_prompt"],
                    product_info["product_description"],
                    article_text
                )
            else:
                enhanced_prompt = scene_prompt["first_prompt"]
            
            # Generate image with optional reference
            image_url = generate_image_nano_banana(
                prompt=enhanced_prompt,
                reference_image_url=product_reference_url,
                reference_description=product_info.get("product_description")
            )
            
            # Generate video (your existing code)
            video_url = generate_video_runway(
                image_url=image_url,
                motion_prompt=scene_prompt["second_prompt"]
            )
            
            scene_videos.append(video_url)
        
        # Step 6: Combine and finalize (your existing code)
        logger.info("Step 6: Combining scenes")
        combined_video = combine_videos_rendi(scene_videos)
        
        # Add voiceover, music, etc. (your existing code)
        final_video = add_voiceover_and_music(
            combined_video,
            article_text,
            language
        )
        
        result["final_video_url"] = final_video
        result["success"] = True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        result["error"] = str(e)
    
    return result
```

This shows the complete integration of product detection into your existing pipeline.
