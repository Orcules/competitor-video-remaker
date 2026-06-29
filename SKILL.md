---
name: Product Video Creator - Implementation Guide
description: Step-by-step guide to add intelligent product detection to video generation pipelines. Ensures product accuracy when recreating videos with new contexts, articles, or languages.
---

# Product Video Creator - Implementation Guide

## Overview

This skill teaches you how to add **intelligent product detection** to your video generation pipeline. When you recreate a video with new content (different article, language, or context), the product will remain **pixel-perfect identical** while everything else adapts.

**Use Case Example:**
- Input: Coca-Cola ad in American beach setting
- Article: "New Israeli study on hydration benefits" (Hebrew)
- Output: Same Coca-Cola bottle (exact colors, branding), but in Israeli kitchen with Hebrew voiceover

## When to Use This Implementation

Add this to your pipeline when you need to:
- Recreate product videos in different languages/cultures
- Adapt advertising content while maintaining brand consistency
- Generate multiple video variations with the same product
- Ensure product accuracy when using AI image generation (Nano Banana, DALL-E, etc.)

## Core Concept

### The Problem
When you send a prompt like "Coca-Cola bottle in modern kitchen" to an AI image generator, you get **approximate** results. The bottle might be slightly different colors, wrong logo style, different proportions.

### The Solution
1. **Detect** the product in the original video
2. **Extract** the best reference frames showing the product clearly
3. **Upload** the reference frame to cloud storage
4. **Send** the reference image URL to Nano Banana along with the prompt
5. **Enhance** prompts to emphasize product accuracy

## Implementation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORIGINAL VIDEO                           │
│              (Input from Google Sheets)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Scene Detection      │ ← Your existing PySceneDetect logic
         │  (PySceneDetect)      │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Extract 5 frames     │ ← Use FFmpeg to extract frames
         │  from first scene     │    from scene timestamps
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ 🆕 PRODUCT DETECTION  │ ← NEW: Send frames to GPT-4o Vision
         │    (GPT-4o Vision)    │    Ask: "Is there a product?"
         └───────────┬───────────┘
                     │
                     ├─── No Product ──→ Continue with normal flow
                     │
                     └─── Product Detected
                          │
                          ▼
              ┌──────────────────────┐
              │ Get Product Details: │
              │ - Type, brand, colors│
              │ - Best frame index   │
              │ - Detailed description│
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Upload reference     │ ← Upload best frame to GCS/S3
              │ frame to cloud       │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Store reference URL  │ ← Save for later use
              │ and description      │
              └──────────┬───────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌────────────────┐          ┌────────────────────┐
│ Generate Scenes│          │ 🆕 ENHANCE PROMPTS │
│ (Your existing │          │ Add product details│
│  logic)        │          │ to each prompt     │
└────────┬───────┘          └─────────┬──────────┘
         │                            │
         └──────────┬─────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Call Nano Banana API │
         │ WITH:                │
         │ - Enhanced prompt    │
         │ - reference_image    │ ← NEW PARAMETERS
         │ - reference_desc     │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Generate Video       │ ← Continue with your existing
         │ (Runway)             │    Runway → Rendi → ElevenLabs
         └──────────┬───────────┘    pipeline
                    │
                    ▼
              [Final Output]
```

## Step-by-Step Implementation

### Step 1: Add Configuration

Add these settings to your configuration system:

```python
# Configuration parameters needed:
ENABLE_PRODUCT_DETECTION = True  # Feature flag
PRODUCT_MIN_CONFIDENCE = 0.7     # Minimum confidence score (0-1)
```

**Where to add:** In your Config class or environment variables file.

### Step 2: Create Product Detection Function

**Function Name:** `detect_product_in_frames(frame_paths, min_confidence)`

**Purpose:** Analyze frames to detect if a product is present.

**Inputs:**
- `frame_paths`: List of image file paths (5 frames from first scene)
- `min_confidence`: Minimum confidence threshold (default 0.7)

**Logic:**
1. Encode each frame to base64
2. Create image_url objects for GPT-4o Vision API
3. Send to OpenAI with this prompt structure:

```
System: You are an expert product analyst.

User: Analyze these frames for a PRODUCT (physical item with brand/packaging).

Describe in detail:
- Product type, brand, exact colors, materials, packaging
- For each frame: visibility score (0-1), suitability score (0-1)
- Which frame shows the product best (index)
- Overall confidence (0-1)

Return JSON:
{
    "has_product": true/false,
    "product_detected": "bottle" | "box" | "device" etc,
    "product_description": "Red glass bottle, 500ml, Coca-Cola brand...",
    "product_details": {
        "type": "bottle",
        "brand": "Coca-Cola",
        "colors": ["cherry red", "white"],
        "materials": ["glass"],
        "packaging": "red plastic cap, paper label"
    },
    "frame_analysis": [
        {"frame_index": 0, "visibility_score": 0.85, "suitability_score": 0.80},
        ...
    ],
    "best_frame_index": 2,
    "overall_confidence": 0.92
}
```

**Returns:** Dictionary with product information or `{"has_product": False}`

**Model to use:** `gpt-4o` (needs vision capability)

### Step 3: Create Prompt Enhancement Function

**Function Name:** `enhance_prompt_with_product(original_prompt, product_description, article_text)`

**Purpose:** Modify image generation prompt to maintain product while adapting scene.

**Inputs:**
- `original_prompt`: The image prompt from your scene analysis
- `product_description`: Detailed product description from detection
- `article_text`: New article content (optional)

**Logic:**
1. Send to GPT-4o with this structure:

```
System: You maintain product accuracy while adapting scenes.

User: 
Adapt this scene keeping the product IDENTICAL:

ORIGINAL SCENE: {original_prompt}

PRODUCT (MUST STAY EXACT): {product_description}

NEW CONTEXT: {article_text}

Rules:
1. Product appearance = 100% identical (colors, branding, packaging)
2. Change ONLY: environment, people, setting, lighting, atmosphere
3. Product stays clearly visible and centered

Generate enhanced prompt (200-300 words).
```

2. Wrap the response with emphasis:

```
[CRITICAL - PRODUCT MUST MATCH EXACTLY]
Product Reference: {product_description}

{enhanced_prompt_from_gpt4o}

[Verification: Product must be pixel-perfect match]
```

**Returns:** Enhanced prompt string

### Step 4: Create Reference Upload Function

**Function Name:** `upload_product_reference(frame_path)`

**Purpose:** Upload the best product frame to cloud storage.

**Inputs:**
- `frame_path`: Path to the best product frame

**Logic:**
1. Generate unique filename (e.g., `product_ref_{timestamp}.jpg`)
2. Use your existing GCS/S3 upload logic
3. Return the public URL

**Returns:** URL string or None if failed

### Step 5: Integrate into Main Pipeline

**Location:** In your `process_video` method (or equivalent)

**Integration Points:**

#### A. After Scene Detection:

```
Pseudo-code:

# Your existing code:
scenes = detect_scenes_with_pyscenedetect(video_path)

# NEW: Product detection block
product_info = None
product_reference_url = None

if ENABLE_PRODUCT_DETECTION and scenes exist:
    
    # Extract 5 frames from first scene
    first_scene_start, first_scene_end = scenes[0]
    detection_frames = extract_frames_evenly(
        video_path, 
        first_scene_start, 
        first_scene_end, 
        num_frames=5
    )
    
    # Detect product
    product_info = detect_product_in_frames(
        detection_frames,
        PRODUCT_MIN_CONFIDENCE
    )
    
    if product_info["has_product"]:
        log("✅ Product detected!")
        
        # Upload reference frame
        best_frame_index = product_info["best_frame_index"]
        best_frame = detection_frames[best_frame_index]
        product_reference_url = upload_product_reference(best_frame)

# Continue with your existing scene processing...
```

#### B. When Generating Images:

```
Pseudo-code:

for each scene:
    
    # Your existing prompt generation
    scene_prompt = analyze_scene_frames(...)
    
    # NEW: Enhance if product detected
    if product_info and product_info["has_product"]:
        enhanced_prompt = enhance_prompt_with_product(
            scene_prompt["first_prompt"],
            product_info["product_description"],
            article_text
        )
    else:
        enhanced_prompt = scene_prompt["first_prompt"]
    
    # NEW: Call Nano Banana with reference
    image_url = nano_banana_generate(
        prompt=enhanced_prompt,
        reference_image_url=product_reference_url,      # NEW
        reference_description=product_info["product_description"]  # NEW
    )
```

### Step 6: Update Nano Banana API Call

**Modification needed:** Add optional parameters to your Nano Banana function.

**Before:**
```python
def generate_image_nano_banana(prompt):
    payload = {
        "prompt": prompt,
        "aspect_ratio": "9:16"
    }
    # ... API call
```

**After:**
```python
def generate_image_nano_banana(
    prompt, 
    reference_image_url=None,      # NEW
    reference_description=None     # NEW
):
    payload = {
        "prompt": prompt,
        "aspect_ratio": "9:16"
    }
    
    # NEW: Add reference if provided
    if reference_image_url:
        payload["reference_image"] = reference_image_url
        payload["reference_description"] = reference_description or "Product reference"
        log("📸 Using product reference for accuracy")
    
    # ... API call
```

## Implementation Checklist

Use this to track your implementation:

- [ ] **Config**: Added ENABLE_PRODUCT_DETECTION and PRODUCT_MIN_CONFIDENCE
- [ ] **Function 1**: Created `detect_product_in_frames()` with GPT-4o Vision
- [ ] **Function 2**: Created `enhance_prompt_with_product()` 
- [ ] **Function 3**: Created `upload_product_reference()`
- [ ] **Integration A**: Added product detection after scene detection
- [ ] **Integration B**: Enhanced prompts before image generation
- [ ] **API Update**: Modified Nano Banana call to accept reference parameters
- [ ] **Testing**: Tested with product video
- [ ] **Testing**: Tested with non-product video
- [ ] **Testing**: Verified feature flag works (on/off)

## Testing Guide

### Test Case 1: Video with Product

**Input:**
- Video: Advertisement with clear product (e.g., Coca-Cola bottle)
- Article: Different topic/language

**Expected Behavior:**
1. Detection logs: "✅ Product detected: bottle"
2. Best frame uploaded to cloud
3. Prompts enhanced with product details
4. Nano Banana receives reference image URL
5. Generated images show identical product

**Success Criteria:**
- Product colors match original exactly
- Product branding/logos correct
- Product proportions maintained
- Only background/context changed

### Test Case 2: Video without Product

**Input:**
- Video: Landscape, people talking, generic scenes
- Article: Any

**Expected Behavior:**
1. Detection logs: "ℹ️ No product detected"
2. Pipeline continues normally
3. No reference image uploaded
4. Standard prompts used (not enhanced)

**Success Criteria:**
- No errors or crashes
- Video generated successfully
- Normal behavior maintained

### Test Case 3: Feature Disabled

**Input:**
- Set `ENABLE_PRODUCT_DETECTION = False`
- Any video

**Expected Behavior:**
- Product detection skipped entirely
- Original pipeline behavior
- No additional API calls

## Error Handling

### Handle These Scenarios:

1. **OpenAI API Failure:**
```python
try:
    product_info = detect_product_in_frames(frames)
except Exception as e:
    log(f"⚠️ Product detection failed: {e}")
    product_info = {"has_product": False}
    # Continue without product detection
```

2. **Upload Failure:**
```python
product_reference_url = upload_product_reference(frame)
if not product_reference_url:
    log("⚠️ Failed to upload reference, continuing without it")
    # Continue with enhanced prompts but no reference URL
```

3. **Low Confidence:**
```python
if product_info["overall_confidence"] < PRODUCT_MIN_CONFIDENCE:
    log(f"ℹ️ Product confidence too low: {confidence}")
    product_info = {"has_product": False}
```

## Performance Considerations

### API Calls Added:
- **1 OpenAI call** per video (product detection)
- **1 OpenAI call** per scene (prompt enhancement) - only if product detected
- **No additional Nano Banana calls** (same number, just with reference)

### Cost Impact:
- Product detection: ~$0.01-0.02 per video (GPT-4o Vision with 5 images)
- Prompt enhancement: ~$0.001 per scene (GPT-4o text)
- Total: ~$0.02-0.05 per video with product

### Performance:
- Detection adds: ~5-10 seconds per video
- Prompt enhancement adds: ~1-2 seconds per scene
- Parallel processing recommended for multiple scenes

## Best Practices

### 1. Extract Enough Frames for Detection
```
Minimum: 3 frames
Recommended: 5 frames
Maximum: 10 frames (diminishing returns)
```
Spread evenly across the first scene for good coverage.

### 2. Use Detailed Product Descriptions
The more detailed the description, the better Nano Banana can match it:
```
Bad:  "red bottle"
Good: "Cherry red glass bottle, 500ml, Coca-Cola white script logo, 
       red plastic screw cap, condensation droplets on surface"
```

### 3. Set Appropriate Confidence Threshold
```
0.5 = Very permissive (may detect non-products)
0.7 = Recommended (good balance)
0.9 = Very strict (may miss subtle products)
```

### 4. Cache Results for Re-runs
If processing the same video multiple times with different articles:
```python
# Check cache first
cached_product = get_from_cache(video_url)
if cached_product:
    product_info = cached_product
else:
    product_info = detect_product_in_frames(frames)
    save_to_cache(video_url, product_info)
```

### 5. Validate Generated Images
After generation, optionally verify the product matches:
```python
# Pseudo-code
generated_image_url = nano_banana_generate(...)

# Optional: Verify product accuracy
verification = verify_product_in_image(
    generated_image_url,
    product_info["product_description"]
)

if verification["accuracy_score"] < 0.8:
    log("⚠️ Generated product doesn't match, regenerating...")
    # Retry with stronger emphasis
```

## Troubleshooting

### Problem: Product Not Detected

**Possible Causes:**
- Product too small in frame
- Product obscured or blurry
- Confidence threshold too high

**Solutions:**
- Extract frames from multiple scenes (not just first)
- Lower PRODUCT_MIN_CONFIDENCE to 0.5
- Manually enhance frames (brightness, contrast) before detection
- Check frame quality (resolution, compression)

### Problem: Generated Product Doesn't Match

**Possible Causes:**
- Prompt not emphatic enough
- Reference image quality low
- Nano Banana ignoring reference

**Solutions:**
- Increase emphasis in prompt wrapper:
  ```
  [CRITICAL - ABSOLUTE PRIORITY - PRODUCT MUST BE IDENTICAL]
  [Any deviation from product description is unacceptable]
  ```
- Use multiple reference angles (upload 2-3 best frames)
- Increase reference weight in Nano Banana parameters (if available)
- Try regenerating with temperature=0 (less randomness)

### Problem: False Positives (Detecting Non-Products)

**Possible Causes:**
- Threshold too low
- Generic objects mistaken for products

**Solutions:**
- Increase PRODUCT_MIN_CONFIDENCE to 0.8+
- Improve detection prompt to be more specific:
  ```
  A product MUST have: distinct branding, packaging, or commercial identity
  NOT products: generic furniture, natural objects, people
  ```

## Advanced: Multi-Product Detection

If your videos have multiple products:

### Modification:
```python
# Instead of returning single product:
{
    "has_product": True,
    "products": [
        {
            "product_id": 1,
            "type": "bottle",
            "description": "...",
            "best_frame": 2
        },
        {
            "product_id": 2,
            "type": "box",
            "description": "...",
            "best_frame": 3
        }
    ]
}

# Then enhance prompts for each:
for product in products:
    enhanced_prompt = enhance_with_product(base_prompt, product)
```

## Integration with Existing Tools

### Google Sheets Integration
If your pipeline uses Google Sheets for inputs:

**Add Column:**
- "Product Detected" - Shows what was detected
- "Product Reference" - URL to reference frame
- "Product Confidence" - Detection confidence score

**Update Sheet:**
```python
if product_info["has_product"]:
    update_sheet_cell(row, "Product Detected", product_info["product_detected"])
    update_sheet_cell(row, "Product Reference", product_reference_url)
    update_sheet_cell(row, "Product Confidence", product_info["overall_confidence"])
```

### Rendi/ElevenLabs Integration
No changes needed - these run after image generation, so product detection doesn't affect them.

## Example Prompts

### Detection Prompt Template:
```
Analyze these video frames for a PRODUCT.

A PRODUCT is a physical commercial item with:
- Distinct branding or packaging
- Commercial/retail nature
- Recognizable visual identity

Examples: beverages, cosmetics, electronics, clothing items, food packages

For each frame, determine:
1. Is there a product? (yes/no)
2. Product type and category
3. Brand name (if visible)
4. Exact colors (specific shades)
5. Materials and textures
6. Packaging details
7. Distinctive features
8. Visibility and suitability scores (0-1)

Return detailed JSON with best frame selection.
```

### Enhancement Prompt Template:
```
Adapt this scene to new content while keeping the product IDENTICAL.

ORIGINAL SCENE:
{original_prompt}

PRODUCT THAT MUST REMAIN UNCHANGED:
{product_description}

NEW CONTEXT/ARTICLE:
{article_summary}

Generate a new prompt that:
1. Describes the EXACT product from the description
2. Places it in the new context/setting from the article
3. Changes environment, characters, mood, lighting
4. Keeps product prominent and clearly visible
5. Maintains product as focal point

The product's appearance (colors, branding, packaging, proportions) must be pixel-perfect identical.
```

## Summary

You've learned how to add intelligent product detection to your video pipeline:

✅ **What you've added:**
- Product detection with GPT-4o Vision
- Reference frame extraction and upload
- Prompt enhancement for product accuracy
- Reference image passing to Nano Banana

✅ **Benefits:**
- Products remain identical across video variations
- Works with any language/cultural adaptation
- Maintains brand consistency automatically
- Minimal performance impact

✅ **Next steps:**
- Implement the 3 core functions
- Integrate into your existing pipeline
- Test with product and non-product videos
- Monitor accuracy and adjust confidence threshold

**Estimated Implementation Time:** 2-4 hours

**Estimated Lines of Code Added:** ~150 lines

**Breaking Changes:** None (backwards compatible)

Good luck with your implementation! 🚀
