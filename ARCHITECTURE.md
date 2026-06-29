# Video Scene Processor - Complete Architecture & Flow

## Overview

The Video Scene Processor is an advanced AI-powered video generation pipeline that takes an original video and article content, then creates a completely new video adapted to a different language, culture, and offer while maintaining the visual style and atmosphere of the original.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT SOURCES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Google Sheets          Original Video         Article Content              │
│  (Configuration)        (GCS/URL)              (Free Text / GCS URL)        │
└──────────┬──────────────────┬─────────────────────────┬─────────────────────┘
           │                  │                         │
           ▼                  ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VIDEO SCENE PROCESSOR                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │   Scene     │   │   Gemini    │   │   Image     │   │   Video     │     │
│  │  Detection  │──▶│  Analysis   │──▶│ Generation  │──▶│ Generation  │     │
│  │(PySceneDetect)  │(Comprehensive)  │(Nano Banana)│   │  (Kling)    │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │   Audio     │   │    Voice    │   │   Music     │   │  Subtitles  │     │
│  │ Extraction  │──▶│ Generation  │──▶│ Generation  │──▶│  (ZapCap)   │     │
│  │  (FFmpeg)   │   │(ElevenLabs) │   │   (Suno)    │   │             │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                       │
│  │   Video     │   │  CTA Button │   │   Final     │                       │
│  │ Concatenation──▶│  Overlay    │──▶│  Upload     │                       │
│  │  (Rendi)    │   │  (Rendi)    │   │  (S3/GCS)   │                       │
│  └─────────────┘   └─────────────┘   └─────────────┘                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Final Video URL        Google Sheets Update        All Intermediate Assets │
│  (S3/GCS)               (All columns)               (Images, Audio, etc.)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Processing Flow

### Step 1: Configuration & Input Reading

**Source:** Google Sheets API

**Columns Read:**
| Column | Purpose |
|--------|---------|
| `Video Link` | Original video URL (GCS or external) |
| `Free text` / `Title 1stP` + `Rest of Content` | Article content for adaptation |
| `Language` | Target language for the new video |
| `Manual Instructions` | Custom instructions for AI |
| `CTA Button` | Enable/disable CTA overlay |
| `CTA Text` | Text to display on CTA button |
| `CTA Duration` | "Whole Video" or "At the End" |
| `Add Subtitles` | Enable/disable ZapCap subtitles |
| `Voice ID` | Custom ElevenLabs voice ID |
| `Animation Model` | "Kling" or "Runway" |
| `Article related to Video` | Yes/No - affects adaptation strategy |

---

### Step 2: Video Download & Scene Detection

```
Original Video (URL)
        │
        ▼
┌───────────────────┐
│  Download Video   │  ← FFmpeg / requests
│  to Temp Directory│
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  PySceneDetect    │  ← AdaptiveDetector
│  Scene Boundaries │     Threshold: 2.5
└────────┬──────────┘     Min Duration: 1s
         │
         ▼
┌───────────────────┐
│  Extract Frames   │  ← 5 frames/second
│  for Analysis     │     Saved as JPEG
└───────────────────┘
```

**Output:**
- List of scenes with start/end timestamps
- Frame images for each second of video
- Video duration and metadata

---

### Step 3: Gemini Comprehensive Video Analysis

**This is the core AI analysis step that understands the entire video.**

```
All Frames + Audio Transcript + Article Text
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│                    GEMINI 2.5 PRO                            │
│                                                              │
│  Analyzes:                                                   │
│  ├── Video type (testimonial, demo, showcase, etc.)         │
│  ├── Visual style (lighting, mood, camera work)             │
│  ├── Subject tracking (appearance changes per scene)        │
│  ├── Product detection (shape, colors, placement)           │
│  ├── Cultural adaptation requirements                        │
│  ├── Scene-by-scene breakdown                                │
│  └── Voice-over script (if original has VO)                 │
│                                                              │
│  Returns:                                                    │
│  ├── Scene prompts (image + motion)                         │
│  ├── Product description (detailed)                         │
│  ├── New VO script (adapted to article)                     │
│  ├── Style prefix for all images                            │
│  └── Audio analysis (has_vo, style, etc.)                   │
└──────────────────────────────────────────────────────────────┘
```

**Key Intelligence:**

1. **Article Related to Video = YES:**
   - Adapt video for new offer/language
   - Keep visual style similar
   - Change product branding if needed

2. **Article Related to Video = NO:**
   - Create completely new content from article
   - Keep ONLY the atmosphere and style
   - Don't use original product/offer

3. **Cultural Adaptation:**
   - Characters match target language/country
   - Example: Arabic original → US English = American-looking people

4. **Product Branding Rules:**
   - Product surface: COMPLETELY CLEAN (no logos/text)
   - Text overlays: Use ACTUAL promotional text from article
   - Example: "50% OFF" not "promotional text badge"

---

### Step 4: Image Generation (Nano Banana)

```
For each scene:
┌────────────────────────────────────────────────┐
│  Image Prompt from Gemini                      │
│  + Product Reference Image (if detected)       │
│  + Style Prefix                                │
│  + Cultural Adaptation Instructions            │
│  + Text Overlay (actual text from article)     │
└──────────────────┬─────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────┐
│            NANO BANANA (Kie.ai)                │
│                                                │
│  Input:                                        │
│  - prompt: Full scene description              │
│  - reference_image_url: Product frame (opt)   │
│  - reference_description: Product details     │
│                                                │
│  Output:                                       │
│  - Generated image URL                         │
└────────────────────────────────────────────────┘
```

**Text Overlay Logic:**
```
Article Content Analysis:
├── Contains discount/sale/% → "50% OFF" (localized)
├── Contains job/career → "APPLY NOW" (localized)
├── Contains health/wellness → "TRY NOW" (localized)
├── Contains learning/course → "LEARN MORE" (localized)
└── Other/Unclear → No text (clean image)
```

---

### Step 5: Video Generation (Kling / Runway)

```
Generated Image + Motion Prompt
              │
              ▼
┌──────────────────────────────────────┐
│         KLING V2.5 (via Kie.ai)      │
│                                      │
│  Parameters:                         │
│  - image_url: Generated image        │
│  - prompt: Motion description        │
│  - duration: 5 seconds               │
│  - mode: professional                │
│                                      │
│  Output:                             │
│  - Animated video clip (5s)          │
└──────────────────────────────────────┘
```

---

### Step 6: Audio Pipeline (Parallel Processing)

```
┌──────────────────────────────────────────────────────────────────────┐
│                     AUDIO PIPELINE                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Step 6a: Extract Audio from Original                                │
│  ┌─────────────────┐                                                 │
│  │ FFmpeg Extract  │ → original_audio.mp3                            │
│  └────────┬────────┘                                                 │
│           │                                                          │
│           ▼                                                          │
│  Step 6b: Detect VO Presence                                         │
│  ┌─────────────────┐                                                 │
│  │ Whisper         │ → has_vo: true/false                            │
│  │ Transcription   │ → transcript: "..."                             │
│  │ Gender Detection│ → gender: "m"/"f"                               │
│  └────────┬────────┘                                                 │
│           │                                                          │
│           ▼                                                          │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                   DECISION POINT                                │ │
│  ├────────────────────────────────────────────────────────────────┤ │
│  │                                                                 │ │
│  │  IF original has VO:                                           │ │
│  │  ┌─────────────────┐   ┌─────────────────┐                     │ │
│  │  │ Generate New VO │   │ Generate Music  │                     │ │
│  │  │ from Article    │   │ (Suno)          │                     │ │
│  │  │ (ElevenLabs TTS)│   │                 │                     │ │
│  │  └─────────────────┘   └─────────────────┘                     │ │
│  │                                                                 │ │
│  │  IF original has NO VO:                                        │ │
│  │  ┌─────────────────────────────────────────────┐              │ │
│  │  │ Generate Music ONLY (no VO)                 │              │ │
│  │  │ (Suno instrumental background)              │              │ │
│  │  └─────────────────────────────────────────────┘              │ │
│  │                                                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

**ElevenLabs TTS:**
- Uses voice_id from sheet (or default)
- Generates speech from Gemini VO script
- Returns word-level timestamps for subtitles

**Suno Music Generation:**
- Analyzes original audio for style
- Generates matching instrumental background
- Vocal cover mode if original has lyrics

---

### Step 7: Video Concatenation & Duration Matching

```
Scene Videos [1, 2, 3, 4, 5...]
              │
              ▼
┌──────────────────────────────────────┐
│         RENDI.DEV API                │
│                                      │
│  1. Trim each scene to match        │
│     original scene duration          │
│                                      │
│  2. Concatenate all scenes           │
│                                      │
│  3. Adjust duration to match VO:     │
│     - If video shorter: slow motion  │
│     - If video longer: trim end      │
│                                      │
│  Output: Combined video              │
└──────────────────────────────────────┘
```

---

### Step 8: Audio + Video Combination

```
Combined Video + Generated Audio (VO + Music)
                    │
                    ▼
┌──────────────────────────────────────┐
│         RENDI.DEV API                │
│                                      │
│  Step 8a: Add Voice-Over            │
│  - Primary audio track               │
│                                      │
│  Step 8b: Add Background Music       │
│  - Overlay at 20% volume             │
│                                      │
│  Output: Video with full audio       │
└──────────────────────────────────────┘
```

---

### Step 9: CTA Button Overlay

```
┌──────────────────────────────────────────────────────────────────┐
│                    CTA BUTTON PIPELINE                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Step 9a: Generate CTA Button Image                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Nano Banana                                                  │ │
│  │ Prompt: "CTA button with text '{cta_text}' on green bg"     │ │
│  │ Output: PNG image with green background                      │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Step 9b: Process CTA Button                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ - Remove green background (chroma key)                       │ │
│  │ - Add glow effect                                            │ │
│  │ - Upload to S3                                               │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Step 9c: Overlay on Video                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Rendi.dev FFmpeg:                                            │ │
│  │                                                              │ │
│  │ CTA Duration = "Whole Video":                                │ │
│  │   - start_time: 0.0                                          │ │
│  │   - end_time: video_duration                                 │ │
│  │   - position: center                                         │ │
│  │                                                              │ │
│  │ CTA Duration = "At the End":                                 │ │
│  │   - start_time: video_duration - 5.0                         │ │
│  │   - end_time: video_duration                                 │ │
│  │   - position: center                                         │ │
│  │                                                              │ │
│  │ Animation: Floating effect (sin wave)                        │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

### Step 10: Subtitles (ZapCap)

```
Video with Audio + Word Timestamps (from ElevenLabs)
                    │
                    ▼
┌──────────────────────────────────────┐
│           ZAPCAP API                 │
│                                      │
│  If TTS was used:                    │
│  - Send word timestamps              │
│  - Precise subtitle timing           │
│                                      │
│  If voice changer was used:          │
│  - Auto-transcription                │
│  - Language-aware                    │
│                                      │
│  Output: Video with burned-in subs   │
└──────────────────────────────────────┘
```

---

### Step 11: Final Upload & Sheet Update

```
Final Video
     │
     ▼
┌──────────────────────────────────────┐
│         AWS S3 / GCS                 │
│                                      │
│  Upload final video                  │
│  Generate public URL                 │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│       GOOGLE SHEETS UPDATE           │
│                                      │
│  Columns Updated:                    │
│  - Final Video (URL)                 │
│  - Subtitled Video (URL)             │
│  - New Voice (URL)                   │
│  - New Music (URL)                   │
│  - Scene 1-N Images (URLs)           │
│  - Scene 1-N Videos (URLs)           │
│  - Scene 1-N Prompts (text)          │
│  - Gender (detected)                 │
│  - Status (completed/error)          │
└──────────────────────────────────────┘
```

---

## Service Dependencies

### External APIs

| Service | Purpose | API |
|---------|---------|-----|
| **Gemini 2.5 Pro** | Video analysis, prompt generation | via Kie.ai |
| **Nano Banana** | Image generation | via Kie.ai |
| **Kling V2.5** | Video animation | via Kie.ai |
| **Runway** | Video animation (alternative) | via Kie.ai |
| **ElevenLabs** | TTS, voice cloning | Direct API |
| **Suno** | Music generation | via Kie.ai |
| **ZapCap** | Subtitle generation | Direct API |
| **Rendi.dev** | Video processing (FFmpeg) | Direct API |
| **OpenAI GPT-4o** | Fallback analysis, VO scripts | Direct API |

### Storage

| Service | Purpose |
|---------|---------|
| **AWS S3** | Final videos, audio files, images |
| **Google Cloud Storage** | Source videos, article files |
| **Local Temp** | Processing workspace |

### Data

| Service | Purpose |
|---------|---------|
| **Google Sheets** | Configuration, status tracking |

---

## Parallel Processing Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONCURRENT EXECUTION                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Row-Level Parallelism (max 2 workers):                             │
│  ┌──────────────┐  ┌──────────────┐                                 │
│  │   Row 2      │  │   Row 3      │                                 │
│  │  Processing  │  │  Processing  │                                 │
│  └──────────────┘  └──────────────┘                                 │
│                                                                      │
│  Scene-Level Parallelism (within each row):                         │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐            │
│  │Scene 1 │ │Scene 2 │ │Scene 3 │ │Scene 4 │ │Scene 5 │            │
│  │ Image  │ │ Image  │ │ Image  │ │ Image  │ │ Image  │            │
│  │ Video  │ │ Video  │ │ Video  │ │ Video  │ │ Video  │            │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘            │
│       ↓         ↓         ↓         ↓         ↓                    │
│       └─────────┴─────────┴────┬────┴─────────┘                    │
│                                │                                     │
│  Audio Pipeline (parallel with scenes):                             │
│  ┌─────────────────────────────────────────────────────┐           │
│  │ Extract Audio → Detect VO → Generate TTS → Music   │           │
│  └─────────────────────────────────────────────────────┘           │
│                                │                                     │
│                                ▼                                     │
│                    ┌─────────────────────┐                          │
│                    │ Wait for all tasks  │                          │
│                    │ Concatenate + Merge │                          │
│                    └─────────────────────┘                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Error Handling & Retry Logic

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ERROR HANDLING                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  API Call Retries:                                                  │
│  - Max retries: 3                                                   │
│  - Exponential backoff: 2s, 4s, 8s                                  │
│  - Timeout: 30-600 seconds (varies by operation)                    │
│                                                                      │
│  Fallback Strategies:                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Gemini fails → Use OpenAI GPT-4o for analysis               │   │
│  │ Kling fails → Use Runway for video generation               │   │
│  │ Local FFmpeg fails → Use Rendi.dev cloud FFmpeg             │   │
│  │ TTS fails → Skip VO (music only)                            │   │
│  │ Music generation fails → Use original audio                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Graceful Degradation:                                              │
│  - If CTA fails → Continue without CTA                              │
│  - If subtitles fail → Upload without subtitles                     │
│  - If one scene fails → Continue with other scenes                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Configuration (Config Class)

```python
class Config:
    # Google Sheets
    SPREADSHEET_ID: str
    WORKSHEET_NAME: str = "Input"
    
    # Column Names
    VIDEO_LINK_COLUMN: str = "Video Link"
    FREE_TEXT_COLUMN: str = "Free text"
    LANGUAGE_COLUMN: str = "Language"
    CTA_BUTTON_COLUMN: str = "CTA Button"
    CTA_TEXT_COLUMN: str = "CTA Text"
    CTA_DURATION_COLUMN: str = "CTA Duration"
    FINAL_VIDEO_COLUMN: str = "Final Video"
    # ... many more columns
    
    # API Keys
    OPENAI_API_KEY: str
    KIE_API_KEY: str
    ELEVENLABS_API_KEY: str
    RENDI_API_KEY: str
    ZAPCAP_API_KEY: str
    
    # Processing Settings
    SCENE_DETECTION_THRESHOLD: float = 2.5
    MIN_SCENE_DURATION: float = 1.0
    FRAMES_PER_SECOND: int = 5
    INFLUENCER_SCENE_DURATION: float = 5.0
    MAX_CONCURRENT_ROWS: int = 2
```

---

## File Structure

```
competitor-video-remaker/
├── video_scene_processor.py    # Main pipeline: config, services, orchestrator
├── scripts/run_final_stages.py # Manual harness for the final assembly stages
├── tests/                      # Unit tests for pure helper functions
├── .github/workflows/ci.yml    # CI: compile checks + pytest
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template (placeholders only)
├── README.md                   # Quick start guide
├── ARCHITECTURE.md             # This file
└── API_EXAMPLES.md             # API usage examples
```

> Credentials (`service_account.json`, `.env`) are provided locally by the user and are gitignored — never committed.

---

## Performance Metrics

| Operation | Typical Duration |
|-----------|------------------|
| Video download | 1-5 seconds |
| Scene detection | 2-5 seconds |
| Frame extraction | 5-15 seconds |
| Gemini analysis | 60-120 seconds |
| Image generation (per scene) | 10-20 seconds |
| Video generation (per scene) | 60-180 seconds |
| Audio pipeline | 30-60 seconds |
| Video concatenation | 10-30 seconds |
| CTA overlay | 10-20 seconds |
| Subtitles | 30-60 seconds |
| **Total per video** | **5-15 minutes** |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic pipeline |
| 2.0 | Added | Gemini comprehensive analysis |
| 3.0 | Added | Cultural adaptation |
| 4.0 | Added | CTA button overlay |
| 5.0 | Added | CTA Duration (whole video / at end) |
| 5.1 | Fixed | CTA position (center), No VO if original has no VO |
| 5.2 | Fixed | Actual text in prompts (not technical descriptions) |
