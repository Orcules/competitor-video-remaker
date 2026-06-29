#!/usr/bin/env python3
"""
Video Scene Processor - TVD X1 Pipeline
========================================
Processes videos from Google Sheets, extracts scenes, generates new content using AI,
and creates new videos with voice-over.

Pipeline Stages:
1. Read video links from Google Sheet
2. Extract 5 frames from every scene using FFmpeg
3. Analyze frames with OpenAI API to generate prompts
4. Generate images with Nano Banana (Kie.ai)
5. Generate videos with Runway (Kie.ai)
6. Combine scene videos with Rendi.dev
7. Change voice with ElevenLabs voice changer
8. Combine video + voice with Rendi.dev
9. Upload final video to AWS S3

Author: Generated for TVD_X1 Pipeline
"""

import os
import io
import re
import json
import time
import base64
import logging
import tempfile
import subprocess
import urllib.parse
import zipfile
import random
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests
import gspread
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from google.oauth2.service_account import Credentials
from google.cloud import storage
from openai import OpenAI
from dotenv import load_dotenv

# Google Gemini for native video analysis
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# PySceneDetect for better scene detection
try:
    from scenedetect import detect, ContentDetector, AdaptiveDetector
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    PYSCENEDETECT_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging with immediate flush for real-time output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Flushes immediately to console
    ]
)
logger = logging.getLogger(__name__)
# Ensure immediate flush
for handler in logging.root.handlers:
    handler.flush()


# =============================================================================
# GOOGLE SHEETS WRITE RATE LIMITER
# =============================================================================
from collections import deque as _deque


class _SheetsRateLimiter:
    """Global token-bucket limiter to keep Google Sheets writes under the
    ~60 requests/min/user quota.

    Paces calls WITHOUT holding any mutex during the wait, so a queued writer
    never freezes other worker threads. This lets us run many rows in parallel
    without triggering the cascading 429/backoff storms that occur when every
    cell write blocks on a single global lock.
    """

    def __init__(self, max_calls: int = 50, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls = _deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                # Drop timestamps outside the rolling window
                while self._calls and self._calls[0] <= now - self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                sleep_for = self.period - (now - self._calls[0])
            # Sleep OUTSIDE the lock so other threads can keep queueing
            time.sleep(max(0.05, min(sleep_for, self.period)))


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    """Configuration settings for the video processor."""
    
    # Google Sheets
    GOOGLE_SHEET_ID: str = "YOUR_SHEET_ID"
    GOOGLE_SHEET_TAB: str = "Input"
    SERVICE_ACCOUNT_FILE: str = os.environ.get("SERVICE_ACCOUNT_FILE", "service_account.json")
    
    # API Keys
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    KIE_API_KEY: str = os.environ.get("KIE_API", "")
    RENDI_API_KEY: str = os.environ.get("RENDI_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.environ.get("ELEVEN_LABS_API_KEY", "")
    GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")  # For Gemini video analysis
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: str = os.environ.get("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    AWS_BUCKET_NAME: str = os.environ.get("AWS_BUCKET_NAME", "your-gcs-bucket")
    AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
    AWS_FOLDER_PATH: str = "Comp/Final_Video/"
    
    # API Endpoints
    KIE_BASE_URL: str = "https://api.kie.ai"
    RENDI_BASE_URL: str = "https://api.rendi.dev"
    ELEVENLABS_BASE_URL: str = "https://api.elevenlabs.io/v1"
    
    # Processing settings
    MAX_SCENES: int = 10
    FRAMES_PER_SECOND: int = 3  # Frames to extract per second of scene duration
    # Number of sheet rows processed concurrently. Sheets writes are globally
    # rate-limited (see _SheetsRateLimiter), so this can safely be > 2.
    ROW_PARALLELISM: int = 6
    
    # PySceneDetect settings (more accurate than FFmpeg)
    # Threshold: 20-35 typical, higher = less sensitive, lower = more sensitive
    # For videos with ~8 scenes in ~16 seconds, try 25-30
    PYSCENEDETECT_THRESHOLD: float = 2.5
    PYSCENEDETECT_MIN_SCENE_DURATION: float = 1  # Minimum scene length in seconds
    PYSCENEDETECT_MAX_SCENE_DURATION: float = 10  # Maximum scene length in seconds (will be split if longer)
    PYSCENEDETECT_USE_ADAPTIVE: bool = True  # AdaptiveDetector adjusts to video content
    
    # ElevenLabs Voice ID (default)
    DEFAULT_VOICE_ID: str = "JBFqnCBsd6RMkjVDRZzb"
    DEFAULT_FEMALE_VOICE_ID: str = "EXAVITQu4vr4xnSDxMaL"  # Sarah voice for influencer mode
    
    # Influencer Mode settings
    DEFAULT_INFLUENCER_SCENES: int = 6  # Default number of scenes when Time column is empty
    INFLUENCER_SCENE_DURATION: float = 5.0  # Duration of each scene in seconds
    
    # Column mappings
    INPUT_VIDEO_COLUMN: str = "Input Videos"
    MANUAL_INSTRUCTIONS_COLUMN: str = "Manual instructions"
    
    # CTA Button columns
    ADD_CTA_BUTTON_COLUMN: str = "Add CTA button"
    CTA_TEXT_COLUMN: str = "CTA Text"
    CTA_DURATION_COLUMN: str = "CTA Duration"  # "Whole Video" or "At the End"
    
    # Opening Text columns
    ADD_OPENING_TEXT_COLUMN: str = "Opening Text?"
    OPENING_TEXT_COLUMN: str = "Opening Text"
    
    # Subtitles column
    ADD_SUBTITLES_COLUMN: str = "Add subtitles"
    
    # Article adaptation columns (optional)
    ARTICLE_COLUMN: str = "Article"
    VERTICAL_COLUMN: str = "Vertical"
    
    # Article-Video relationship column
    # "Yes" = Article is similar to video content, adapt video for new offer/language
    # "No" = Article is fundamentally different, keep video style but create new content
    ARTICLE_RELATED_TO_VIDEO_COLUMN: str = "Article related to Video"
    
    # Language column (for ZapCap subtitles language)
    LANGUAGE_COLUMN: str = "Language"
    
    # Manual override columns (optional)
    MANUAL_VO_TEXT_COLUMN: str = "Manual text for VO"
    MANUAL_MUSIC_LINK_COLUMN: str = "Manual music link"
    FREE_TEXT_COLUMN: str = "Free text"  # Overrides Title, 1stP, Rest of Content if provided
    
    # Influencer Mode columns (used when Input Videos is empty)
    IMAGE_1_COLUMN: str = "Image 1"
    IMAGE_2_COLUMN: str = "Image 2"
    IMAGE_3_COLUMN: str = "Image 3"
    IMAGE_4_COLUMN: str = "Image 4"
    TIME_COLUMN: str = "Time"  # Number of scenes to generate (default 6)
    VOICE_ID_COLUMN: str = "Voice id"  # Custom ElevenLabs voice ID (optional)
    
    # Article data columns (populated from GCS when Article contains a URL)
    TITLE_COLUMN: str = "Title"
    FIRST_PARAGRAPH_COLUMN: str = "1stP"
    REST_CONTENT_COLUMN: str = "Rest of Content"
    
    # GCS Configuration (for fetching article data from URLs)
    GCS_CREDENTIALS_FILE: str = os.environ.get("GCS_CREDENTIALS_FILE", "gcs_service_account.json")
    GCS_BUCKET_NAME: str = os.environ.get("GCS_BUCKET_NAME", "your-gcs-bucket")
    GCS_FOLDER_NAME: str = os.environ.get("GCS_FOLDER_NAME", "articles2025")
    
    # ZapCap settings
    ZAPCAP_API_KEY: str = os.environ.get("ZAPCAP_API_KEY", "")
    ZAPCAP_BASE_URL: str = "https://api.zapcap.ai"
    ZAPCAP_TEMPLATE_ID: str = os.environ.get("ZAPCAP_TEMPLATE_ID", "your-zapcap-template-id")
    
    # ==========================================================================
    # GEMINI VIDEO ANALYSIS SETTINGS
    # ==========================================================================
    ENABLE_GEMINI_VIDEO_ANALYSIS: bool = True  # Use Gemini for comprehensive video analysis
    GEMINI_MODEL: str = "gemini-1.5-flash"     # Model to use (flash is faster/cheaper, pro is more detailed)
    GEMINI_MAX_VIDEO_DURATION: int = 3600      # Max video duration in seconds (1 hour)
    
    # ==========================================================================
    # PRODUCT DETECTION SETTINGS
    # ==========================================================================
    ENABLE_PRODUCT_DETECTION: bool = True  # Feature flag to enable/disable product detection
    PRODUCT_MIN_CONFIDENCE: float = 0.7    # Minimum confidence score (0-1) to consider product detected
    PRODUCT_DETECTION_FRAMES: int = 60     # Number of frames to analyze for comprehensive video understanding
    PRODUCT_REFERENCE_FOLDER: str = "product_references"  # GCS folder for reference images
    
    # Output columns (Scene 1-8)
    SCENE_FIRST_PROMPT_PREFIX: str = "Scene {n} - First prompt"
    SCENE_SECOND_PROMPT_PREFIX: str = "Scene {n} - Second prompt"
    SCENE_NEW_IMAGE_PREFIX: str = "Scene {n} - new image"
    SCENE_NEW_VIDEO_PREFIX: str = "Scene {n} - new video"
    RENDI_SCENE_COLUMN: str = "RENDI Scene"
    NEW_VOICE_COLUMN: str = "New Voice"
    NEW_MUSIC_COLUMN: str = "New music"
    RENDI_SCENE_VOICE_COLUMN: str = "RENDI Scene & Voice"
    SUBTITLED_VIDEO_COLUMN: str = "Subtitled Video"
    FINAL_VIDEO_COLUMN: str = "Final Video"
    
    # Gender detection column (m for male, f for female)
    GENDER_COLUMN: str = "Gender"
    
    # Animation model column - "runway" (default) or "kling"
    ANIMATION_MODEL_COLUMN: str = "Animation model"
    
    # Product image input column (optional): if empty, generate product image and write URL here; if has link, use it as product reference
    PRODUCT_IMAGE_COLUMN: str = "Product image(optional)"
    # Product detection output columns (ENHANCED)
    PRODUCT_DETECTED_COLUMN: str = "Product Detected"
    PRODUCT_REFERENCE_COLUMN: str = "Product Reference"
    PRODUCT_CONFIDENCE_COLUMN: str = "Product Confidence"
    PRODUCT_PURPOSE_COLUMN: str = "Product Purpose"  # What the product does
    PRODUCT_USAGE_COLUMN: str = "Product Usage"      # How it's applied/used
    PRODUCT_CONTEXTS_COLUMN: str = "Usage Contexts"  # How it appears in different scenes
    
    # ==========================================================================
    # CULTURAL AND REGIONAL ADAPTATION SETTINGS
    # ==========================================================================
    
    # Map language codes to cultural regions
    REGION_MAPPING: dict = None  # Will be initialized in __post_init__
    
    # Cultural styles for each region (used in image/video generation prompts)
    CULTURAL_STYLES: dict = None  # Will be initialized in __post_init__
    
    # Hook styles for opening text by region
    HOOK_STYLES: dict = None  # Will be initialized in __post_init__
    
    def __post_init__(self):
        """Initialize complex dict fields after dataclass creation."""
        # Map language codes to cultural regions
        self.REGION_MAPPING = {
            # Latin America
            'es': 'latam', 'pt': 'latam', 'pt-BR': 'latam',
            # Western Europe
            'de': 'western_europe', 'fr': 'western_europe', 'it': 'western_europe',
            'nl': 'western_europe', 'da': 'western_europe', 'sv': 'western_europe',
            'no': 'western_europe', 'fi': 'western_europe',
            # Eastern Europe
            'hu': 'eastern_europe', 'pl': 'eastern_europe', 'cs': 'eastern_europe',
            'sk': 'eastern_europe', 'ro': 'eastern_europe', 'bg': 'eastern_europe',
            'uk': 'eastern_europe', 'ru': 'eastern_europe',
            # East Asia
            'zh': 'east_asia', 'ja': 'east_asia', 'ko': 'east_asia',
            'zh-TW': 'east_asia', 'zh-CN': 'east_asia',
            # Southeast Asia
            'th': 'southeast_asia', 'vi': 'southeast_asia', 'id': 'southeast_asia',
            'ms': 'southeast_asia', 'tl': 'southeast_asia',
            # METAP (Middle East, Turkey, Africa, Pakistan)
            'ar': 'metap', 'tr': 'metap', 'he': 'metap', 'fa': 'metap',
            'hi': 'metap', 'ur': 'metap', 'bn': 'metap',
            # North America / Australia / UK
            'en': 'namer', 'en-US': 'namer', 'en-GB': 'western_europe',
            'en-AU': 'namer',
        }
        
        # Cultural styles for each region (ethnicity, environment, style)
        self.CULTURAL_STYLES = {
            'latam': {
                'ethnicity': 'Latin American/Hispanic features, warm skin tones, dark hair',
                'environment': 'vibrant colors, colonial architecture, tropical or urban Latin settings',
                'style': 'warm, family-oriented, emotional, passionate',
                'clothing': 'casual modern Latin American fashion, bright colors',
                'names': 'names like Sofia, Diego, Isabella, Carlos, Maria, Juan'
            },
            'western_europe': {
                'ethnicity': 'diverse Western European features, mix of skin tones',
                'environment': 'modern European cities, clean architecture, historic buildings',
                'style': 'professional, sophisticated, understated elegance',
                'clothing': 'smart casual European fashion, neutral and earth tones',
                'names': 'names like Emma, Liam, Sophie, Felix, Anna, Max'
            },
            'eastern_europe': {
                'ethnicity': 'Eastern European/Slavic features, fair to medium skin tones',
                'environment': 'Eastern European cities, mix of historic and Soviet-era architecture',
                'style': 'practical, direct, resilient, no-nonsense',
                'clothing': 'practical European fashion, darker colors, layered outfits',
                'names': 'names like Katya, Ivan, Marta, Pavel, Olga, Dmitri'
            },
            'east_asia': {
                'ethnicity': 'East Asian features, Chinese/Japanese/Korean appearance',
                'environment': 'modern Asian cities, blend of traditional and ultra-modern',
                'style': 'refined, tech-savvy, minimalist, respectful',
                'clothing': 'modern Asian fashion, clean lines, often monochromatic',
                'names': 'names like Wei, Yuki, Min-ji, Kenji, Mei, Hiroshi'
            },
            'southeast_asia': {
                'ethnicity': 'Southeast Asian features, warm skin tones',
                'environment': 'tropical settings, bustling markets, modern Asian cities',
                'style': 'friendly, community-oriented, vibrant',
                'clothing': 'light fabrics, bright colors, tropical-appropriate fashion',
                'names': 'names like Anh, Putri, Somchai, Maria, Budi, Linh'
            },
            'metap': {
                'ethnicity': 'Middle Eastern, South Asian, or African features as appropriate',
                'environment': 'diverse - from modern Gulf cities to traditional markets',
                'style': 'respectful, family-values, hospitable',
                'clothing': 'modest modern fashion, appropriate for the specific culture',
                'names': 'names like Ahmed, Fatima, Priya, Rahul, Amina, Yusuf'
            },
            'namer': {
                'ethnicity': 'diverse North American features, multicultural mix',
                'environment': 'American suburbs, modern offices, diverse urban settings',
                'style': 'confident, aspirational, diverse, inclusive',
                'clothing': 'casual American fashion, athleisure, diverse styles',
                'names': 'names like Jessica, Michael, Ashley, Brandon, Emily, Tyler'
            }
        }
        
        # Hook styles for opening text by region
        self.HOOK_STYLES = {
            'latam': 'emotional appeal, family benefits, passionate language, urgency',
            'western_europe': 'factual benefits, professional tone, quality focus',
            'eastern_europe': 'practical advantages, value proposition, direct approach',
            'east_asia': 'social proof, technology benefits, quality assurance',
            'southeast_asia': 'community benefits, friendly tone, accessible language',
            'metap': 'family benefits, trust-building, respectful tone',
            'namer': 'aspirational messaging, personal success, opportunity focus'
        }


config = Config()


# =============================================================================
# VOICE ID VALIDATION UTILITY
# =============================================================================
def is_valid_voice_id(voice_id: str) -> bool:
    """Check if a voice_id is valid (not empty, not #N/A, etc.).
    
    Args:
        voice_id: The voice ID to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    if not voice_id:
        return False
    
    # Check for common invalid values from spreadsheets
    invalid_values = [
        '#n/a', '#na', 'n/a', 'na', '#ref!', '#error!', '#value!', 
        'null', 'none', 'undefined', '-', ''
    ]
    
    normalized = voice_id.strip().lower()
    return normalized not in invalid_values and len(normalized) > 3


def get_validated_voice_id(voice_id: str, default_voice_id: str = None) -> str:
    """Get a validated voice_id, falling back to default if invalid.
    
    Args:
        voice_id: The voice ID to validate.
        default_voice_id: Default to use if voice_id is invalid.
        
    Returns:
        Valid voice_id or default.
    """
    if is_valid_voice_id(voice_id):
        return voice_id
    return default_voice_id or config.DEFAULT_VOICE_ID


# =============================================================================
# LANGUAGE DETECTION UTILITY
# =============================================================================
def detect_language(text: str) -> str:
    """Detect language from text.
    
    Uses langdetect library if available, otherwise falls back to simple heuristics.
    
    Args:
        text: Text to analyze for language detection.
        
    Returns:
        ISO 639-1 language code (e.g., 'en', 'de', 'he', 'es', 'fr').
        Defaults to 'en' if detection fails.
    """
    if not text or len(text.strip()) < 10:
        logger.warning("⚠️ Text too short for language detection, defaulting to English")
        return "en"
    
    try:
        # Try using langdetect library
        from langdetect import detect as langdetect_detect
        detected = langdetect_detect(text)
        logger.info(f"🌍 Detected language: {detected}")
        return detected
    except ImportError:
        # Fallback: simple heuristic based on character sets
        logger.warning("⚠️ langdetect not installed, using heuristic detection")
        
        # Hebrew detection (Hebrew characters)
        if re.search(r'[\u0590-\u05FF]', text):
            return "he"
        # Arabic detection
        if re.search(r'[\u0600-\u06FF]', text):
            return "ar"
        # Chinese detection
        if re.search(r'[\u4E00-\u9FFF]', text):
            return "zh"
        # Japanese detection (Hiragana/Katakana)
        if re.search(r'[\u3040-\u30FF]', text):
            return "ja"
        # Korean detection
        if re.search(r'[\uAC00-\uD7AF]', text):
            return "ko"
        # Russian/Cyrillic detection
        if re.search(r'[\u0400-\u04FF]', text):
            return "ru"
        # German detection (common German words)
        german_words = ['und', 'die', 'der', 'das', 'ist', 'für', 'mit', 'von', 'nicht', 'eine']
        text_lower = text.lower()
        german_count = sum(1 for word in german_words if f' {word} ' in f' {text_lower} ')
        if german_count >= 3:
            return "de"
        # French detection
        french_words = ['le', 'la', 'les', 'de', 'et', 'est', 'une', 'que', 'pour', 'dans']
        french_count = sum(1 for word in french_words if f' {word} ' in f' {text_lower} ')
        if french_count >= 3:
            return "fr"
        # Spanish detection
        spanish_words = ['el', 'la', 'los', 'las', 'de', 'que', 'es', 'en', 'un', 'una']
        spanish_count = sum(1 for word in spanish_words if f' {word} ' in f' {text_lower} ')
        if spanish_count >= 3:
            return "es"
        
        # Default to English
        return "en"
    except Exception as e:
        logger.error(f"❌ Language detection failed: {e}, defaulting to English")
        return "en"


# =============================================================================
# GOOGLE SHEETS SERVICE
# =============================================================================
class GoogleSheetsService:
    """Service for managing Google Sheets operations."""
    
    def __init__(self, service_account_file: str):
        """Initialize Google Sheets service.
        
        Args:
            service_account_file: Path to the service account JSON file.
        """
        self.gc = None
        self._initialize_client(service_account_file)
    
    def _initialize_client(self, service_account_file: str) -> None:
        """Initialize the Google Sheets client.
        
        Args:
            service_account_file: Path to the service account JSON file.
        """
        try:
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = Credentials.from_service_account_file(
                service_account_file,
                scopes=scopes
            )
            
            self.gc = gspread.authorize(credentials)
            logger.info("✅ Google Sheets client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Sheets client: {e}")
            raise
    
    def get_worksheet_data(
        self, 
        sheet_id: str, 
        worksheet_name: str,
        max_retries: int = 3,
        base_delay: float = 2.0
    ) -> Tuple[List[str], List[List[str]]]:
        """Get all data from a worksheet with retry logic.
        
        Args:
            sheet_id: Google Sheet ID.
            worksheet_name: Name of the worksheet tab.
            max_retries: Maximum retry attempts for API errors.
            base_delay: Base delay for exponential backoff.
            
        Returns:
            Tuple of (headers, data_rows).
        """
        for attempt in range(max_retries + 1):
            try:
                spreadsheet = self.gc.open_by_key(sheet_id)
                # Try to get worksheet by name, fallback to first sheet
                try:
                    worksheet = spreadsheet.worksheet(worksheet_name)
                except Exception:
                    logger.warning(f"⚠️ Worksheet '{worksheet_name}' not found, using first sheet")
                    worksheet = spreadsheet.get_worksheet(0)
                    if worksheet:
                        logger.info(f"✅ Using sheet: {worksheet.title}")
                all_values = worksheet.get_all_values()
                
                if not all_values:
                    return [], []
                
                headers = all_values[0]
                data_rows = all_values[1:] if len(all_values) > 1 else []
                
                logger.info(f"✅ Retrieved {len(data_rows)} rows from {worksheet_name}")
                return headers, data_rows
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a retryable error
                is_retryable = any(err in error_str for err in [
                    'rate limit', 'quota', '429', '503', '500', 
                    'service unavailable', 'internal error',
                    'timeout', 'connection', 'temporarily unavailable',
                    'apiexception', 'exceeded', 'resource exhausted'
                ])
                
                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + (time.time() % 1)
                    logger.warning(
                        f"⚠️ Google Sheets read error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                        f"\n   Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Error getting worksheet data: {e}")
                    raise
        
        return [], []  # Should never reach here
    
    def get_column_index(self, headers: List[str], column_name: str) -> int:
        """Get column index by name.
        
        Args:
            headers: List of header names.
            column_name: Name of the column to find.
            
        Returns:
            Column index (0-based).
            
        Raises:
            ValueError: If column not found.
        """
        try:
            return headers.index(column_name)
        except ValueError:
            raise ValueError(f"Column '{column_name}' not found in headers: {headers[:10]}...")
    
    def get_row(
        self,
        sheet_id: str,
        worksheet_name: str,
        row_num: int,
        max_retries: int = 3,
        base_delay: float = 2.0
    ) -> Optional[List[str]]:
        """Get a single row from the worksheet.
        
        Args:
            sheet_id: Google Sheet ID.
            worksheet_name: Name of the worksheet tab.
            row_num: Row number (1-indexed, where 1 is header).
            max_retries: Maximum retry attempts for API errors.
            base_delay: Base delay for exponential backoff.
            
        Returns:
            List of cell values for the row, or None if failed.
        """
        for attempt in range(max_retries + 1):
            try:
                spreadsheet = self.gc.open_by_key(sheet_id)
                try:
                    worksheet = spreadsheet.worksheet(worksheet_name)
                except Exception:
                    worksheet = spreadsheet.get_worksheet(0)
                
                row_values = worksheet.row_values(row_num)
                return row_values
                
            except Exception as e:
                error_str = str(e).lower()
                
                is_retryable = any(err in error_str for err in [
                    'rate limit', 'quota', '429', '503', '500',
                    'service unavailable', 'internal error',
                    'timeout', 'connection', 'resource exhausted'
                ])
                
                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + (time.time() % 1)
                    logger.warning(f"⚠️ Get row error (attempt {attempt + 1}): {e}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Error getting row {row_num}: {e}")
                    return None
        
        return None
    
    def update_cell(
        self, 
        sheet_id: str, 
        worksheet_name: str, 
        row: int, 
        column_name: str, 
        value: str,
        headers: List[str],
        max_retries: int = 6,
        base_delay: float = 10.0
    ) -> None:
        """Update a single cell in the worksheet with retry logic.
        
        Args:
            sheet_id: Google Sheet ID.
            worksheet_name: Name of the worksheet tab.
            row: Row number (1-based).
            column_name: Name of the column.
            value: Value to set.
            headers: List of header names.
            max_retries: Maximum retry attempts for API errors.
            base_delay: Base delay for exponential backoff.
        """
        col_idx = self.get_column_index(headers, column_name) + 1  # gspread uses 1-based
        
        for attempt in range(max_retries + 1):
            try:
                spreadsheet = self.gc.open_by_key(sheet_id)
                worksheet = spreadsheet.worksheet(worksheet_name)
                worksheet.update_cell(row, col_idx, value)
                logger.info(f"✅ Updated cell ({row}, {column_name})")
                return
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a retryable error
                is_retryable = any(err in error_str for err in [
                    'rate limit', 'quota', '429', '503', '500', 
                    'service unavailable', 'internal error',
                    'timeout', 'connection', 'temporarily unavailable',
                    'apiexception', 'exceeded', 'resource exhausted'
                ])
                
                # Check if it's specifically a rate limit error (needs longer delay)
                is_rate_limit = any(err in error_str for err in [
                    'rate limit', 'quota', '429', 'exceeded', 'resource exhausted',
                    'read requests', 'write requests'
                ])
                
                if is_retryable and attempt < max_retries:
                    if is_rate_limit:
                        # For rate limits: wait 65+ seconds to let quota reset (60 req/min limit)
                        delay = 65 + (attempt * 10) + (time.time() % 5)
                        logger.warning(
                            f"⚠️ Google Sheets RATE LIMIT (attempt {attempt + 1}/{max_retries + 1}): "
                            f"Quota exceeded. Waiting {delay:.0f}s for quota to reset..."
                        )
                    else:
                        # Standard exponential backoff for other errors
                        delay = base_delay * (2 ** attempt) + (time.time() % 1)
                        logger.warning(
                            f"⚠️ Google Sheets API error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                            f"\n   Retrying in {delay:.1f}s..."
                        )
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Error updating cell ({row}, {column_name}): {e}")
                    raise
    
    def batch_update_cells(
        self,
        sheet_id: str,
        worksheet_name: str,
        updates: List[Dict[str, Any]],
        headers: List[str],
        max_retries: int = 6,
        base_delay: float = 10.0
    ) -> None:
        """Batch update multiple cells with retry logic.
        
        Args:
            sheet_id: Google Sheet ID.
            worksheet_name: Name of the worksheet tab.
            updates: List of dicts with 'row', 'column', 'value' keys.
            headers: List of header names.
            max_retries: Maximum retry attempts for API errors.
            base_delay: Base delay for exponential backoff.
        """
        if not updates:
            return
        
        for attempt in range(max_retries + 1):
            try:
                spreadsheet = self.gc.open_by_key(sheet_id)
                worksheet = spreadsheet.worksheet(worksheet_name)
                
                batch_data = []
                for update in updates:
                    row_num = update['row']
                    column = update['column']
                    value = update['value']
                    
                    col_idx = self.get_column_index(headers, column) + 1
                    col_letter = self._column_index_to_letter(col_idx)
                    cell_address = f"{col_letter}{row_num}"
                    
                    batch_data.append({
                        'range': cell_address,
                        'values': [[str(value)]]
                    })
                
                if batch_data:
                    worksheet.batch_update(batch_data)
                    logger.info(f"✅ Batch updated {len(batch_data)} cells")
                return
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a retryable error
                is_retryable = any(err in error_str for err in [
                    'rate limit', 'quota', '429', '503', '500', 
                    'service unavailable', 'internal error',
                    'timeout', 'connection', 'temporarily unavailable',
                    'apiexception', 'exceeded', 'resource exhausted'
                ])
                
                # Check if it's specifically a rate limit error (needs longer delay)
                is_rate_limit = any(err in error_str for err in [
                    'rate limit', 'quota', '429', 'exceeded', 'resource exhausted',
                    'read requests', 'write requests'
                ])
                
                if is_retryable and attempt < max_retries:
                    if is_rate_limit:
                        # For rate limits: wait 65+ seconds to let quota reset (60 req/min limit)
                        delay = 65 + (attempt * 10) + (time.time() % 5)
                        logger.warning(
                            f"⚠️ Google Sheets RATE LIMIT (attempt {attempt + 1}/{max_retries + 1}): "
                            f"Quota exceeded. Waiting {delay:.0f}s for quota to reset..."
                        )
                    else:
                        # Standard exponential backoff for other errors
                        delay = base_delay * (2 ** attempt) + (time.time() % 1)
                        logger.warning(
                            f"⚠️ Google Sheets batch update error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                            f"\n   Retrying in {delay:.1f}s..."
                        )
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Error in batch update: {e}")
                    raise
    
    def _column_index_to_letter(self, col_idx: int) -> str:
        """Convert column index to letter (1=A, 2=B, etc.)."""
        result = ""
        while col_idx > 0:
            col_idx -= 1
            result = chr(65 + col_idx % 26) + result
            col_idx //= 26
        return result


# =============================================================================
# FFMPEG VIDEO PROCESSOR (with Rendi.dev cloud fallback)
# =============================================================================
class FFmpegProcessor:
    """Service for video processing using FFmpeg (local or cloud via Rendi.dev)."""
    
    _ffmpeg_available: Optional[bool] = None
    _rendi_api_key: Optional[str] = None
    
    @classmethod
    def check_ffmpeg_installed(cls) -> bool:
        """Check if FFmpeg is installed locally.
        
        Returns:
            True if FFmpeg is available, False otherwise.
        """
        if cls._ffmpeg_available is not None:
            return cls._ffmpeg_available
        
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=10
            )
            cls._ffmpeg_available = result.returncode == 0
            if cls._ffmpeg_available:
                logger.info("✅ FFmpeg is installed locally")
            else:
                logger.warning("⚠️ FFmpeg not found, will use cloud processing via Rendi.dev")
            return cls._ffmpeg_available
        except Exception:
            cls._ffmpeg_available = False
            logger.warning("⚠️ FFmpeg not installed locally, will use cloud processing via Rendi.dev")
            return False
    
    @classmethod
    def set_rendi_api_key(cls, api_key: str) -> None:
        """Set Rendi API key for cloud fallback.
        
        Args:
            api_key: Rendi.dev API key.
        """
        cls._rendi_api_key = api_key
    
    @staticmethod
    def download_video(video_url: str, output_path: str) -> bool:
        """Download a video from URL.
        
        Args:
            video_url: URL of the video to download.
            output_path: Path to save the downloaded video.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            logger.info(f"📥 Downloading video from: {video_url[:50]}...")
            response = requests.get(video_url, stream=True, timeout=120)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"✅ Video downloaded to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading video: {e}")
            return False
    
    @staticmethod
    def detect_scenes(
        video_path: str, 
        threshold: float = 5.0,
        min_scene_duration: float = 1.0,
        use_adaptive: bool = True
    ) -> List[float]:
        """Detect scene changes in a video using PySceneDetect (preferred) or FFmpeg fallback.
        
        Args:
            video_path: Path to the video file.
            threshold: Scene change detection threshold.
                       For PySceneDetect ContentDetector: 20-35 is typical (higher = less sensitive)
                       For FFmpeg: 0.1-0.5 (lower = more sensitive)
            min_scene_duration: Minimum scene duration in seconds to prevent over-detection.
            use_adaptive: If True, uses AdaptiveDetector which adjusts to video content.
            
        Returns:
            List of timestamps (in seconds) where scenes start.
        """
        # Try PySceneDetect first (better results)
        if PYSCENEDETECT_AVAILABLE:
            try:
                return FFmpegProcessor._detect_scenes_pyscenedetect(
                    video_path, 
                    threshold=threshold,
                    min_scene_duration=min_scene_duration,
                    use_adaptive=use_adaptive
                )
            except Exception as e:
                logger.warning(f"⚠️ PySceneDetect failed: {e}, falling back to FFmpeg")
        
        # Fallback to FFmpeg
        return FFmpegProcessor._detect_scenes_ffmpeg(video_path, threshold=0.3)
    
    @staticmethod
    def _detect_scenes_pyscenedetect(
        video_path: str,
        threshold: float = 27.0,
        min_scene_duration: float = 1.0,
        use_adaptive: bool = True
    ) -> List[float]:
        """Detect scenes using PySceneDetect library (more accurate).
        
        Args:
            video_path: Path to the video file.
            threshold: Detection threshold (20-35 typical, higher = less sensitive).
            min_scene_duration: Minimum scene duration in seconds.
            use_adaptive: Use AdaptiveDetector (better for varying content).
            
        Returns:
            List of timestamps (in seconds) where scenes start.
        """
        logger.info(f"🎬 Detecting scenes with PySceneDetect: {video_path}")
        logger.info(f"   Threshold: {threshold}, Min duration: {min_scene_duration}s, Adaptive: {use_adaptive}")
        
        try:
            # Choose detector
            if use_adaptive:
                # AdaptiveDetector adjusts threshold based on local content
                # Good for videos with varying scene types
                detector = AdaptiveDetector(
                    adaptive_threshold=threshold,
                    min_scene_len=int(min_scene_duration * 30)  # Assuming ~30fps
                )
                logger.info("   Using AdaptiveDetector")
            else:
                # ContentDetector uses fixed threshold
                # Good when you know the video style
                detector = ContentDetector(
                    threshold=threshold,
                    min_scene_len=int(min_scene_duration * 30)
                )
                logger.info("   Using ContentDetector")
            
            # Detect scenes
            scene_list = detect(video_path, detector)
            
            # Get video duration for filtering short last scenes
            # Try OpenCV first (doesn't require FFmpeg), then FFmpeg, then fallback
            video_duration = 0
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0 and frame_count > 0:
                        video_duration = frame_count / fps
                        logger.info(f"   Video duration (OpenCV): {video_duration:.2f}s")
                    cap.release()
            except Exception as e:
                logger.debug(f"   Could not get duration via OpenCV: {e}")
            
            # Fallback to FFmpeg if OpenCV didn't work
            if video_duration <= 0:
                video_duration = FFmpegProcessor.get_video_duration(video_path)
            
            # Final fallback
            if video_duration <= 0:
                video_duration = 30.0
                logger.warning(f"   ⚠️ Could not determine video duration, using fallback: {video_duration}s")
            
            # Extract start timestamps
            timestamps = [0.0]  # Always include start
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                if start_time > 0 and start_time not in timestamps:
                    timestamps.append(start_time)
            
            # Sort timestamps
            timestamps.sort()
            
            # Get max scene duration from config
            max_scene_duration = config.PYSCENEDETECT_MAX_SCENE_DURATION
            
            # Step 1: Split scenes that are too long
            split_timestamps = []
            for i, ts in enumerate(timestamps):
                split_timestamps.append(ts)
                
                if i + 1 < len(timestamps):
                    next_ts = timestamps[i + 1]
                else:
                    next_ts = video_duration
                
                scene_duration = next_ts - ts
                
                # If scene is too long, split it into smaller segments
                if scene_duration > max_scene_duration:
                    num_splits = int(scene_duration / max_scene_duration)
                    split_duration = scene_duration / (num_splits + 1)
                    
                    for j in range(1, num_splits + 1):
                        new_ts = ts + (j * split_duration)
                        if new_ts < next_ts - 0.5:  # Don't add if too close to next scene
                            split_timestamps.append(new_ts)
                            logger.info(f"   ✂️ Splitting long scene: added timestamp at {new_ts:.2f}s")
            
            # Sort after splitting
            split_timestamps.sort()
            
            # Step 2: Filter out scenes that would be too short
            filtered_timestamps = []
            for i, ts in enumerate(split_timestamps):
                if i + 1 < len(split_timestamps):
                    scene_duration = split_timestamps[i + 1] - ts
                else:
                    scene_duration = video_duration - ts  # Last scene duration
                
                if scene_duration >= min_scene_duration or i == 0:  # Always keep first scene
                    filtered_timestamps.append(ts)
                else:
                    logger.info(f"   ⏭️ Filtering out scene at {ts:.2f}s (duration: {scene_duration:.2f}s < {min_scene_duration}s)")
            
            timestamps = filtered_timestamps[:config.MAX_SCENES]
            
            logger.info(f"✅ PySceneDetect found {len(timestamps)} scenes (after filtering)")
            for i, ts in enumerate(timestamps):
                logger.info(f"   Scene {i+1}: starts at {ts:.2f}s")
            
            return timestamps
            
        except Exception as e:
            logger.error(f"❌ PySceneDetect error: {e}")
            raise
    
    @staticmethod
    def _detect_scenes_ffmpeg(video_path: str, threshold: float = 0.3) -> List[float]:
        """Detect scene changes using FFmpeg (fallback method).
        
        Args:
            video_path: Path to the video file.
            threshold: Scene change threshold (0-1, lower = more sensitive).
            
        Returns:
            List of timestamps (in seconds) where scenes start.
        """
        # Check if FFmpeg is available
        if not FFmpegProcessor.check_ffmpeg_installed():
            logger.info("🌐 Using estimated scene intervals (no local FFmpeg)")
            return FFmpegProcessor._get_equal_intervals_simple(config.MAX_SCENES)
        
        try:
            logger.info(f"🎬 Detecting scenes with FFmpeg: {video_path}")
            
            # Use ffprobe to get scene changes
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-show_entries', 'frame=pkt_pts_time',
                '-select_streams', 'v',
                '-of', 'csv=p=0',
                '-f', 'lavfi',
                f"movie={video_path},select='gt(scene,{threshold})'"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.warning("⚠️ FFmpeg scene detection failed, using equal intervals")
                return FFmpegProcessor._get_equal_intervals(video_path)
            
            # Parse timestamps
            timestamps = [0.0]  # Always include start
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        timestamp = float(line)
                        timestamps.append(timestamp)
                    except ValueError:
                        continue
            
            logger.info(f"✅ FFmpeg detected {len(timestamps)} scenes")
            return timestamps[:config.MAX_SCENES]
            
        except Exception as e:
            logger.error(f"❌ FFmpeg scene detection error: {e}")
            return FFmpegProcessor._get_equal_intervals(video_path)
    
    @staticmethod
    def _get_equal_intervals_simple(num_scenes: int, assumed_duration: float = 30.0) -> List[float]:
        """Get equal time intervals without needing FFmpeg.
        
        Args:
            num_scenes: Number of scenes to create.
            assumed_duration: Assumed video duration if unknown.
            
        Returns:
            List of timestamps for equal intervals.
        """
        interval = assumed_duration / num_scenes
        return [i * interval for i in range(num_scenes)]
    
    @staticmethod
    def _get_equal_intervals(video_path: str) -> List[float]:
        """Get equal time intervals for a video.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            List of timestamps for equal intervals.
        """
        try:
            duration = FFmpegProcessor.get_video_duration(video_path)
            if duration <= 0:
                duration = 30.0  # Default 30 seconds
            
            num_scenes = min(config.MAX_SCENES, max(1, int(duration / 5)))
            interval = duration / num_scenes
            
            return [i * interval for i in range(num_scenes)]
            
        except Exception as e:
            logger.error(f"❌ Error getting equal intervals: {e}")
            return [0.0]
    
    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """Get the duration of a video in seconds.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            Duration in seconds.
        """
        if not FFmpegProcessor.check_ffmpeg_installed():
            return 30.0  # Default assumed duration
        
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
            
            return 30.0
            
        except Exception as e:
            logger.error(f"❌ Error getting video duration: {e}")
            return 30.0
    
    @staticmethod
    def extract_frames_entire_video(
        video_path: str,
        video_duration: float,
        output_dir: str,
        fps: int = 1
    ) -> List[Tuple[float, str]]:
        """Extract frames from the entire video at specified FPS.
        
        Args:
            video_path: Path to the video file.
            video_duration: Total video duration in seconds.
            output_dir: Directory to save extracted frames.
            fps: Frames per second to extract (default 1).
            
        Returns:
            List of (timestamp, frame_path) tuples.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate frame timestamps
        frame_interval = 1.0 / fps
        num_frames = max(1, int(video_duration * fps))
        
        logger.info(f"🎬 Extracting {num_frames} frames from entire video ({fps}/sec)...")
        
        frames_with_timestamps = []
        
        # Check if local FFmpeg is available
        if FFmpegProcessor.check_ffmpeg_installed():
            # Use local FFmpeg
            for i in range(num_frames):
                timestamp = (i * frame_interval) + (frame_interval / 2)  # Center of each slot
                if timestamp >= video_duration:
                    break
                
                output_path = os.path.join(output_dir, f"frame_{i:04d}_{timestamp:.1f}s.jpg")
                
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-ss', str(timestamp),
                    '-i', video_path,
                    '-vframes', '1',
                    '-q:v', '2',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    frames_with_timestamps.append((timestamp, output_path))
            
            logger.info(f"✅ Extracted {len(frames_with_timestamps)} frames (local FFmpeg)")
        else:
            # Use OpenCV (available via PySceneDetect)
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    logger.error("❌ Could not open video with OpenCV")
                    return []
                
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                
                for i in range(num_frames):
                    timestamp = (i * frame_interval) + (frame_interval / 2)
                    if timestamp >= video_duration:
                        break
                    
                    # Seek to frame
                    frame_number = int(timestamp * video_fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    
                    ret, frame = cap.read()
                    if ret:
                        output_path = os.path.join(output_dir, f"frame_{i:04d}_{timestamp:.1f}s.jpg")
                        cv2.imwrite(output_path, frame)
                        frames_with_timestamps.append((timestamp, output_path))
                
                cap.release()
                logger.info(f"✅ Extracted {len(frames_with_timestamps)} frames (OpenCV)")
                
            except Exception as e:
                logger.error(f"❌ Error extracting frames with OpenCV: {e}")
                return []
        
        return frames_with_timestamps
    
    @staticmethod
    def extract_frames_entire_video_cloud(
        video_url: str,
        video_duration: float,
        output_dir: str,
        rendi_api_key: str,
        fps: int = 1
    ) -> List[Tuple[float, str]]:
        """Extract frames from entire video using Rendi.dev cloud.
        
        Args:
            video_url: URL of the video.
            video_duration: Total video duration in seconds.
            output_dir: Local directory to save downloaded frames.
            rendi_api_key: Rendi.dev API key.
            fps: Frames per second to extract (default 1).
            
        Returns:
            List of (timestamp, frame_path) tuples.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        frame_interval = 1.0 / fps
        num_frames = max(1, int(video_duration * fps))
        
        logger.info(f"🌐 Extracting {num_frames} frames from video via Rendi.dev cloud...")
        
        frames_with_timestamps = []
        headers = {
            "X-API-KEY": rendi_api_key,
            "Content-Type": "application/json"
        }
        base_url = config.RENDI_BASE_URL
        
        for i in range(num_frames):
            timestamp = (i * frame_interval) + (frame_interval / 2)
            if timestamp >= video_duration:
                break
            
            # Create FFmpeg command to extract single frame
            ffmpeg_command = f"-ss {timestamp} -i {{{{in_1}}}} -vframes 1 -q:v 2 {{{{out_1}}}}"
            
            payload = {
                "ffmpeg_command": ffmpeg_command,
                "input_files": {"in_1": video_url},
                "output_files": {"out_1": f"frame_{i}.jpg"},
                "vcpu_count": 2,
                "max_command_run_seconds": 60
            }
            
            try:
                response = requests.post(
                    f"{base_url}/v1/run-ffmpeg-command",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    command_id = result.get("command_id")
                    
                    if command_id:
                        frame_url = FFmpegProcessor._wait_for_rendi_frame(
                            command_id, headers, base_url
                        )
                        
                        if frame_url:
                            local_path = os.path.join(output_dir, f"frame_{i:04d}_{timestamp:.1f}s.jpg")
                            if FFmpegProcessor._download_frame(frame_url, local_path):
                                frames_with_timestamps.append((timestamp, local_path))
                                if (i + 1) % 5 == 0:  # Log every 5 frames
                                    logger.info(f"   ✅ Extracted {i+1}/{num_frames} frames...")
                
                time.sleep(0.3)  # Small delay between requests
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to extract frame at {timestamp:.1f}s: {e}")
        
        logger.info(f"✅ Extracted {len(frames_with_timestamps)} frames via cloud")
        return frames_with_timestamps
    
    @staticmethod
    def extract_frames(
        video_path: str, 
        start_time: float, 
        end_time: float, 
        output_dir: str
    ) -> List[str]:
        """Extract frames from a video segment (1 frame per second).
        
        Args:
            video_path: Path to the video file.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            output_dir: Directory to save extracted frames.
            
        Returns:
            List of paths to extracted frame images.
        """
        if not FFmpegProcessor.check_ffmpeg_installed():
            # Use Rendi.dev cloud extraction
            return FFmpegProcessor._extract_frames_cloud(
                video_path, start_time, end_time, output_dir
            )
        
        try:
            duration = end_time - start_time
            if duration <= 0:
                duration = 5.0  # Default 5 seconds
            
            # Extract frames based on FRAMES_PER_SECOND config
            fps = config.FRAMES_PER_SECOND
            num_frames = max(1, int(duration * fps))
            frame_interval = 1.0 / fps  # Time between frames
            frame_paths = []
            
            logger.info(f"🎬 Extracting {num_frames} frames ({fps}/sec) from scene [{start_time:.1f}s - {end_time:.1f}s]")
            
            for i in range(num_frames):
                # Calculate timestamp: start + (frame_index * interval) + half_interval (center of each slot)
                timestamp = start_time + (i * frame_interval) + (frame_interval / 2)
                if timestamp >= end_time:
                    break
                    
                output_path = os.path.join(output_dir, f"frame_{int(start_time)}_{i}.jpg")
                
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-ss', str(timestamp),
                    '-i', video_path,
                    '-vframes', '1',
                    '-q:v', '2',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    frame_paths.append(output_path)
                    logger.debug(f"✅ Extracted frame at {timestamp:.2f}s")
            
            logger.info(f"✅ Extracted {len(frame_paths)} frames ({fps}/sec) from scene [{start_time:.1f}s - {end_time:.1f}s]")
            return frame_paths
            
        except Exception as e:
            logger.error(f"❌ Error extracting frames: {e}")
            return []
    
    @staticmethod
    def _extract_frames_cloud(
        video_path: str,
        start_time: float,
        end_time: float,
        output_dir: str
    ) -> List[str]:
        """Extract frames using Rendi.dev cloud FFmpeg (1 frame per second).
        
        This method uploads the video URL to Rendi and extracts frames in the cloud.
        """
        if not FFmpegProcessor._rendi_api_key:
            logger.error("❌ Rendi API key not set for cloud frame extraction")
            return []
        
        try:
            # For cloud extraction, we need the original video URL, not local path
            # This is a limitation - we'll return empty and let the pipeline handle it
            logger.warning("⚠️ Cloud frame extraction requires video URL - using URL-based extraction")
            return []
            
        except Exception as e:
            logger.error(f"❌ Error in cloud frame extraction: {e}")
            return []
    
    @staticmethod
    def extract_frames_from_url(
        video_url: str,
        start_time: float,
        end_time: float,
        output_dir: str,
        rendi_api_key: str
    ) -> List[str]:
        """Extract frames from video URL using Rendi.dev cloud FFmpeg (1 per second).
        
        Args:
            video_url: URL of the video.
            start_time: Start time of scene in seconds.
            end_time: End time of scene in seconds.
            output_dir: Local directory to save downloaded frames.
            rendi_api_key: Rendi.dev API key.
            
        Returns:
            List of paths to extracted frame images.
        """
        try:
            duration = end_time - start_time
            fps = config.FRAMES_PER_SECOND
            num_frames = max(1, int(duration * fps))  # Frames based on config FPS
            frame_interval = 1.0 / fps  # Time between frames
            
            # Generate timestamps based on configured FPS
            timestamps = []
            for i in range(num_frames):
                # Calculate timestamp: start + (frame_index * interval) + half_interval (center of slot)
                timestamp = start_time + (i * frame_interval) + (frame_interval / 2)
                if timestamp < end_time:
                    timestamps.append(timestamp)
            
            if not timestamps:
                timestamps = [start_time + duration / 2]  # At least middle frame
            
            logger.info(f"🌐 Extracting {len(timestamps)} frames ({fps}/sec) via Rendi.dev cloud...")
            
            frame_paths = []
            headers = {
                "X-API-KEY": rendi_api_key,
                "Content-Type": "application/json"
            }
            base_url = config.RENDI_BASE_URL
            
            for i, timestamp in enumerate(timestamps):
                # Create FFmpeg command to extract single frame
                ffmpeg_command = f"-ss {timestamp} -i {{{{in_1}}}} -vframes 1 -q:v 2 {{{{out_1}}}}"
                
                payload = {
                    "ffmpeg_command": ffmpeg_command,
                    "input_files": {"in_1": video_url},
                    "output_files": {"out_1": f"frame_{i}.jpg"},
                    "vcpu_count": 2,
                    "max_command_run_seconds": 60
                }
                
                response = requests.post(
                    f"{base_url}/v1/run-ffmpeg-command",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    command_id = result.get("command_id")
                    
                    if command_id:
                        # Poll for completion
                        frame_url = FFmpegProcessor._wait_for_rendi_frame(
                            command_id, headers, base_url
                        )
                        
                        if frame_url:
                            # Download frame locally
                            local_path = os.path.join(output_dir, f"frame_{i}.jpg")
                            if FFmpegProcessor._download_frame(frame_url, local_path):
                                frame_paths.append(local_path)
                                logger.info(f"✅ Extracted frame {i+1}/{len(timestamps)} at {timestamp:.1f}s")
                
                # Small delay between requests
                time.sleep(0.5)
            
            logger.info(f"✅ Extracted {len(frame_paths)} frames ({fps}/sec) via cloud")
            return frame_paths
            
        except Exception as e:
            logger.error(f"❌ Error extracting frames from URL: {e}")
            return []
    
    @staticmethod
    def _wait_for_rendi_frame(
        command_id: str, 
        headers: Dict, 
        base_url: str,
        timeout: int = 120
    ) -> Optional[str]:
        """Wait for Rendi frame extraction to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{base_url}/v1/commands/{command_id}",
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status", "").lower()
                    
                    if status in ["completed", "success"]:
                        output_files = result.get("output_files", {})
                        if "out_1" in output_files:
                            return output_files["out_1"].get("storage_url")
                    
                    elif status == "failed":
                        return None
                
                time.sleep(3)
                
            except Exception:
                time.sleep(3)
        
        return None
    
    @staticmethod
    def _download_frame(url: str, output_path: str) -> bool:
        """Download a frame image from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return os.path.exists(output_path)
            
        except Exception:
            return False
    
    @staticmethod
    def extract_audio(video_path: str, output_path: str) -> bool:
        """Extract audio track from video.
        
        Args:
            video_path: Path to the video file.
            output_path: Path to save the extracted audio.
            
        Returns:
            True if successful, False otherwise.
        """
        if not FFmpegProcessor.check_ffmpeg_installed():
            logger.warning("⚠️ FFmpeg not available for local audio extraction")
            return False
        
        try:
            cmd = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'libmp3lame',
                '-q:a', '2',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"✅ Extracted audio to: {output_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error extracting audio: {e}")
            return False
    
    @staticmethod
    def extract_audio_from_url(
        video_url: str,
        output_path: str,
        rendi_api_key: str
    ) -> Optional[str]:
        """Extract audio from video URL using Rendi.dev cloud FFmpeg.
        
        Args:
            video_url: URL of the video.
            output_path: Local path to save audio (used for naming).
            rendi_api_key: Rendi.dev API key.
            
        Returns:
            URL of the extracted audio, or None if failed.
        """
        try:
            logger.info("🌐 Extracting audio via Rendi.dev cloud...")
            
            headers = {
                "X-API-KEY": rendi_api_key,
                "Content-Type": "application/json"
            }
            base_url = config.RENDI_BASE_URL
            
            ffmpeg_command = "-i {{in_1}} -vn -acodec libmp3lame -q:a 2 {{out_1}}"
            
            payload = {
                "ffmpeg_command": ffmpeg_command,
                "input_files": {"in_1": video_url},
                "output_files": {"out_1": "extracted_audio.mp3"},
                "vcpu_count": 2,
                "max_command_run_seconds": 300
            }
            
            response = requests.post(
                f"{base_url}/v1/run-ffmpeg-command",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                command_id = result.get("command_id")
                
                if command_id:
                    # Poll for completion
                    audio_url = FFmpegProcessor._wait_for_rendi_audio(
                        command_id, headers, base_url
                    )
                    return audio_url
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting audio from URL: {e}")
            return None
    
    @staticmethod
    def _wait_for_rendi_audio(
        command_id: str,
        headers: Dict,
        base_url: str,
        timeout: int = 300
    ) -> Optional[str]:
        """Wait for Rendi audio extraction to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{base_url}/v1/commands/{command_id}",
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status", "").lower()
                    
                    if status in ["completed", "success"]:
                        output_files = result.get("output_files", {})
                        if "out_1" in output_files:
                            audio_url = output_files["out_1"].get("storage_url")
                            logger.info(f"✅ Audio extracted via cloud")
                            return audio_url
                    
                    elif status == "failed":
                        logger.error("❌ Cloud audio extraction failed")
                        return None
                
                time.sleep(5)
                
            except Exception:
                time.sleep(5)
        
        logger.error("❌ Audio extraction timeout")
        return None


# =============================================================================
# GEMINI SERVICE (Native Video Analysis via Kie.ai)
# =============================================================================
class GeminiService:
    """Service for Gemini 3 Pro via Kie.ai API - Native video analysis with reasoning."""
    
    def __init__(self, api_key: str, s3_service=None):
        """Initialize Gemini service via Kie.ai.
        
        Args:
            api_key: Kie.ai API key (same as used for other Kie.ai services).
            s3_service: S3 service for uploading videos to get public URLs.
        """
        self.api_key = api_key
        self.s3_service = s3_service
        # Updated to Gemini 3 Pro for better reasoning capabilities
        self.base_url = "https://api.kie.ai/gemini-3-pro/v1/chat/completions"
        self.initialized = False
        
        if not api_key:
            logger.warning("⚠️ Gemini not available - KIE_API_KEY not set")
            return
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.initialized = True
        logger.info("✅ Gemini 3 Pro client initialized (via Kie.ai)")
    
    def _prepare_video_for_analysis(self, video_path: str) -> Tuple[str, Optional[str]]:
        """Prepare a minimal-weight copy of the video for Gemini (single request).
        
        Uses 480p, 10fps, CRF 30 to keep payload small and reduce timeout risk.
        Keeps full duration and audio. Returns (path_to_upload, tmp_dir_or_none).
        """
        if not os.path.isfile(video_path):
            return (video_path, None)
        if not FFmpegProcessor.check_ffmpeg_installed():
            logger.info("   FFmpeg not available, uploading original video for Gemini")
            return (video_path, None)
        max_height, fps, crf = 480, 10, 30
        tmp_dir = tempfile.mkdtemp()
        out_path = os.path.join(tmp_dir, "gemini_light.mp4")
        try:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"scale=-2:{max_height}",
                "-r", str(fps),
                "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
                "-c:a", "aac", "-b:a", "64k",
                "-movflags", "+faststart",
                out_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0 or not os.path.isfile(out_path):
                logger.warning(f"   FFmpeg compress failed: {result.stderr[:200] if result.stderr else 'no output'}")
                try:
                    if os.path.isfile(out_path):
                        os.remove(out_path)
                    os.rmdir(tmp_dir)
                except Exception:
                    pass
                return (video_path, None)
            orig_size = os.path.getsize(video_path)
            new_size = os.path.getsize(out_path)
            logger.info(f"   Video prepared for Gemini: {orig_size // (1024*1024)}MB -> {new_size // (1024*1024)}MB (480p, 10fps, audio 64k)")
            return (out_path, tmp_dir)
        except Exception as e:
            logger.warning(f"   Prepare video for Gemini failed: {e}")
            try:
                if os.path.isfile(out_path):
                    os.remove(out_path)
                os.rmdir(tmp_dir)
            except Exception:
                pass
            return (video_path, None)
    
    def _split_video_into_three_parts(self, video_path: str) -> Tuple[List[Tuple[str, float, float]], Optional[str]]:
        """Split video into 3 parts by duration. Returns ([(path, start_sec, end_sec), ...], tmp_dir) or ([], None)."""
        if not os.path.isfile(video_path) or not FFmpegProcessor.check_ffmpeg_installed():
            return ([], None)
        duration = FFmpegProcessor.get_video_duration(video_path)
        if duration < 15.0:
            return ([], None)
        tmp_dir = tempfile.mkdtemp()
        out_paths = []
        try:
            t0, t1, t2 = 0.0, duration / 3.0, 2.0 * duration / 3.0
            for i, (start, end) in enumerate([(t0, t1), (t1, t2), (t2, duration)]):
                seg_path = os.path.join(tmp_dir, f"part_{i}.mp4")
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-ss", str(start), "-to", str(end),
                    "-c", "copy", seg_path
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if r.returncode != 0 or not os.path.isfile(seg_path):
                    logger.warning(f"   Segment {i + 1}/3 ffmpeg failed: {r.stderr[:150] if r.stderr else 'no output'}")
                    for p, _, _ in out_paths:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                    try:
                        os.rmdir(tmp_dir)
                    except Exception:
                        pass
                    return ([], None)
                out_paths.append((seg_path, start, end))
            logger.info(f"   Video split into 3 parts: 0-{t1:.1f}s, {t1:.1f}-{t2:.1f}s, {t2:.1f}-{duration:.1f}s")
            return (out_paths, tmp_dir)
        except Exception as e:
            logger.warning(f"   Split video failed: {e}")
            for p, _, _ in out_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
            try:
                os.rmdir(tmp_dir)
            except Exception:
                pass
            return ([], None)
    
    def _merge_segment_analyses(
        self,
        segment_analyses: List[Dict[str, Any]],
        segment_starts: List[float]
    ) -> Dict[str, Any]:
        """Merge 3 segment analysis JSONs into one. Add segment offset to scene times (model may return relative or full)."""
        if not segment_analyses:
            return self._get_empty_analysis()
        base = segment_analyses[0].copy()
        all_scenes = []
        for idx, anal in enumerate(segment_analyses):
            offset = segment_starts[idx] if idx < len(segment_starts) else 0.0
            for s in anal.get("scenes", []):
                sc = s.copy()
                sc["start_time"] = sc.get("start_time", 0) + offset
                if "end_time" in sc:
                    sc["end_time"] = sc["end_time"] + offset
                all_scenes.append(sc)
        base["scenes"] = sorted(all_scenes, key=lambda x: x.get("start_time", 0))
        if len(segment_analyses) > 1:
            last = segment_analyses[-1]
            base["cta"] = last.get("cta", base.get("cta", {}))
            vo_parts = [a.get("new_voiceover", {}).get("full_script", "") for a in segment_analyses]
            base["new_voiceover"] = {
                "full_script": " ".join(p for p in vo_parts if p).strip(),
                "word_count": sum(a.get("new_voiceover", {}).get("word_count", 0) for a in segment_analyses),
                "style": base.get("new_voiceover", {}).get("style", "")
            }
        return base
    
    def _build_segment_prompt(
        self,
        segment_index: int,
        num_segments: int,
        segment_start: float,
        segment_end: float,
        article_content: Optional[Dict[str, str]],
        transcript_excerpt: str,
        target_language: str
    ) -> str:
        """Short prompt for one video segment to avoid timeout."""
        article_short = ""
        if article_content:
            t = article_content.get("title", "")
            p = (article_content.get("first_paragraph") or "")[:300]
            article_short = f"Title: {t}\nSummary: {p}"
        transcript_short = (transcript_excerpt or "")[:800]
        return f"""You are analyzing segment {segment_index + 1} of {num_segments} of a video. This segment covers {segment_start:.1f}s to {segment_end:.1f}s of the full video (duration of this segment: {segment_end - segment_start:.1f}s).

TASK: Watch this segment only. Output valid JSON with:
- "scenes": array of scenes in this segment. Each scene: start_time and end_time in seconds RELATIVE to this segment (0 to {segment_end - segment_start:.1f}). Include understanding (what_happens, narrative_role, story_beat, story_connection, transition_logic), prompts (visible_elements, image_prompt, motion_prompt). product_visible true/false per scene.
- "product": detected, type, visual_description, purpose, usage_method, application_rules, best_frame_timestamps (times in seconds relative to this segment).
- "video_story": type, one_sentence_summary (for this segment), narrative_arc, subject_changes (has_visible_change, start_state, end_state).
- "new_voiceover": full_script (VO text for THIS segment only), word_count, style.
- "cta": needs_cta, button_text, scene_number (only for last segment).
- "style": aesthetic, lighting, mood, style_prefix.
- "audio": original_has_vo, original_vo_style, original_vo_gender, music_mood.

RULES: product_visible only when product is visible in scene. image_prompt and motion_prompt must match what you see. motion_prompt under 200 chars. visible_elements list what is visible; motion_prompt may only reference those. No text/logos on product.

Article context:
{article_short}

Transcript (full video):
{transcript_short}

Return valid JSON only."""

    def _call_gemini_one_request(self, prompt_text: str, video_url: str) -> Optional[Dict[str, Any]]:
        """Single Gemini API call: POST prompt+video, parse JSON. Returns analysis dict or None."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": video_url}}
                    ]
                }
            ],
            "stream": False,
            "include_thoughts": False,
            "reasoning_effort": "low"
        }
        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=180
                )
                if not response.ok:
                    err_info = f"HTTP {response.status_code}"
                    if response.status_code == 524:
                        err_info += " (gateway timeout - Kie.ai took too long)"
                    elif response.status_code == 500:
                        err_info += " (server exception)"
                    elif response.status_code == 429:
                        err_info += " (rate limit - too many requests)"
                    logger.warning(f"   Gemini segment: {err_info} (attempt {attempt + 1}/3)")
                    if response.status_code in (500, 524, 429) and attempt < 2:
                        wait = 45 * (attempt + 1)
                        logger.info(f"   Retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    return None
                response.raise_for_status()
                result = response.json()
                if result.get("code") is not None and result.get("code") != 200:
                    err_msg = result.get("msg", "Unknown")
                    logger.warning(f"   Gemini segment: Kie.ai code={result.get('code')} msg={err_msg} (attempt {attempt + 1}/3)")
                    if attempt < 2:
                        time.sleep(45 * (attempt + 1))
                        continue
                    return None
                choices = result.get("choices") or result.get("data", {}).get("choices")
                if not choices:
                    logger.warning(f"   Gemini segment: no choices in response keys={list(result.keys())} (attempt {attempt + 1}/3)")
                    return None
                msg = choices[0].get("message") or choices[0]
                text = (msg.get("content") or msg.get("text") or "").strip()
                for prefix in ("```json", "```"):
                    if text.startswith(prefix):
                        text = text[len(prefix):].strip()
                if text.endswith("```"):
                    text = text[:-3].strip()
                if not text:
                    logger.warning(f"   Gemini segment: empty content from API (attempt {attempt + 1}/3)")
                    if attempt < 2:
                        time.sleep(45 * (attempt + 1))
                    continue
                try:
                    return json.loads(text)
                except json.JSONDecodeError as je:
                    logger.warning(f"   Gemini segment: invalid JSON (attempt {attempt + 1}/3): {je}")
                    logger.info(f"   Response preview (first 400 chars): {repr(text[:400])}")
                    if attempt < 2:
                        time.sleep(45 * (attempt + 1))
                    continue
            except requests.exceptions.Timeout as e:
                logger.warning(f"   Gemini segment: request timeout (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(45 * (attempt + 1))
            except Exception as e:
                logger.warning(f"   Gemini segment request failed (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(45 * (attempt + 1))
        return None
    
    def _upload_video_to_s3(self, video_path: str) -> Optional[str]:
        """Upload video to S3 and return public URL.
        
        Args:
            video_path: Path to local video file.
            
        Returns:
            Public URL of the uploaded video, or None if failed.
        """
        if not self.s3_service:
            logger.warning("⚠️ S3 service not available for video upload")
            return None
        
        try:
            import uuid
            
            # Generate unique filename
            video_id = str(uuid.uuid4())[:8]
            s3_key = f"gemini_analysis/{video_id}.mp4"
            
            # Read video file
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            # Upload to S3
            logger.info(f"📤 Uploading video to S3 for Gemini analysis...")
            
            self.s3_service.client.put_object(
                Bucket=self.s3_service.bucket,
                Key=s3_key,
                Body=video_data,
                ContentType='video/mp4'
            )
            
            # Generate public URL
            video_url = f"https://{self.s3_service.bucket}.s3.{self.s3_service.region}.amazonaws.com/{s3_key}"
            logger.info(f"✅ Video uploaded to S3: {video_url[:60]}...")
            
            return video_url
            
        except Exception as e:
            logger.error(f"❌ Error uploading video to S3: {e}")
            return None
    
    def _cleanup_s3_video(self, video_url: str):
        """Delete temporary video from S3.
        
        Args:
            video_url: URL of the video to delete.
        """
        if not self.s3_service or not video_url:
            return
        
        try:
            # Extract key from URL
            parts = video_url.split('.amazonaws.com/')
            if len(parts) > 1:
                s3_key = parts[1]
                self.s3_service.client.delete_object(
                    Bucket=self.s3_service.bucket,
                    Key=s3_key
                )
                logger.info("🗑️ Cleaned up S3 video file")
        except Exception:
            pass  # Non-critical, ignore errors
    
    def _get_cultural_adaptation_instructions(self, target_language: str) -> str:
        """Get cultural adaptation instructions based on target language.
        
        CRITICAL: Characters and backgrounds must ALWAYS match the target country,
        regardless of what appears in the original video.
        
        Args:
            target_language: Target language code (e.g., 'en', 'es', 'ar', 'de').
            
        Returns:
            Detailed cultural adaptation instructions for prompts.
        """
        # Map language codes to regions and cultural details
        cultural_mapping = {
            # English - US/UK/AU
            'en': {
                'region': 'North America/Western',
                'country': 'United States',
                'ethnicity': 'diverse American population - Caucasian, African American, Hispanic, Asian American',
                'names': 'American names like Emma, Olivia, Liam, Noah, Michael, Jennifer',
                'environment': 'American urban and suburban settings - modern offices, American homes, shopping malls',
                'clothing': 'casual American fashion - jeans, t-shirts, sneakers, business casual',
            },
            'en-US': {
                'region': 'North America',
                'country': 'United States',
                'ethnicity': 'diverse American population - Caucasian, African American, Hispanic, Asian American',
                'names': 'American names like Emma, Olivia, Liam, Noah, Michael, Jennifer',
                'environment': 'American settings - NYC skyline, suburban homes, modern offices, American streets',
                'clothing': 'American fashion - casual wear, business casual, athleisure',
            },
            'en-GB': {
                'region': 'Western Europe',
                'country': 'United Kingdom',
                'ethnicity': 'British population - diverse, including British Asian, British African',
                'names': 'British names like Oliver, George, Amelia, Charlotte, Harry, Sophie',
                'environment': 'British settings - London streets, British homes, UK offices, red brick buildings',
                'clothing': 'British fashion - smart casual, conservative, classic styles',
            },
            # Spanish
            'es': {
                'region': 'Latin America',
                'country': 'Latin America',
                'ethnicity': 'Hispanic/Latino - Mexican, Colombian, Argentine features, warm skin tones, dark hair',
                'names': 'Spanish names like Sofia, Isabella, Diego, Carlos, Maria, Juan, Valentina',
                'environment': 'Latin American settings - colorful streets, colonial architecture, warm climates',
                'clothing': 'Latin American fashion - vibrant colors, casual and stylish, tropical appropriate',
            },
            # German
            'de': {
                'region': 'Western Europe',
                'country': 'Germany',
                'ethnicity': 'German/Central European - fair to light skin, varied hair colors',
                'names': 'German names like Lukas, Leon, Mia, Emma, Felix, Hannah, Maximilian',
                'environment': 'German settings - modern cities, efficient infrastructure, clean streets, German architecture',
                'clothing': 'German fashion - practical, high quality, understated elegance',
            },
            # French
            'fr': {
                'region': 'Western Europe',
                'country': 'France',
                'ethnicity': 'French population - diverse including African French, North African French',
                'names': 'French names like Emma, Gabriel, Léa, Louis, Chloé, Hugo, Camille',
                'environment': 'French settings - Parisian streets, French cafes, elegant architecture, countryside',
                'clothing': 'French fashion - chic, elegant, sophisticated, classic styles',
            },
            # Arabic
            'ar': {
                'region': 'Middle East / North Africa',
                'country': 'Arab World (UAE, Saudi Arabia, Egypt)',
                'ethnicity': 'Arab/Middle Eastern - olive to brown skin tones, dark hair, Middle Eastern features',
                'names': 'Arabic names like Mohammed, Ahmed, Fatima, Aisha, Omar, Layla, Youssef',
                'environment': 'Middle Eastern settings - modern Dubai, traditional markets, desert landscapes, Islamic architecture',
                'clothing': 'Middle Eastern fashion - modest clothing, traditional and modern mix, hijabs for women (optional)',
            },
            # Hebrew
            'he': {
                'region': 'Middle East',
                'country': 'Israel',
                'ethnicity': 'Israeli/Jewish - diverse including Ashkenazi, Sephardi, Mizrahi, Ethiopian',
                'names': 'Hebrew names like Noam, David, Tamar, Yael, Itai, Maya, Omer',
                'environment': 'Israeli settings - Tel Aviv beaches, Jerusalem, modern cities, Mediterranean climate',
                'clothing': 'Israeli fashion - casual, relaxed, Mediterranean style',
            },
            # Portuguese
            'pt': {
                'region': 'Latin America',
                'country': 'Brazil',
                'ethnicity': 'Brazilian - very diverse, mixed race, African Brazilian, European Brazilian',
                'names': 'Brazilian names like Pedro, Gabriel, Ana, Julia, Lucas, Maria, Beatriz',
                'environment': 'Brazilian settings - Rio beaches, São Paulo urban, tropical nature, vibrant cities',
                'clothing': 'Brazilian fashion - colorful, casual, beach-appropriate, trendy',
            },
            'pt-BR': {
                'region': 'Latin America',
                'country': 'Brazil',
                'ethnicity': 'Brazilian - very diverse, mixed race, African Brazilian, European Brazilian',
                'names': 'Brazilian names like Pedro, Gabriel, Ana, Julia, Lucas, Maria, Beatriz',
                'environment': 'Brazilian settings - Rio beaches, São Paulo urban, tropical nature, vibrant cities',
                'clothing': 'Brazilian fashion - colorful, casual, beach-appropriate, trendy',
            },
            # Italian
            'it': {
                'region': 'Southern Europe',
                'country': 'Italy',
                'ethnicity': 'Italian/Mediterranean - olive skin, dark hair, Southern European features',
                'names': 'Italian names like Francesco, Leonardo, Sofia, Giulia, Alessandro, Aurora',
                'environment': 'Italian settings - Roman streets, Venetian canals, Italian piazzas, Mediterranean coast',
                'clothing': 'Italian fashion - stylish, designer-conscious, elegant casual',
            },
            # Japanese
            'ja': {
                'region': 'East Asia',
                'country': 'Japan',
                'ethnicity': 'Japanese - East Asian features, typically black hair',
                'names': 'Japanese names like Haruto, Yui, Sota, Hina, Ren, Mei, Yuto',
                'environment': 'Japanese settings - Tokyo streets, traditional temples, modern cities, anime aesthetic',
                'clothing': 'Japanese fashion - modern Tokyo street style, clean lines, minimalist',
            },
            # Korean
            'ko': {
                'region': 'East Asia',
                'country': 'South Korea',
                'ethnicity': 'Korean - East Asian features, typically black hair, K-beauty aesthetic',
                'names': 'Korean names like Min-jun, Seo-yeon, Ji-ho, Ha-yun, Joon, Soo-ah',
                'environment': 'Korean settings - Seoul streets, K-pop aesthetic, modern cities, cafes',
                'clothing': 'Korean fashion - trendy K-fashion, modern, colorful, youthful',
            },
            # Chinese
            'zh': {
                'region': 'East Asia',
                'country': 'China',
                'ethnicity': 'Chinese - East Asian features, typically black hair',
                'names': 'Chinese names like Wei, Li, Ming, Xiao, Chen, Lin, Zhang',
                'environment': 'Chinese settings - modern Shanghai, Beijing, traditional temples, bustling cities',
                'clothing': 'Chinese fashion - modern Chinese urban style, mix of traditional and contemporary',
            },
            # Russian
            'ru': {
                'region': 'Eastern Europe',
                'country': 'Russia',
                'ethnicity': 'Russian/Slavic - fair skin, varied hair colors, Eastern European features',
                'names': 'Russian names like Dmitri, Anastasia, Ivan, Natalia, Alexei, Olga',
                'environment': 'Russian settings - Moscow streets, Russian architecture, winter scenes',
                'clothing': 'Russian fashion - practical, layered, fur accents, elegant',
            },
            # Hindi
            'hi': {
                'region': 'South Asia',
                'country': 'India',
                'ethnicity': 'Indian/South Asian - brown skin tones, dark hair, diverse Indian features',
                'names': 'Indian names like Aarav, Priya, Arjun, Ananya, Vihaan, Diya, Rohan',
                'environment': 'Indian settings - Delhi, Mumbai, colorful markets, Bollywood aesthetic',
                'clothing': 'Indian fashion - mix of traditional (saree, kurta) and modern Western',
            },
            # Turkish
            'tr': {
                'region': 'Middle East / Europe',
                'country': 'Turkey',
                'ethnicity': 'Turkish - Mediterranean to Middle Eastern, olive skin, dark hair',
                'names': 'Turkish names like Mehmet, Zeynep, Ali, Elif, Mustafa, Defne',
                'environment': 'Turkish settings - Istanbul streets, Turkish bazaars, Bosphorus views',
                'clothing': 'Turkish fashion - modern European mixed with traditional elements',
            },
            # Polish
            'pl': {
                'region': 'Eastern Europe',
                'country': 'Poland',
                'ethnicity': 'Polish/Slavic - fair skin, varied hair colors, Eastern European features',
                'names': 'Polish names like Jan, Zofia, Jakub, Julia, Kacper, Zuzanna',
                'environment': 'Polish settings - Warsaw, Krakow, European cities, Polish architecture',
                'clothing': 'Polish fashion - European casual, practical, modern',
            },
            # Thai
            'th': {
                'region': 'Southeast Asia',
                'country': 'Thailand',
                'ethnicity': 'Thai/Southeast Asian - tan skin, dark hair, Southeast Asian features',
                'names': 'Thai names like Somchai, Suda, Niran, Ploy, Chai, Kwang',
                'environment': 'Thai settings - Bangkok streets, Thai temples, tropical beaches, markets',
                'clothing': 'Thai fashion - light fabrics, bright colors, tropical appropriate',
            },
            # Vietnamese
            'vi': {
                'region': 'Southeast Asia',
                'country': 'Vietnam',
                'ethnicity': 'Vietnamese - East/Southeast Asian features, typically black hair',
                'names': 'Vietnamese names like Minh, Linh, Huy, Thao, Duc, Mai',
                'environment': 'Vietnamese settings - Hanoi, Ho Chi Minh City, Vietnamese streets, tropical',
                'clothing': 'Vietnamese fashion - modern Asian style, traditional ao dai for formal',
            },
        }
        
        # Get cultural info or use default
        lang_code = target_language.lower()
        cultural_info = cultural_mapping.get(lang_code, {
            'region': 'International',
            'country': 'Target country',
            'ethnicity': 'diverse population appropriate for the target region',
            'names': 'culturally appropriate names for the target language',
            'environment': 'settings appropriate for the target country',
            'clothing': 'fashion appropriate for the target culture',
        })
        
        return f"""
🌍🌍🌍 CRITICAL - CULTURAL ADAPTATION (MANDATORY!) 🌍🌍🌍
═══════════════════════════════════════════════════════════════════════════════════════

⚠️ YOU MUST CHANGE THE CHARACTERS AND ENVIRONMENT TO MATCH THE TARGET COUNTRY! ⚠️

TARGET LANGUAGE: {target_language.upper()}
TARGET COUNTRY/REGION: {cultural_info['country']} / {cultural_info['region']}

🚫 DO NOT KEEP THE ORIGINAL VIDEO'S CHARACTERS OR ENVIRONMENT! 🚫
The original video's people and backgrounds are for a DIFFERENT market.
You MUST create NEW characters and environments for the TARGET market.

✅ REQUIRED CHANGES:

1. **CHARACTERS (MANDATORY CHANGE!):**
   - Use {cultural_info['ethnicity']}
   - Use {cultural_info['names']}
   - DO NOT copy the original video's characters!
   - Example: If original has Arab person → For English US market → Use American person

2. **ENVIRONMENT/BACKGROUND (MANDATORY CHANGE!):**
   - Use {cultural_info['environment']}
   - DO NOT copy the original video's backgrounds!
   - Example: If original has Arabic text/architecture → For US market → Use American settings

3. **CLOTHING & STYLE:**
   - Use {cultural_info['clothing']}
   - Adapt to local fashion and cultural norms

4. **ALL TEXT & VOICEOVER:**
   - Must be in {target_language.upper()}
   - Use culturally appropriate phrases and expressions

📋 EXAMPLE TRANSFORMATION:
Original video: Arab woman in traditional dress, Arabic text, Middle Eastern city background
Target: English (US)
New video: American woman (diverse ethnicity), casual American fashion, American city/suburban background, English text

🎯 YOUR PROMPTS MUST DESCRIBE:
- Characters that look like they're from {cultural_info['country']}
- Environments that look like {cultural_info['region']}
- Clothing and fashion from {cultural_info['clothing']}
- Names like {cultural_info['names']}

THIS IS MANDATORY - DO NOT SKIP CULTURAL ADAPTATION!
"""
    
    def analyze_video_comprehensive(
        self, 
        video_path: str,
        article_content: Dict[str, str] = None,
        manual_instructions: str = "",
        original_transcript: str = "",
        target_language: str = "en",
        article_related_to_video: bool = True
    ) -> Dict[str, Any]:
        """Analyze entire video with Gemini 2.5 Flash via Kie.ai.
        
        This is much more cost-effective than sending individual frames to GPT-4o.
        Gemini processes the entire video natively for comprehensive analysis.
        
        Args:
            video_path: Path to the video file.
            article_content: Optional article content for context.
            manual_instructions: Optional manual instructions.
            original_transcript: Transcript of what's said in the video (from Whisper).
            target_language: Target language code for VO script (e.g., 'en', 'es', 'de').
            article_related_to_video: True if article is similar to video content (adapt video for new offer/language),
                                      False if article is fundamentally different (keep style but create new content).
            
        Returns:
            Comprehensive analysis including:
            - scene_breakdown: List of scenes with timestamps, descriptions, and purposes
            - product_info: Detected product details (type, purpose, usage, appearance)
            - visual_style: Color palette, lighting, composition, mood
            - narrative_structure: Hook, problem, solution, CTA structure
            - usage_contexts: How the product appears/is used in different scenes
            - audio_visual_relationship: How the VO relates to what's shown
            - style_prompt_prefix: Ready-to-use style prefix for generation prompts
        """
        if not self.initialized:
            logger.warning("⚠️ Gemini not initialized, returning empty analysis")
            return self._get_empty_analysis()
        
        video_url = None
        gemini_temp_dir = None
        
        try:
            # Single request: prepare one lightweight video (minimal quality to reduce tokens/timeout)
            path_to_upload, gemini_temp_dir = self._prepare_video_for_analysis(video_path)
            video_url = self._upload_video_to_s3(path_to_upload)
            if not video_url:
                logger.warning("⚠️ Could not upload video to S3, falling back to GPT-4o")
                return self._get_empty_analysis()
            
            # Build the comprehensive analysis prompt
            article_context = ""
            if article_content:
                title = article_content.get("title", "")
                first_p = article_content.get("first_paragraph", "")
                free_text = article_content.get("free_text", "")
                article_text_combined = free_text or f"{title}\n{first_p}"
                
                if title or first_p or free_text:
                    # Get cultural adaptation instructions based on target language
                    cultural_instructions = self._get_cultural_adaptation_instructions(target_language)
                    
                    if article_related_to_video:
                        # YES - Article is SIMILAR to video content
                        # Adapt the video to match the new article with a different offer/language
                        article_context = f"""
🔗 ARTICLE-VIDEO RELATIONSHIP: SIMILAR CONTENT (Article IS related to Video)
═══════════════════════════════════════════════════════════════════════════

ARTICLE CONTENT:
Title: {title}
Summary: {first_p[:500] if first_p else 'N/A'}
Full Text: {free_text[:1000] if free_text else 'N/A'}

✅ ADAPTATION STRATEGY (SIMILAR CONTENT):
The article describes a SIMILAR offer/product to what's shown in the video.
Your goal is to adapt the video for the new offer while keeping visuals SIMILAR to the original:

1. **KEEP THE SAME VISUAL STYLE** - The video's scenes, composition, and style should remain similar
2. **ADAPT THE PRODUCT/OFFER** - Replace the original product with the article's product (similar type)
3. **ADAPT THE MESSAGING** - Update text overlays and voiceover to match the article content
4. **ADAPT THE LANGUAGE** - All text and VO in the target language
5. **KEEP THE NARRATIVE STRUCTURE** - Same story flow (hook, problem, solution, CTA)

Example: Original video shows weight loss patches → Article is about different weight loss patches
→ Keep the visual style, body transformation narrative, but adapt for the new product

{cultural_instructions}

"""
                    else:
                        # NO - Article is FUNDAMENTALLY DIFFERENT from video content
                        # Keep the video's style and atmosphere but create entirely new content
                        article_context = f"""
🔄 ARTICLE-VIDEO RELATIONSHIP: DIFFERENT CONTENT (Article is NOT related to Video)
═══════════════════════════════════════════════════════════════════════════════════

ARTICLE CONTENT (NEW TOPIC):
Title: {title}
Summary: {first_p[:500] if first_p else 'N/A'}
Full Text: {free_text[:1000] if free_text else 'N/A'}

⚠️⚠️⚠️ CRITICAL ADAPTATION STRATEGY (DIFFERENT CONTENT) ⚠️⚠️⚠️
The article describes a COMPLETELY DIFFERENT offer/product than what's shown in the video.
You must CREATE NEW content while KEEPING the video's STYLE and ATMOSPHERE:

1. **KEEP THE VISUAL STYLE** - Preserve the video's aesthetic: lighting, camera work, mood, color palette, framing
2. **KEEP THE ATMOSPHERE** - Same energy level, same emotional tone, same production quality
3. **KEEP THE PACING** - Same scene durations and rhythm
4. **DO NOT USE THE ORIGINAL PRODUCT/OFFER** - The original video's product is IRRELEVANT
5. **CREATE NEW CONTENT FOR THE ARTICLE** - Base all visuals and messaging on the ARTICLE content only

🎯 YOUR MISSION: Create a NEW video that:
- LOOKS AND FEELS like the original (same style, mood, quality)
- But ADVERTISES the article's product/offer (NOT the original video's product)
- Has NEW visuals appropriate for the article content
- Has NEW voiceover based on the article
- Has NEW text overlays based on the article

Example: Original video shows shoe advertisement → Article is about work-from-home jobs
→ Keep the video's professional, energetic style → Create scenes showing people working from home
→ Do NOT show shoes anywhere → Create new narrative about remote work opportunities

IMPORTANT: When generating prompts, describe scenes that would be APPROPRIATE for the article's topic,
using the STYLE elements from the original video (lighting, camera angles, mood, energy).

{cultural_instructions}

"""
            
            # Language context for VO script
            language_context = f"""
⚠️⚠️⚠️ CRITICAL - LANGUAGE REQUIREMENT ⚠️⚠️⚠️
TARGET LANGUAGE: {target_language.upper()}
The new voiceover script (full_script) MUST be written ENTIRELY in {target_language.upper()}.
Do NOT use any other language. The script will be read by a TTS system in {target_language}.
"""
            
            instructions_context = ""
            if manual_instructions:
                instructions_context = f"""
MANUAL INSTRUCTIONS:
{manual_instructions}
"""
            
            # Include transcript if available
            transcript_context = ""
            if original_transcript:
                transcript_context = f"""
AUDIO TRANSCRIPT (what is being said in the video):
\"\"\"{original_transcript}\"\"\"

IMPORTANT: Analyze how the audio/voiceover relates to what's shown visually in each scene.
"""
            
            # Build goal statement based on article-video relationship
            if article_related_to_video:
                goal_statement = """You are an expert video director and storyteller. Your job is to DEEPLY UNDERSTAND this video's story and create PRECISE, ACCURATE prompts that recreate the ORIGINAL video's visuals and story exactly.

⚠️⚠️⚠️ YOUR GOAL: ADAPT the video for a NEW OFFER while keeping SIMILAR visuals ⚠️⚠️⚠️
- Watch the ORIGINAL video carefully - understand its visual style
- The new video should LOOK SIMILAR to the original
- But adapt the product/offer and messaging to match the ARTICLE content
- Your prompts should recreate the visual style while adapting the content"""
            else:
                goal_statement = """You are an expert video director and storyteller. Your job is to understand this video's VISUAL STYLE and create NEW content that matches the article while keeping the same STYLE and ATMOSPHERE.

⚠️⚠️⚠️ YOUR GOAL: CREATE NEW CONTENT with the SAME VISUAL STYLE ⚠️⚠️⚠️
- Watch the ORIGINAL video to understand its STYLE (lighting, camera work, mood, energy, quality)
- DO NOT copy the original video's content/product - it's COMPLETELY DIFFERENT from the article
- CREATE NEW visuals that are appropriate for the ARTICLE content
- The new video should FEEL LIKE the original (same style/mood) but SHOW the article's content
- Your prompts should describe NEW scenes for the article content, using the original's style"""

            # Build workflow steps based on article-video relationship
            if article_related_to_video:
                workflow_steps = """🎬 YOUR MISSION: Understand the VIDEO'S COMPLETE STORY and generate prompts that ADAPT it for the new offer.

⚠️⚠️⚠️ CRITICAL WORKFLOW - FOLLOW THIS EXACTLY ⚠️⚠️⚠️

STEP 1: UNDERSTAND THE COMPLETE STORY (DO THIS FIRST!)
1. **WATCH THE ENTIRE VIDEO** - Don't just analyze frames, watch the complete narrative
2. **IDENTIFY THE STORY TYPE** - Is it transformation? Demo? Testimonial? Problem-solution? Before/after?
3. **UNDERSTAND THE NARRATIVE ARC** - Beginning → Middle → End. What's the journey?
4. **UNDERSTAND SCENE CONNECTIONS** - How do scenes connect? What changes between scenes? Why?
5. **TRACK SUBJECT CHANGES** - Does the subject look different in different scenes? Why? (e.g., weight loss, mood change, clothing change)
6. **UNDERSTAND PRODUCT ROLE** - When does the product appear? What's its role in the story? How does it connect to the narrative?

STEP 2: ANALYZE EACH SCENE INDIVIDUALLY
For EACH scene, watch the ORIGINAL video at that scene's timestamp:
1. What do you ACTUALLY see? (subject appearance, clothing, setting, lighting, camera angle)
2. What's the EXACT visual state? (match the original exactly)
3. Is the product visible? (set product_visible accurately)
4. How does this scene connect to the previous scene? (what changed?)
5. What's the subject's state in THIS scene? (match the original exactly)

STEP 3: CREATE ADAPTED PROMPTS
Only AFTER understanding the complete story AND analyzing each scene, create prompts that:
- KEEP the ORIGINAL video's visual style (camera angles, lighting, mood)
- ADAPT the product to match the ARTICLE's product/offer
- ADAPT the messaging to match the ARTICLE content
- Match the scene structure of the original (same number of scenes, similar durations)
- Include the ARTICLE's product when appropriate (replacing the original product)"""
            else:
                workflow_steps = """🎬 YOUR MISSION: Extract the video's VISUAL STYLE and create NEW content for the article.

⚠️⚠️⚠️ CRITICAL WORKFLOW FOR DIFFERENT CONTENT - FOLLOW THIS EXACTLY ⚠️⚠️⚠️

STEP 1: EXTRACT THE VIDEO'S VISUAL STYLE (DO THIS FIRST!)
1. **WATCH THE ENTIRE VIDEO** - Focus on HOW it looks, not WHAT it shows
2. **IDENTIFY THE STYLE ELEMENTS:**
   - Lighting style (natural, studio, dramatic, soft, etc.)
   - Camera work (static, handheld, smooth movements, etc.)
   - Color palette (warm, cool, vibrant, muted, etc.)
   - Mood/energy (energetic, calm, professional, casual, etc.)
   - Production quality (UGC style, professional, cinematic, etc.)
   - Framing preferences (close-ups, wide shots, etc.)
3. **IDENTIFY THE PACING** - How long are scenes? What's the rhythm?
4. **IDENTIFY THE NARRATIVE STRUCTURE** - Hook → Problem → Solution → CTA?
5. **DO NOT FOCUS ON THE PRODUCT** - The original product is IRRELEVANT for this task

STEP 2: UNDERSTAND THE ARTICLE CONTENT
For EACH piece of information in the article:
1. What is the product/offer? (This is what we're advertising)
2. What are the benefits? (These should be shown in the video)
3. Who is the target audience? (People like this should appear in scenes)
4. What emotions should the video evoke? (Match the article's tone)
5. What call-to-action is needed? (What should viewers do?)

STEP 3: CREATE NEW PROMPTS WITH ORIGINAL STYLE
Create prompts for a NEW video that:
- HAS THE SAME STYLE as the original (lighting, camera, mood, quality, pacing)
- SHOWS NEW CONTENT appropriate for the ARTICLE
- DOES NOT include the original video's product AT ALL
- Features people, settings, and actions relevant to the ARTICLE
- Uses the same narrative structure (hook, problem, solution, CTA) but for the NEW topic
- Has the same number of scenes with similar durations as the original"""

            analysis_prompt = f"""{goal_statement}

{language_context}
{article_context}
{instructions_context}
{transcript_context}

{workflow_steps}

You will output:
1. **VIDEO STORY UNDERSTANDING** - Complete narrative, story type, subject journey, scene connections
2. **IMAGE PROMPTS** - PRECISE prompts for Nano Banana that match the ORIGINAL video's visuals exactly
3. **MOTION PROMPTS** - PRECISE animation prompts for Kling/Runway that match the ORIGINAL video's movement
4. **NEW VOICEOVER SCRIPT** - Complete VO script in the target language that matches the story
5. **CTA BUTTON** - If the video needs a call-to-action button

🔑 CRITICAL RULES FOR PROMPTS:

⚠️⚠️⚠️ PRODUCT VISIBILITY - MOST IMPORTANT ⚠️⚠️⚠️
THE PRODUCT DOES NOT APPEAR IN EVERY SCENE! Include it ONLY when it CONTRIBUTES TO THE STORY.
- Set product_visible=true ONLY when: (1) the product is visible in the scene AND (2) showing it adds to the story (e.g. product demo, application, result, showcase, CTA). Do NOT force the product into hook, problem, transition, or emotional shots if it doesn't belong.
- Set product_visible=false when: the product is not visible, OR the scene works better without it (e.g. hook, problem statement, reaction shot, transition). Not every scene needs the product.
- Example: Scene 1 hook (person talking) → product_visible=false; Scene 2 product application → product_visible=true; Scene 3 person walking away → product_visible=false; Scene 4 result with product → product_visible=true

**LOGICAL & HUMAN (reduce hallucinations):**
- Every scene must be LOGICAL and COHERENT: no surreal or inconsistent elements. People must look HUMAN and NATURAL (realistic poses, expressions, proportions).
- When product_visible=true, ALWAYS include a short verbal description of the product in the image_prompt (how it looks: shape, color, size, placement) so the model stays consistent - in addition to the reference image.
- Descriptions should be dynamic but believable; avoid exaggerated or artificial wording that causes AI artifacts.

**IMAGE PROMPTS (for Nano Banana) - MUST MATCH ORIGINAL VIDEO EXACTLY:**
- Watch the ORIGINAL video at this scene's timestamp - what do you ACTUALLY see?
- Recreate the EXACT visual: subject appearance, clothing, setting, lighting, camera angle
- ⚠️ ONLY include product if product_visible=true for this scene!
- If product_visible=true: describe product EXACTLY as it appears (color, shape, size, materials, EXACT location on body/object) in the prompt text as well
- If product_visible=false: DO NOT mention the product at all in the prompt
- ⚠️ CRITICAL: If subject changes between scenes (weight, appearance, mood, clothing) - describe the EXACT state in THIS scene
- Match the ORIGINAL video's visual style: camera angle, framing, lighting, mood
- Be SPECIFIC and DETAILED - the prompt should recreate the original scene visually
- Focus on describing what IS visible: people, objects, environments, lighting, mood, camera angles
- Format: "Photorealistic [exact shot type from original], [exact subject appearance from original], [exact action from original], [exact setting from original], [exact lighting from original], [exact mood from original]"

**MOTION PROMPTS (for video generation) - MUST MATCH ORIGINAL VIDEO EXACTLY:**
- Watch the ORIGINAL video at this scene's timestamp - what movement do you ACTUALLY see?
- Describe the EXACT movement: camera movement, subject motion, speed, direction
- Match the ORIGINAL video's pacing and style
- Keep under 200 characters
- Be SPECIFIC about the movement type (zoom, pan, static, tracking, etc.)
- Format: "[Exact camera movement from original], [exact subject action from original], [exact speed/style from original]"

**PRODUCT LOGIC - CRITICAL FOR ACCURACY:**
⚠️⚠️⚠️ Include the product ONLY when it CONTRIBUTES TO THE STORY in that scene!

For EACH scene:
1. Watch the original video at that scene's timestamp
2. Ask: Does showing the product in THIS scene add value to the story? (e.g. demo, application, result, CTA = yes; hook, problem, transition, emotional reaction = often no)
3. Set product_visible=true ONLY if the product is visible AND relevant to the story in this scene
4. Set product_visible=false if the product is not visible, or if the scene works better without it (do not force the product into every scene)

**PRODUCT VISUAL DESCRIPTION - EXTREMELY DETAILED:**
When describing the product in "visual_description", you MUST provide an EXTREMELY DETAILED 400+ word description that includes:

1. **EXACT SHAPE AND DIMENSIONS:**
   - Precise shape (circular, rectangular, oval, irregular, etc.)
   - Exact dimensions (e.g., "2cm diameter", "3cm x 4cm rectangle", "covers palm of hand")
   - Relative size to body parts or objects (e.g., "half the size of a credit card", "slightly larger than a quarter")

2. **EXACT COLORS (with specific hex codes or precise color names):**
   - Primary color with specific shade (e.g., "bright orange #FF6600", "pale beige #F5F5DC", "deep navy blue #000080")
   - Secondary colors if any (e.g., "white outer ring #FFFFFF", "transparent center")
   - Gradients or color transitions if present
   - NOT just "orange" or "white" - be SPECIFIC!

3. **EXACT MATERIALS AND TEXTURES:**
   - Material type (e.g., "smooth adhesive gel center", "matte white outer ring", "transparent film backing")
   - Texture description (e.g., "glossy surface", "matte finish", "smooth", "textured", "opaque", "semi-transparent")
   - How light interacts (e.g., "reflective surface", "absorbs light", "translucent")

4. **PRODUCT BRANDING (CRITICAL - MUST REMOVE!):**
   ⚠️ DO NOT include any text, logos, or branding on the product surface!
   - If the original product has text/logos → REMOVE them in your prompts
   - The product surface must be CLEAN and PLAIN
   - Describe the product's physical features ONLY (shape, color, material, texture)
   - Example: Instead of "patch with 'SlimFast' logo" → "plain circular patch with orange center"

5. **EXACT PACKAGING (if visible):**
   - Package color, shape, material
   - How product appears in packaging
   - NOTE: Package branding should also be removed/ignored

6. **EXACT PLACEMENT AND ORIENTATION:**
   - Precise location on body/object (e.g., "centered on lower abdomen 5cm below navel", "on right cheekbone")
   - Orientation (e.g., "horizontal", "vertical", "diagonal at 45 degrees")
   - How it sits on surface (e.g., "flat against skin", "slightly raised", "curved to match body contour")

7. **EXACT LIGHTING AND SHADOWS:**
   - How light hits the product (e.g., "soft natural light from above creates subtle highlight on center")
   - Shadow details (e.g., "slight shadow cast on skin below", "no shadow, flush with skin")

8. **EXACT PERSPECTIVE AND CAMERA ANGLE:**
   - Camera angle relative to product (e.g., "top-down view", "45-degree angle from side", "eye-level")
   - How product appears from this angle (e.g., "circular shape appears slightly oval from this angle")

9. **UNIQUE FEATURES:**
   - Any patterns, designs, or distinguishing marks
   - Any special characteristics that make this product identifiable

When product_visible=true, describe it POSITIVELY and SPECIFICALLY:
⚠️ CRITICAL: Use POSITIVE descriptions, NOT negative ones!
❌ BAD: "patch not on forehead, not on clothes"
✅ GOOD: "small circular patch (2cm diameter) with bright orange gel center (#FF6600) and matte white outer ring (#FFFFFF), adhered to the bare skin of the lower abdomen 5cm below navel, visible on clean exposed stomach area, soft natural lighting creates subtle highlight on glossy gel surface"

1. **SKIN PRODUCTS (patches, stickers, creams, serums):**
   - **Patches/stickers for weight loss/slimming:**
     * MUST say: "adhered to the bare skin of the [specific body part: stomach/abdomen, arm, thigh, etc.]"
     * MUST say: "on clean exposed skin" or "on bare skin visible under/above clothing"
     * Example: "small circular patch adhered to the bare skin of the lower abdomen, visible on clean exposed stomach area"
   
   - **Face patches/creams:**
     * MUST say: "applied to the [specific face area: forehead, cheek, under-eye, etc.]"
     * Example: "patch adhered to the forehead" or "cream being massaged into the cheek"
   
   - **Body creams/lotions:**
     * MUST say: "being rubbed/massaged into the [specific body part: arm, leg, stomach, etc.]"
     * Example: "cream being massaged into the bare skin of the arm"

2. **PET PRODUCTS (toys, treats, accessories):**
   - MUST say: "[pet type] actively [action: playing with, chewing, interacting with] the [product]"
   - Example: "Border Collie actively chewing and playing with the colorful ball toy"

3. **FOOD/DRINKS:**
   - MUST say: "[person/pet] [action: eating, drinking, preparing] the [product]"
   - Example: "person drinking from the bottle" or "hands preparing the food"

4. **SUPPLEMENTS/PILLS:**
   - MUST say: "hand holding [product] near mouth" or "[product] being taken from package"

**CRITICAL - PRODUCT VISIBILITY RULES:**
⚠️ THE PRODUCT DOES NOT APPEAR IN EVERY SCENE!
- Analyze the ORIGINAL video carefully: In which scenes is the product VISIBLE?
- In scenes where the product IS visible → Include it with POSITIVE, SPECIFIC description with EXACT colors (hex codes), dimensions, materials, and placement
- In scenes where the product is not visible → Focus on describing what IS visible: the subject, clothing, setting, lighting, mood, camera angle
- Example: Only include product where it contributes to the story (e.g. scene 2 application, scene 4 result). Hook, problem, transition scenes often work better without the product.

🚫🚫🚫 CRITICAL - BRANDING AND TEXT RULES 🚫🚫🚫

**PRODUCT SURFACE = NO TEXT/BRANDING (MANDATORY!)**
- Products must be shown COMPLETELY CLEAN - no text, no logos, no branding
- If original product has brand name/logo → REMOVE IT in your prompts
- The product surface must be plain and clean
- Example: Original shows "SlimPatch™" on product → Describe as "plain circular patch" (NO text on product)

**TEXT OVERLAYS - STRICT RULES!**

🚨 RULE 1: CHECK IF ORIGINAL VIDEO HAS TEXT OVERLAY
- Watch the original video carefully
- Does it have text/branding overlays on the screen (not on product)?
- If YES → You may add text overlay (extracted from article)
- If NO → DO NOT add any text overlay! Keep image CLEAN!

🚨 RULE 2: CHECK MANUAL INSTRUCTIONS
- If Manual Instructions say "remove text" or "no text" → NO text overlay at all!
- If Manual Instructions say "add text" → Add text even if original didn't have
- Manual Instructions OVERRIDE the original video analysis

🚨 RULE 3: TEXT MUST BE FROM THE ARTICLE CONTENT
- The text MUST be extracted from the article/Free text content
- Find the main offer, discount, benefit, or call-to-action IN THE ARTICLE
- Write it in the target language

✅ CORRECT EXAMPLES (text extracted from article):
- Article says "50% discount on all models" → Text: "50% OFF ALL MODELS"
- Article says "we're hiring drivers" → Text: "WE'RE HIRING!"
- Article says "free shipping this week" → Text: "FREE SHIPPING"
- Article says "משלוח חינם לכל הארץ" → Text: "משלוח חינם"

❌ WRONG - DON'T DO THIS:
- "promotional text" ← Technical description, not real text!
- "SALE" when article doesn't mention a sale ← Not from article!
- "BUY NOW" when article is about job hiring ← Unrelated!
- Adding text when original video had no text ← Violates Rule 1!

📋 DECISION FLOWCHART:
1. Does Manual Instructions say "remove text"? → NO TEXT
2. Does Manual Instructions say "add text"? → ADD TEXT (from article)
3. Does original video have text overlays? → If NO → NO TEXT
4. If original has text → Extract message from article → ADD THAT TEXT

| Condition | Action |
|-----------|--------|
| Manual says "remove text" | NO text overlay |
| Manual says "add text" | ADD text from article |
| Original has text + no manual override | ADD text from article |
| Original has NO text + no manual override | NO text overlay |

**WHEN DESCRIBING PRODUCTS IN IMAGE PROMPTS - POSITIVE LANGUAGE ONLY:**
⚠️ REMEMBER: Only describe product if product_visible=true for THIS scene!

If product_visible=true:
- ✅ DO: "patch adhered to the bare skin of the stomach area, visible on clean exposed abdomen"
- ❌ DON'T: "patch not on forehead, not on clothes"
- ✅ DO: "small circular patch on the lower abdomen, skin visible around it"
- ❌ DON'T: "patch not floating, not on fabric"
- Be SPECIFIC about EXACT location (stomach, arm, face area, etc.)
- Be SPECIFIC about EXACT action (adhered, being applied, being massaged, etc.)
- Include POSITIVE context (bare skin visible, clean exposed area, etc.)

If product_visible=false:
- ✅ DO: Describe the scene focusing on what IS visible - the subject, clothing, setting, lighting, mood, camera angle
- ✅ DO: "Photorealistic medium shot, young woman with slim athletic build wearing black workout clothes, confidently walking through bright modern bedroom, natural window lighting, energetic happy mood"
- Focus on describing the visual elements that ARE present in the scene

**VOICEOVER SCRIPT:**
- Match the STYLE and TONE of the original
- Use content from the article provided
- Match the video duration
- If original is energetic → new VO should be energetic
- If original is calm → new VO should be calm

Return a JSON object with these sections:

{{
  "scenes": [
    {{
      "scene_number": 1,
      "start_time": "0:00",
      "end_time": "0:03",
      "duration_seconds": 3,
      
      "understanding": {{
        "what_happens": "<describe EXACTLY what happens in this scene - watch the original video at this timestamp and describe what you see>",
        "narrative_role": "<hook/problem/solution/benefit/demo/result/cta/transition>",
        "story_beat": "<one of: hook, problem, agitation, solution_intro, demo, result, social_proof, cta. This defines the emotional purpose of this scene in the story arc.>",
        "story_connection": "<How does this scene connect to the previous scene? What changed? What's the progression?>",
        "transition_logic": "<How does this scene lead into the NEXT scene? e.g. 'close-up on painful expression -> next scene introduces the solution product'. For the last scene, describe the final call-to-action feeling.>",
        "subject_appearance": "<CRITICAL: How does the subject look in THIS SPECIFIC scene? Be EXACT: body type, clothing, expression, state, position. Match the ORIGINAL video exactly>",
        "visual_details": "<EXACT visual details from original: camera angle, framing, lighting, setting, colors, mood>",
        "text_on_screen": "<ONLY if original video has text overlay on screen: Extract the main offer/message from the article and write it here (e.g., '50% OFF', 'משלוח חינם'). If Manual Instructions say 'remove text' → leave EMPTY. If original has NO text overlay → leave EMPTY. The text MUST come from the article content!>",
        "has_branding_overlay": true | false,
        "product_visible": true | false,
        "product_action": "<what's being done with product, if visible - be specific about the action and HOW the product is used/held/displayed>",
        "changes_from_previous": "<What changed from the previous scene? Subject appearance? Setting? Mood? Product visibility?>"
      }},
      
      "prompts": {{
        "visible_elements": ["<LIST every body part, person, object, and item that is VISIBLE in this scene's image. Examples: 'face', 'hands', 'full body', 'product on table', 'coffee cup', 'feet', 'legs'. The motion_prompt may ONLY reference items from THIS list. If 'hands' is not listed here, motion_prompt MUST NOT mention hand movement.>"],
        
        "image_prompt": "<COMPLETE, READY-TO-USE prompt for Nano Banana. Must be photorealistic, detailed, include all visual elements. 
        ⚠️ CRITICAL: Only include the product if 'product_visible' is true (i.e. product contributes to the story in this scene). Do not force the product into every scene.
        🚫 PRODUCT MUST BE CLEAN - NO text, logos, or branding on the product surface!
        
        PRODUCT RULES:
        - If product_visible=true: Include product with POSITIVE, SPECIFIC description (colors, dimensions, materials, placement, and FUNCTION - what does this product do?). Product must be PLAIN with no text/branding.
        - If product_visible=false: Do NOT mention the product. Describe only subject, setting, lighting, mood, camera angle. Many scenes (hook, problem, transition) work better without the product.
        
        🚨 TEXT OVERLAY RULES (STRICT!):
        1. If Manual Instructions say 'remove text' or 'no text' → DO NOT include any text in prompt
        2. If original video has NO text overlays → DO NOT include any text in prompt
        3. ONLY if original has text AND Manual Instructions don't forbid it → Include text FROM THE ARTICLE
        
        When adding text (only if rules 1-3 allow):
        - Extract the actual offer/message from the article content
        - Write the real text, not a description
        - Example: 'small white text 50% OFF in top-right corner' (where 50% OFF is from article)
        
        ❌ NEVER DO:
        - Add text when original video had no text
        - Add text when Manual Instructions say to remove it
        - Write 'promotional text' or 'text overlay' instead of the actual text
        - Use generic text like 'SALE' if article doesn't mention a sale
        
        Example with product (no text): 'Photorealistic medium shot, young woman showing her flat stomach with a small plain circular patch (2.5cm diameter, bright orange #FF6600 center, NO text on patch), bright modern bedroom, natural lighting'
        Example WITH text (only if original had text AND article mentions this offer): 'Photorealistic medium shot, small white text 50% OFF in top-right corner, young woman showing clean product, natural lighting'
        Example NO text (original had no text): 'Photorealistic medium shot, young woman confidently walking through bright modern bedroom, natural window lighting, energetic happy mood'>",
        
        "motion_prompt": "<Animation for Kling/Runway, under 200 chars. MUST start with camera movement (e.g. Slow zoom in, Subtle pan right, Static shot). Then ONLY describe motion for items listed in 'visible_elements' above. If 'hands' is NOT in visible_elements, do NOT say hand movement. If an object is NOT in visible_elements, do NOT mention it. Example: visible_elements=['face'] -> 'Slow zoom in, subtle smile'. visible_elements=['face','hands','product'] -> 'Gentle pan right, slight hand movement'. NEVER invent motion for elements not in visible_elements.>"
      }}
    }}
  ],
  
  "product": {{
    "detected": true | false,
    "type": "<product category: patch, cream, toy, supplement, etc.>",
    "visual_description": "<EXTREMELY DETAILED 500+ word description for image generation. MUST include ALL of the following in extreme detail:
    
    **COLORS (CRITICAL - BE EXTREMELY SPECIFIC):**
    - Primary color with EXACT hex code (e.g., 'bright orange #FF6600', 'pale beige #F5F5DC', 'deep navy blue #000080')
    - Secondary colors with EXACT hex codes (e.g., 'white outer ring #FFFFFF', 'transparent center #FFFFFF with 30% opacity')
    - Color gradients if present (e.g., 'gradient from #FF6600 at center to #FF8833 at edges')
    - Color patterns if present (e.g., 'striped pattern with alternating #FF6600 and #FFFFFF')
    - Color saturation and brightness (e.g., 'highly saturated bright orange', 'muted pale beige')
    - NOT just 'orange' or 'white' - MUST include hex codes and specific shade descriptions!
    
    **SHAPE AND DIMENSIONS (CRITICAL - BE EXTREMELY SPECIFIC):**
    - Precise shape (circular, rectangular, oval, irregular, etc.) with EXACT measurements
    - Exact dimensions in cm/mm (e.g., '2.5cm diameter circle', '3cm x 4cm rectangle', 'oval 2cm x 3cm')
    - Relative size to body parts or objects (e.g., 'half the size of a credit card', 'covers palm of hand', 'slightly larger than a quarter')
    - Thickness/depth if visible (e.g., '2mm thick', 'flat and flush with skin', 'slightly raised 1mm above skin')
    - Edge details (e.g., 'rounded edges', 'sharp corners', 'beveled edge')
    
    **MATERIALS AND TEXTURES (CRITICAL - BE EXTREMELY SPECIFIC):**
    - Material type with texture details (e.g., 'smooth adhesive gel center with glossy surface', 'matte white outer ring with paper-like texture', 'transparent film backing with slight texture')
    - Surface finish (e.g., 'glossy reflective surface', 'matte non-reflective finish', 'semi-gloss with subtle sheen')
    - Texture description (e.g., 'smooth', 'textured', 'opaque', 'semi-transparent', 'translucent', 'grainy', 'satin finish')
    - How light interacts (e.g., 'reflective surface creates bright highlight', 'absorbs light creating matte appearance', 'translucent allowing light to pass through')
    - Material properties (e.g., 'flexible and conforms to skin', 'rigid and maintains shape', 'stretchy elastic material')
    
    **PRODUCT BRANDING (CRITICAL - MUST BE REMOVED!):**
    ⚠️⚠️⚠️ DO NOT include any text, logos, or branding ON THE PRODUCT SURFACE! ⚠️⚠️⚠️
    - When describing the product, OMIT all text/logos that appear on it
    - The product must be shown as CLEAN and PLAIN - no branding
    - Only describe physical characteristics: shape, colors, materials, textures, dimensions
    - Example: Original has "SlimPatch™" logo → Describe as "plain circular patch" (NO text)
    
    **PACKAGING (if visible):**
    - Package color with hex codes (e.g., 'white box #FFFFFF', 'blue label #0066CC')
    - Package shape and size (e.g., 'rectangular box 5cm x 8cm', 'circular container 4cm diameter')
    - REMOVE package branding/text - describe only physical appearance
    - How product appears in packaging (e.g., 'product visible through transparent window', 'product wrapped in foil')
    
    **PLACEMENT AND ORIENTATION (CRITICAL - BE EXTREMELY SPECIFIC):**
    - Precise location on body/object (e.g., 'centered on lower abdomen 5cm below navel', 'on right cheekbone 2cm from eye', 'on upper arm 10cm from shoulder')
    - Orientation (e.g., 'horizontal alignment', 'vertical alignment', 'diagonal at 45 degrees', 'rotated 30 degrees clockwise')
    - How it sits on surface (e.g., 'flat against skin with no gaps', 'slightly raised 1mm above skin', 'curved to match body contour', 'adhered flush with no visible edges')
    - Relationship to surrounding elements (e.g., 'surrounded by bare skin', 'partially covered by clothing', 'visible through transparent material')
    
    **LIGHTING AND SHADOWS (CRITICAL - BE EXTREMELY SPECIFIC):**
    - How light hits the product (e.g., 'soft natural light from above creates subtle highlight on center', 'harsh directional light creates strong contrast', 'diffused light creates even illumination')
    - Shadow details (e.g., 'slight shadow cast on skin below creating depth', 'no shadow, flush with skin', 'soft shadow around edges')
    - Highlights and reflections (e.g., 'bright highlight on glossy center', 'matte surface shows no reflections', 'reflective surface shows window reflection')
    - Light temperature (e.g., 'warm 3000K lighting', 'cool 6000K daylight', 'neutral 5000K lighting')
    
    **PERSPECTIVE AND CAMERA ANGLE (CRITICAL - BE EXTREMELY SPECIFIC):**
    - Camera angle relative to product (e.g., 'top-down view looking straight down', '45-degree angle from side', 'eye-level view', 'close-up macro view')
    - How product appears from this angle (e.g., 'circular shape appears slightly oval from this angle', 'rectangular shape appears as trapezoid due to perspective')
    - Distance from camera (e.g., 'close-up filling 30% of frame', 'medium shot with product in center', 'wide shot with product as small element')
    - Depth of field (e.g., 'product in sharp focus with blurred background', 'entire scene in focus', 'shallow depth of field with product sharp')
    
    **FUNCTIONALITY AND USAGE (CRITICAL - BE EXTREMELY SPECIFIC):**
    - How the product functions (e.g., 'adhesive patch that sticks to skin', 'cream that is massaged into skin', 'toy that pet interacts with')
    - How it's being used in the scene (e.g., 'being applied to bare skin', 'already adhered and visible', 'being removed from packaging')
    - Interaction with user/environment (e.g., 'person's hand applying the patch', 'patch visible on person's body', 'product being held near face')
    - State of product (e.g., 'new unused product', 'product in use', 'product showing results')
    
    **UNIQUE FEATURES AND DISTINGUISHING MARKS:**
    - Any patterns, designs, or distinguishing marks (e.g., 'circular pattern in center', 'logo in corner', 'serial number visible')
    - Any special characteristics that make this product identifiable (e.g., 'unique color combination', 'distinctive shape', 'characteristic texture')
    - Branding elements (e.g., 'brand logo visible', 'product name printed', 'certification mark')
    
    This description must be so detailed that an AI image generator can recreate the product pixel-perfectly with exact colors, dimensions, materials, and appearance.>",
    "purpose": "<what does this product do?>",
    "usage_method": "<how is it used? step by step>",
    "application_rules": "<CRITICAL: where does it go? bare skin? with pet? etc.>",
    "best_frame_timestamps": ["0:02", "0:08"],
    "product_image_details": "<If product is visible in frames, describe the EXACT visual appearance in the best frame with EXTREME DETAIL:
    - EXACT color composition: Primary color with hex code, secondary colors with hex codes, gradients, patterns, saturation, brightness
    - EXACT shape and proportions: Precise measurements (cm/mm), relative size to known objects, thickness/depth, edge details
    - EXACT text/logos: Transcribe word-for-word, font style, size, position, color with hex code, text effects
    - EXACT material appearance: Material type, surface finish (glossy/matte/semi-gloss), texture (smooth/textured/grainy), opacity (opaque/transparent/translucent), material properties
    - EXACT lighting interaction: How light hits product, shadow details, highlights and reflections, light temperature
    - EXACT position and orientation: Precise location, orientation (horizontal/vertical/diagonal), how it sits on surface, relationship to surrounding elements
    - EXACT perspective: Camera angle, distance, depth of field, how product appears from this angle
    - EXACT functionality: How product functions, how it's being used, interaction with user/environment, state of product
    - Unique features: Patterns, designs, distinguishing marks, branding elements, special characteristics
    This must be detailed enough for pixel-perfect recreation.>"
  }},
  
  "video_story": {{
    "type": "<transformation/demo/testimonial/lifestyle/tutorial/problem_solution/ugc_review>",
    "one_sentence_summary": "<What happens in this video in one sentence - the complete story>",
    "narrative_arc": "<Describe the complete story arc: beginning → middle → end. What's the journey?>",
    "key_moments": [
      "<Important moment 1: what happens and why it matters>",
      "<Important moment 2: what happens and why it matters>"
    ],
    "scene_connections": "<How do scenes connect? What's the progression? What changes between scenes?>",
    "subject_changes": {{
      "has_visible_change": true | false,
      "start_state": "<How subject looks/feels at START - be specific: overweight, tired, messy, clothing, expression, etc.>",
      "end_state": "<How subject looks/feels at END - be specific: slim, energetic, groomed, clothing, expression, etc.>",
      "subject_appearance_per_scene": {{
        "1": "<EXACT appearance in scene 1: body type, clothing, expression, state>",
        "2": "<EXACT appearance in scene 2: body type, clothing, expression, state>",
        "3": "<EXACT appearance in scene 3: body type, clothing, expression, state>"
      }}
    }},
    "product_role_in_story": "<What's the product's role in the story? When does it appear? How does it connect to the narrative?>"
  }},
  
  "new_voiceover": {{
    "full_script": "<COMPLETE voiceover script for the new video. Use article content. Match original style and duration. This is the EXACT text for TTS.>",
    "word_count": <number>,
    "style": "<enthusiastic/calm/professional/friendly - match original>"
  }},
  
  "cta": {{
    "needs_cta": true | false,
    "button_text": "<CTA button text if needed, e.g., 'Shop Now', 'Try Now'>",
    "scene_number": <which scene to add CTA to, usually last>
  }},
  
  "style": {{
    "aesthetic": "<social_media/cinematic/ugc/professional/minimalist>",
    "lighting": "<natural/studio/dramatic/soft>",
    "mood": "<energetic/calm/luxurious/casual/playful>",
    "style_prefix": "<Concise style description for all prompts, e.g., 'UGC social media style, bright natural lighting, handheld camera feel, casual authentic mood'>"
  }},
  
  "audio": {{
    "original_has_vo": true | false,
    "original_vo_style": "<enthusiastic/calm/professional/friendly>",
    "original_vo_gender": "<male/female/unknown>",
    "music_mood": "<energetic/calm/uplifting/dramatic/none>"
  }}
}}

⚠️ CRITICAL - READ CAREFULLY:

0. **STORY ACCURACY AND COHERENCE - MOST IMPORTANT:**
   - ⚠️⚠️⚠️ YOU MUST WATCH THE ORIGINAL VIDEO CAREFULLY FOR EACH SCENE! ⚠️⚠️⚠️
   - Your prompts MUST recreate the ORIGINAL video's visuals EXACTLY - don't invent new visuals!
   - Match the ORIGINAL: camera angles, framing, lighting, subject appearance, setting, mood, colors, textures
   - **NARRATIVE ARC**: Understand the COMPLETE STORY before creating prompts. Each scene has a purpose (story_beat): hook → problem → agitation → solution_intro → demo → result → social_proof → cta. The scenes must flow logically and emotionally.
   - **SCENE CONNECTIONS**: Each scene must specify WHY it follows the previous one (story_connection) and HOW it leads to the next (transition_logic). Scenes are NOT isolated - they tell a coherent story together.
   - Track how scenes CONNECT - what changes between scenes? Why? What's the story progression?
   - Each prompt should reflect the EXACT visual state from the ORIGINAL video at that timestamp
   - **EMOTIONAL PROGRESSION**: image_prompt must include the emotional state and action that fits the story_beat (e.g., hook=curiosity/surprise, problem=frustration/pain, result=happiness/relief, cta=excitement/urgency)
   - If the original shows a transformation (e.g., overweight → slim), your prompts MUST show the CORRECT state for each scene
   - If the original shows different settings/clothing/mood in different scenes, your prompts MUST match this exactly
   - The goal is to recreate the ORIGINAL video's story and visuals, not create a new story

1. **PRODUCT VISIBILITY - SECOND MOST IMPORTANT:**
   - ⚠️ THE PRODUCT DOES NOT APPEAR IN EVERY SCENE! Include it only when it CONTRIBUTES TO THE STORY.
   - Set product_visible=true ONLY when the product is visible AND adds value in that scene (e.g. product demo, application, result, CTA). Do NOT force it into hook, problem, transition, or emotional shots.
   - Set product_visible=false when the product is not visible, or when the scene works better without it (e.g. hook, problem, reaction, transition).
   - If product_visible=false → DO NOT mention the product in image_prompt at all
   - If product_visible=true → Include product with POSITIVE, SPECIFIC description and EXACT location

2. **IMAGE PROMPTS MUST MATCH ORIGINAL VIDEO EXACTLY:**
   - Watch the ORIGINAL video at this scene's timestamp - what do you ACTUALLY see?
   - Recreate EXACTLY: subject appearance, clothing, setting, lighting, camera angle, mood
   - ⚠️ ONLY include product if product_visible=true for that scene
   - Be SPECIFIC about subject's physical state in EACH scene (if subject changes, show the CORRECT state for THIS scene)
   - Match the ORIGINAL video's visual style - don't invent new visuals
   - Include ALL details you see in the original: colors, textures, expressions, positions

3. **MOTION PROMPTS MUST MATCH ORIGINAL VIDEO AND VISIBLE_ELEMENTS:**
   - Watch the ORIGINAL video at this scene's timestamp - what movement do you ACTUALLY see?
   - Describe the EXACT movement: camera movement, subject motion, speed, direction
   - **CRITICAL: motion_prompt may ONLY reference items listed in visible_elements.** If "hands" is NOT in visible_elements, do NOT mention hand movement. If "bag" is NOT in visible_elements, do NOT mention the bag.
   - Match the ORIGINAL video's pacing and style
   - Keep under 200 characters
   - Be SPECIFIC about the movement type (zoom, pan, static, tracking, etc.)

4. **NEW VOICEOVER MUST MATCH VIDEO DURATION AND STORY:**
   - Use article content provided
   - Match the style/energy of the original
   - Match the STORY STRUCTURE of the original (hook, problem, solution, etc.)
   - Calculate approximate word count based on duration (2-3 words per second)

5. **PRODUCT LOGIC IS CRITICAL:**
   - Patches/stickers → BARE SKIN only (not on clothes!) - describe EXACT location
   - Pet products → Pet must be visible and interacting
   - Creams → On visible skin - describe EXACT location
   
6. **TRACK SUBJECT CHANGES ACCURATELY:**
   - If someone is overweight in scene 1 and slim in scene 5 → reflect this EXACTLY in prompts!
   - Each scene's image_prompt should show subject in the CORRECT state for that scene
   - Use subject_appearance_per_scene to track changes accurately
   - Match the ORIGINAL video's subject appearance at each timestamp

Return valid JSON only."""

            # 500 = server exception; 524 = gateway timeout (Kie.ai/Cloudflare gave up waiting for Gemini).
            # Long prompt + video often causes 524. Shorten prompt when too long to reduce timeout risk.
            GEMINI_PROMPT_MAX_CHARS = 22000
            if len(analysis_prompt) > GEMINI_PROMPT_MAX_CHARS:
                head_keep = 11000
                tail_keep = 10000
                analysis_prompt = (
                    analysis_prompt[:head_keep]
                    + "\n\n[... middle truncated to avoid 524 timeout ...]\n\n"
                    + analysis_prompt[-tail_keep:]
                )
                logger.info(f"   Prompt truncated to {len(analysis_prompt)} chars (was over {GEMINI_PROMPT_MAX_CHARS}) to reduce timeout risk.")
            prompt_len = len(analysis_prompt)
            logger.info("🔍 Analyzing video with Gemini 3 Pro (via Kie.ai)...")
            logger.info(f"   Request: prompt length={prompt_len} chars, video_url={video_url[:80] if video_url else 'N/A'}...")

            # Build request payload for Kie.ai Gemini 3 Pro endpoint
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": video_url
                                }
                            }
                        ]
                    }
                ],
                "stream": False,
                "include_thoughts": False,
                "reasoning_effort": "low"  # Fast response for video analysis
            }
            
            # Send request to Kie.ai Gemini endpoint (retry on 500 server error or 524 timeout)
            max_retries = 2
            result = None
            for attempt in range(max_retries + 1):
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=300  # 5 minute timeout for video analysis
                )
                # 524 = gateway timeout (origin took too long); 500 = server exception
                if not response.ok and response.status_code in (500, 524) and attempt < max_retries:
                    wait_sec = 45 * (attempt + 1)
                    logger.warning(f"   Kie.ai returned {response.status_code} ({'timeout' if response.status_code == 524 else 'server error'}). Retrying in {wait_sec}s...")
                    time.sleep(wait_sec)
                    continue
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"📥 Gemini response status: {response.status_code}")
                logger.info(f"📥 Gemini response keys: {list(result.keys())}")
                
                # Kie.ai error format: {"code": 500, "msg": "Server exception, ..."}
                if result.get("code") is not None and result.get("code") != 200:
                    err_msg = result.get("msg", "Unknown error")
                    logger.warning(f"⚠️ Kie.ai Gemini returned code={result.get('code')}: {err_msg}")
                    if attempt < max_retries:
                        wait_sec = 45 * (attempt + 1)  # 45s, then 90s - give Kie.ai server time to recover
                        logger.info(f"   Retrying in {wait_sec}s (attempt {attempt + 2}/{max_retries + 1})...")
                        time.sleep(wait_sec)
                        continue
                    logger.error(f"❌ Gemini analysis failed after {max_retries + 1} attempts. Kie.ai: {err_msg}")
                    self._cleanup_s3_video(video_url)
                    return self._get_empty_analysis()
                
                if "error" in result:
                    logger.error(f"❌ Gemini API error: {result.get('error')}")
                    self._cleanup_s3_video(video_url)
                    return self._get_empty_analysis()
                
                if "choices" in result and len(result["choices"]) > 0:
                    break
                if attempt < max_retries:
                    wait_sec = 45 * (attempt + 1)  # 45s, then 90s
                    logger.warning(f"⚠️ No choices in Gemini response. Retrying in {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    logger.error(f"❌ No choices in Gemini response after {max_retries + 1} attempts. Keys: {list(result.keys())}")
                    logger.error(f"❌ Full response: {json.dumps(result)[:1000]}")
                    self._cleanup_s3_video(video_url)
                    return self._get_empty_analysis()
            
            # Extract content from response
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                logger.info(f"📥 Choice keys: {list(choice.keys())}")
                message = choice.get("message", {})
                logger.info(f"📥 Message keys: {list(message.keys())}")
                response_text = message.get("content", "")
                logger.info(f"📥 Content length: {len(response_text)} chars")
            else:
                logger.error(f"❌ No choices in Gemini response. Keys: {list(result.keys())}")
                logger.error(f"❌ Full response: {json.dumps(result)[:1000]}")
                return self._get_empty_analysis()
            
            # Clean up response if needed
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Log raw response for debugging
            logger.info(f"📥 Gemini raw response (first 500 chars): {response_text[:500]}")
            
            analysis = json.loads(response_text.strip())
            
            logger.info(f"✅ Gemini video analysis complete:")
            logger.info(f"   - Scenes detected: {len(analysis.get('scenes', []))}")
            logger.info(f"   - Product detected: {analysis.get('product', {}).get('detected', False)}")
            logger.info(f"   - Video type: {analysis.get('video_story', {}).get('type', 'unknown')}")
            logger.info(f"   - Style: {analysis.get('style', {}).get('aesthetic', 'unknown')}")
            
            # Log prompts info
            scenes = analysis.get('scenes', [])
            if scenes:
                logger.info(f"   - First scene image prompt: {scenes[0].get('prompts', {}).get('image_prompt', 'N/A')[:60]}...")
            if analysis.get('new_voiceover', {}).get('full_script'):
                logger.info(f"   - New VO script: {analysis['new_voiceover']['full_script'][:60]}...")
            
            # Clean up the uploaded video from S3
            self._cleanup_s3_video(video_url)
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse Gemini response as JSON: {e}")
            if video_url:
                self._cleanup_s3_video(video_url)
            return self._get_empty_analysis()
        except Exception as e:
            logger.error(f"❌ Error in Gemini video analysis: {e}")
            import traceback
            traceback.print_exc()
            if video_url:
                self._cleanup_s3_video(video_url)
            return self._get_empty_analysis()
        finally:
            # Remove temp dir from compressed video for Gemini (if any)
            if gemini_temp_dir and os.path.isdir(gemini_temp_dir):
                try:
                    for f in os.listdir(gemini_temp_dir):
                        p = os.path.join(gemini_temp_dir, f)
                        if os.path.isfile(p):
                            os.remove(p)
                    os.rmdir(gemini_temp_dir)
                except Exception:
                    pass
    
    def _get_empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure when Gemini is unavailable."""
        return {
            "scenes": [],
            "product": {
                "detected": False,
                "type": "",
                "visual_description": "",
                "purpose": "",
                "usage_method": "",
                "application_rules": "",
                "best_frame_timestamps": []
            },
            "video_story": {
                "type": "unknown",
                "one_sentence_summary": "",
                "subject_changes": {
                    "has_visible_change": False,
                    "start_state": "",
                    "end_state": ""
                }
            },
            "new_voiceover": {
                "full_script": "",
                "word_count": 0,
                "style": ""
            },
            "cta": {
                "needs_cta": False,
                "button_text": "",
                "scene_number": 0
            },
            "style": {
                "aesthetic": "modern",
                "lighting": "",
                "mood": "",
                "style_prefix": ""
            },
            "audio": {
                "original_has_vo": False,
                "original_vo_style": "",
                "original_vo_gender": "unknown",
                "music_mood": ""
            }
        }
    
    def get_scene_prompt_context(
        self, 
        analysis: Dict[str, Any], 
        scene_number: int
    ) -> Dict[str, Any]:
        """Extract relevant context for a specific scene from the comprehensive analysis.
        
        Args:
            analysis: Full video analysis from analyze_video_comprehensive()
            scene_number: 1-indexed scene number
            
        Returns:
            Context dict with scene-specific and global style information.
        """
        scenes = analysis.get("scenes", [])
        product = analysis.get("product", {})
        style = analysis.get("style", {})
        
        # Find the specific scene
        scene_info = {}
        for scene in scenes:
            if scene.get("scene_number") == scene_number:
                scene_info = scene
                break
        
        # Get prompts from scene
        prompts = scene_info.get("prompts", {})
        understanding = scene_info.get("understanding", {})
        
        return {
            "scene_info": scene_info,
            "understanding": understanding,
            "prompts": prompts,
            "product": product,
            "style": style,
            "style_prefix": style.get("style_prefix", ""),
            "narrative_role": understanding.get("narrative_role", ""),
            "image_prompt": prompts.get("image_prompt", ""),
            "motion_prompt": prompts.get("motion_prompt", ""),
            "product_visible": understanding.get("product_visible", False),
            "product_action": understanding.get("product_action", "")
        }


# =============================================================================
# OPENAI SERVICE
# =============================================================================
class OpenAIService:
    """Service for OpenAI API interactions."""
    
    def __init__(self, api_key: str):
        """Initialize OpenAI service.
        
        Args:
            api_key: OpenAI API key.
        """
        self.client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI client initialized")
    
    def _get_cultural_style_instructions(self, language: str) -> str:
        """Get cultural style instructions for image/video prompts based on target language.
        
        CRITICAL: Characters and environments MUST be changed to match the target country.
        DO NOT keep the original video's characters or backgrounds!
        
        Args:
            language: Target language code (e.g., 'es', 'de', 'hu').
            
        Returns:
            String with detailed cultural adaptation instructions.
        """
        # Get region from language
        region = config.REGION_MAPPING.get(language, 'namer')
        cultural_style = config.CULTURAL_STYLES.get(region, {})
        
        if not cultural_style:
            return """
⚠️⚠️⚠️ CULTURAL ADAPTATION (MANDATORY!) ⚠️⚠️⚠️
You MUST change the characters and environment to match the target language/country.
DO NOT keep the original video's characters or backgrounds!
Use diverse, multicultural representations appropriate for the target market.
"""
        
        instructions = f"""
🌍🌍🌍 CRITICAL - CULTURAL ADAPTATION (MANDATORY!) 🌍🌍🌍

⚠️ YOU MUST CHANGE THE CHARACTERS AND ENVIRONMENT! ⚠️
DO NOT keep the original video's people or backgrounds!
Create NEW characters and environments for the TARGET market: {region.upper().replace('_', ' ')}

**MANDATORY CHANGES:**

1. **REPLACE CHARACTERS (DO NOT USE ORIGINAL!):**
   - Use: {cultural_style.get('ethnicity', 'diverse features appropriate for target region')}
   - Clothing: {cultural_style.get('clothing', 'modern casual fashion for target region')}
   - Names if needed: {cultural_style.get('names', 'culturally appropriate names')}
   
   Example: If original has Arab woman → For US market → Use American woman
   Example: If original has Asian man → For German market → Use German/European man

2. **REPLACE ENVIRONMENT (DO NOT USE ORIGINAL!):**
   - Use: {cultural_style.get('environment', 'settings appropriate for target country')}
   - Remove any text, signs, or architecture from the original country
   - Add environment details matching the target country

3. **CULTURAL STYLE:**
   - Tone: {cultural_style.get('style', 'confident and professional')}
   - All text in target language
   - Culturally appropriate expressions and gestures

⚠️ THIS IS MANDATORY - DO NOT SKIP! ⚠️
"""
        return instructions
    
    def _analyze_article_video_relevance(self, article_text: str, video_description: str) -> Dict[str, Any]:
        """Analyze the relevance between article content and video content.
        
        This helps determine the best blending strategy for content adaptation.
        
        Args:
            article_text: The article content.
            video_description: Description of what's shown in the video.
            
        Returns:
            Dict with relevance_score (0-1), common_themes, blend_strategy.
        """
        try:
            if not article_text or not video_description:
                return {
                    "relevance_score": 0.5,
                    "common_themes": [],
                    "blend_strategy": "video_priority",
                    "blend_instructions": "Focus on video content, use article for general context only."
                }
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at analyzing content relevance.
Compare the article content with the video description and determine:
1. How related they are (0-1 score)
2. What common themes exist
3. Best strategy to blend them

Return JSON:
{
  "relevance_score": 0.0-1.0,
  "common_themes": ["theme1", "theme2"],
  "video_subject": "what the video shows",
  "article_subject": "what the article is about",
  "blend_strategy": "full_blend" | "partial_blend" | "video_priority",
  "blend_instructions": "specific instructions for content creators"
}

STRATEGIES:
- full_blend (score > 0.7): Article and video are about the same topic. Use article content fully.
- partial_blend (score 0.3-0.7): Some overlap. Keep video visuals, adapt messaging to find common ground.
- video_priority (score < 0.3): No connection. Ignore article, focus on video content only."""
                    },
                    {
                        "role": "user",
                        "content": f"""ARTICLE CONTENT:
{article_text[:1000]}

VIDEO DESCRIPTION:
{video_description[:500]}

Analyze the relevance and provide blending strategy."""
                    }
                ],
                max_tokens=300,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"📊 Article-Video Relevance: {result.get('relevance_score', 0):.2f} - Strategy: {result.get('blend_strategy', 'unknown')}")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Could not analyze article-video relevance: {e}")
            return {
                "relevance_score": 0.5,
                "common_themes": [],
                "blend_strategy": "partial_blend",
                "blend_instructions": "Try to find common ground between video and article content."
            }
    
    def analyze_scene_frames(
        self, 
        frame_paths: List[str],
        manual_instructions: str = ""
    ) -> Dict[str, Any]:
        """Analyze scene frames and generate prompts using two separate OpenAI calls.
        
        Args:
            frame_paths: List of paths to frame images (1 per second of scene).
            manual_instructions: Optional custom instructions from user.
            
        Returns:
            Dict containing analysis, first_prompt (image), second_prompt (motion).
        """
        try:
            logger.info(f"🔍 Analyzing {len(frame_paths)} frames with OpenAI (2 calls)...")
            
            # Encode images to base64
            image_contents = []
            for frame_path in frame_paths:
                with open(frame_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high"  # Use high detail for text extraction
                        }
                    })
            
            # Call 1: Generate Image Prompt (using gpt-4o-mini)
            logger.info("📸 Generating image prompt with gpt-4o-mini...")
            image_result = self._generate_image_prompt(image_contents, manual_instructions)
            
            # Call 2: Generate Motion Prompt (using gpt-4o)
            logger.info("🎬 Generating motion prompt with gpt-4o...")
            motion_result = self._generate_motion_prompt(image_contents, manual_instructions)
            
            # Combine results
            result = {
                "analysis": image_result.get("analysis", ""),
                "text_content": image_result.get("text_content", {}),
                "first_prompt": image_result.get("first_prompt", ""),
                "second_prompt": motion_result.get("second_prompt", "")
            }
            
            logger.info("✅ Scene analysis complete (both prompts generated)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing scene: {e}")
            return {
                "analysis": "Unable to analyze scene",
                "text_content": {"exact_text": "", "language": "", "position": "", "style": ""},
                "first_prompt": "",
                "second_prompt": ""
            }
    
    # =========================================================================
    # PRODUCT DETECTION FUNCTIONS
    # =========================================================================
    
    def detect_product_in_frames(
        self, 
        frame_paths: List[str],
        min_confidence: float = 0.7,
        audio_transcript: str = "",
        video_duration: float = 0
    ) -> Dict[str, Any]:
        """Comprehensive video analysis: detect product, understand narrative, and correlate with VO.
        
        Analyzes 60 frames spread across the video + audio transcript to understand:
        - What the product IS (type, brand, visual details)
        - What the product DOES (purpose, function, how it's used)
        - How it APPEARS in different contexts (static, being applied, in-use, etc.)
        - SEQUENTIAL narrative: what happens from start to finish
        - Audio-Visual correlation: what is said when what is shown
        
        Args:
            frame_paths: List of paths to frame images (60 frames spread across video).
            min_confidence: Minimum confidence threshold (0-1).
            audio_transcript: The transcribed VO/audio from the video.
            video_duration: Total video duration in seconds.
            
        Returns:
            Dict with comprehensive video understanding including:
                - Sequential narrative breakdown
                - Product info with usage contexts
                - Audio-visual correlation
        """
        try:
            logger.info(f"🔍 [VIDEO ANALYSIS] Analyzing {len(frame_paths)} frames + audio for comprehensive understanding...")
            
            # Encode images to base64
            image_contents = []
            for frame_path in frame_paths:
                if not os.path.exists(frame_path):
                    logger.warning(f"⚠️ [PRODUCT] Frame not found: {frame_path}")
                    continue
                    
                with open(frame_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high"
                        }
                    })
            
            if not image_contents:
                logger.warning("⚠️ [PRODUCT] No valid frames to analyze")
                return {"has_product": False}
            
            # Build audio context if available
            audio_context = ""
            if audio_transcript and len(audio_transcript) > 10:
                # Calculate approximate timing per frame
                frames_count = len(frame_paths)
                seconds_per_frame = video_duration / frames_count if video_duration > 0 and frames_count > 0 else 0.5
                audio_context = f"""
=== AUDIO/VOICEOVER TRANSCRIPT ===
"{audio_transcript}"

Video duration: {video_duration:.1f} seconds
Frames analyzed: {frames_count} (1 frame every ~{seconds_per_frame:.2f} seconds)

IMPORTANT: Correlate what is SAID in the VO with what is SHOWN in frames.
Frame 0 = start of video (0:00), Frame {frames_count-1} = end of video ({video_duration:.1f}s)
=================================
"""
            
            # Enhanced system prompt for comprehensive video analysis
            system_prompt = """You are an expert video analyst specializing in advertising and commercial content.

Your task is to provide a COMPREHENSIVE ANALYSIS of the video by:
1. UNDERSTANDING THE NARRATIVE: What story does the video tell from start to end?
2. IDENTIFYING THE PRODUCT: What is being sold? How does it look exactly?
3. ANALYZING EACH SCENE: What happens in each part of the video?
4. CORRELATING AUDIO-VISUAL: What is said when what is shown?

You are viewing 60 FRAMES spread EVENLY across the entire video.
- Frame 0 = START of video
- Frame 30 = MIDDLE of video  
- Frame 59 = END of video
- Analyze them SEQUENTIALLY to understand the flow!

Examples of what to identify:
- Weight loss patch: Hook shows problem → Demo shows application → Results shown → CTA
- Dog toy: Hook shows excited dog → Problem shows bored dog → Solution shows toy → Demo shows play
- Skincare: Problem shows skin issue → Solution introduces product → Demo shows application → Results

Be EXTREMELY detailed - this analysis will be used to recreate similar videos!"""

            # Build user prompt (avoiding f-string issues with JSON template)
            frame_count = len(frame_paths)
            user_prompt_parts = [
                f"Analyze these {frame_count} video frames SEQUENTIALLY to understand the COMPLETE VIDEO.",
                audio_context,
                """
=== PART 1: SEQUENTIAL VIDEO NARRATIVE ===
Analyze frames in ORDER (0 to last) and describe what happens at each stage:
- OPENING (first 20%): What's the hook? How does the video start?
- BUILD-UP (20-40%): What problem/story is introduced?
- CORE MESSAGE (40-70%): What's the main demonstration/solution?
- RESOLUTION (70-85%): What benefits/results are shown?
- CLOSING (last 15%): What's the CTA? How does it end?

=== PART 2: ULTRA-DETAILED PRODUCT IDENTIFICATION ===
Describe the product for AI image generation:
- EXACT SHAPE: Round? Square? Curved? Dimensions?
- EXACT COLORS: Specific shades (not "blue" but "deep navy with cyan accents")
- EXACT SIZE: Compare to common objects (credit card sized, palm-sized, etc.)
- EXACT MATERIALS: Matte/glossy plastic? Fabric? Metal?
- EXACT TEXTURES: Smooth? Ridged? Perforated?
- BRANDING: Logos, text, patterns?
- PACKAGING: Colors, design, text on packaging?

=== PART 3: PRODUCT USAGE CONTEXTS ===
For each context type, identify which frames show it:
- "static_display": Product alone on surface
- "in_packaging": Product in box/wrapper
- "being_applied": Product being used/applied
- "in_hand": Held by person
- "close_up": Detail shot of product
- "before_after": Comparison shots
- "lifestyle": Product in real-life context
- "not_visible": Product not in frame

=== PART 4: AUDIO-VISUAL CORRELATION ===
Match what is SAID to what is SHOWN:
- When does the VO mention the product? What's shown then?
- When are benefits mentioned? What visuals accompany?
- When is the CTA spoken? What's on screen?

Return JSON:"""
            ]
            
            user_prompt = "\n".join(user_prompt_parts) + """
{
    "has_product": true/false,
    "product_detected": "patch/cream/device/supplement/toy/accessory/etc",
    
    "video_narrative": {
        "video_type": "product_demo/testimonial/lifestyle/before_after/tutorial/ugc",
        "opening_hook": "What happens in the first 2-3 seconds to grab attention",
        "main_story": "The core narrative/message of the video",
        "climax": "The key moment - product reveal, transformation, or benefit demonstration",
        "closing": "How the video ends - CTA, final message",
        "emotional_journey": "The emotional arc: curiosity to problem to hope to solution to action",
        "pacing": "fast/medium/slow",
        "style": "professional/ugc/influencer/cinematic/casual"
    },
    
    "sequential_breakdown": [
        {
            "segment": "opening/build_up/core/resolution/closing",
            "frame_range": [0, 10],
            "timestamp_range": "0:00-0:05",
            "what_happens": "Detailed description of what happens in this segment",
            "product_visibility": "none/glimpse/partial/full/close_up",
            "audio_content": "What is being said during this segment (from transcript)",
            "key_visuals": ["visual element 1", "visual element 2"],
            "purpose": "hook/problem/solution/demo/benefit/cta"
        }
    ],
    
    "audio_visual_sync": [
        {
            "vo_text": "The exact text being spoken",
            "frame_range": [15, 25],
            "visual_description": "What is shown while this is said",
            "sync_quality": "perfect/good/loose",
            "key_message": "The main point being communicated"
        }
    ],
    
    "product_description": "EXTREMELY DETAILED 300+ word VISUAL description for AI image generation. Include exact shape, dimensions, colors (with specific shades), materials, textures, branding, and unique features.",
    
    "product_purpose": "DETAILED explanation of what the product does, benefits, target audience, and problem it solves.",
    
    "product_usage_method": "STEP-BY-STEP usage instructions with body positioning and actions.",
    
    "product_details": {
        "type": "specific product type",
        "brand": "brand name if visible, or unbranded",
        "shape": "exact shape description",
        "dimensions": "approximate dimensions",
        "colors": {
            "primary": "main color with exact shade",
            "secondary": "secondary color",
            "accent": "accent colors",
            "packaging_colors": ["packaging colors"]
        },
        "materials": ["material descriptions"],
        "textures": ["texture descriptions"],
        "packaging": "detailed packaging description",
        "branding_elements": ["logo", "text", "patterns"],
        "distinctive_features": ["unique features"]
    },
    
    "usage_contexts": [
        {
            "context_type": "static_display/being_applied/in_hand/close_up/lifestyle/before_after",
            "description": "How product appears in this context",
            "visual_elements": "Other elements in frame",
            "action_description": "Movement/action happening",
            "frame_indices": [0, 3, 5],
            "vo_during_context": "What is said during this context"
        }
    ],
    
    "key_frames": {
        "best_product_frame": 0,
        "best_usage_frame": 0,
        "best_result_frame": 0,
        "hook_frame": 0,
        "cta_frame": 0
    },
    
    "overall_confidence": 0.0-1.0,
    
    "recreation_notes": "Key insights for recreating a similar video - what makes this video effective, what elements to preserve"
}

If NO product detected:
{
    "has_product": false,
    "product_detected": null,
    ...
}

REMEMBER: The product_description will be used DIRECTLY for image generation. Make it so detailed that an artist could draw the exact product without ever seeing it!"""

            # Build message content
            user_content = [{"type": "text", "text": user_prompt}] + image_contents
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use gpt-4o for best vision capability
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=8000,  # Increased for comprehensive video analysis with 60 frames
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content is None:
                logger.warning("⚠️ [PRODUCT] OpenAI returned None content")
                return {"has_product": False}
            
            result = json.loads(content.strip())
            
            # Check confidence threshold
            if result.get("has_product") and result.get("overall_confidence", 0) < min_confidence:
                logger.info(f"ℹ️ [PRODUCT] Product detected but confidence too low: {result.get('overall_confidence'):.2f} < {min_confidence}")
                result["has_product"] = False
            
            # Log result
            if result.get("has_product"):
                logger.info(f"✅ [PRODUCT] Detected: {result.get('product_detected')}")
                logger.info(f"   Brand: {result.get('product_details', {}).get('brand', 'unknown')}")
                logger.info(f"   Purpose: {result.get('product_purpose', 'unknown')[:100]}...")
                logger.info(f"   Confidence: {result.get('overall_confidence', 0):.2f}")
                logger.info(f"   Best frame: {result.get('best_frame_index')}")
                
                # Log usage contexts found
                usage_contexts = result.get("usage_contexts", [])
                if usage_contexts:
                    context_types = [c.get("context_type") for c in usage_contexts]
                    logger.info(f"   Usage contexts: {', '.join(context_types)}")
            else:
                logger.info("ℹ️ [PRODUCT] No product detected, continuing with standard flow")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [PRODUCT] Detection error: {e}")
            return {"has_product": False, "error": str(e)}
    
    def analyze_video_structure(
        self,
        frame_paths: List[str],
        article_content: Dict[str, str] = None,
        manual_instructions: str = "",
        product_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze the video's narrative structure and plan scene content based on article and product.
        
        This function scans the entire video to understand:
        1. The video's narrative flow (intro, problem, solution, CTA, etc.)
        2. How scenes transition and what each scene communicates
        3. How to adapt each scene to the article content
        4. Where the product should appear and how (static, being used, etc.)
        
        Args:
            frame_paths: List of frame paths from across the video.
            article_content: Dict with keys: 'free_text', 'title', 'first_paragraph', 'rest_content'
            manual_instructions: Optional manual instructions from the sheet.
            product_info: Product detection results (if available).
            
        Returns:
            Dict containing:
                - video_structure: Overall narrative structure type
                - scene_plan: List of planned scenes with content assignments
                - content_mapping: How article content maps to scenes
        """
        try:
            logger.info("📊 [STRUCTURE] Analyzing video structure with article context...")
            
            # Prepare article content
            article = article_content or {}
            free_text = article.get('free_text', '')
            title = article.get('title', '')
            first_para = article.get('first_paragraph', '')
            rest_content = article.get('rest_content', '')
            
            # Combine article content for context
            full_article = ""
            if free_text:
                full_article = free_text
            else:
                parts = [p for p in [title, first_para, rest_content] if p]
                full_article = "\n\n".join(parts)
            
            # Prepare product context
            product_context = ""
            if product_info and product_info.get("has_product"):
                product_context = f"""
PRODUCT DETECTED:
- Type: {product_info.get('product_detected', 'unknown')}
- Purpose: {product_info.get('product_purpose', 'unknown')}
- Usage method: {product_info.get('product_usage_method', 'unknown')}
- Usage contexts in video: {', '.join([c.get('context_type', '') for c in product_info.get('usage_contexts', [])])}
"""
            
            # Encode sample frames (use 5 evenly distributed)
            image_contents = []
            sample_indices = [0, len(frame_paths)//4, len(frame_paths)//2, (len(frame_paths)*3)//4, len(frame_paths)-1]
            sample_indices = list(set([min(i, len(frame_paths)-1) for i in sample_indices]))
            
            for idx in sorted(sample_indices)[:5]:
                frame_path = frame_paths[idx] if idx < len(frame_paths) else frame_paths[-1]
                if os.path.exists(frame_path):
                    with open(frame_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                        image_contents.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "low"
                            }
                        })
            
            if not image_contents:
                logger.warning("⚠️ [STRUCTURE] No frames available for analysis")
                return {"video_structure": "unknown", "scene_plan": []}
            
            # System prompt for structure analysis
            system_prompt = """You are an expert video advertising analyst. Your job is to analyze video structure and plan how to adapt it to new content.

You must understand:
1. VIDEO STRUCTURE: What type of advertising video is this? (problem-solution, testimonial, product demo, lifestyle, before-after, etc.)
2. SCENE NARRATIVE: What role does each scene play? (hook, problem statement, solution reveal, product showcase, CTA, etc.)
3. CONTENT MAPPING: How should the article content be distributed across scenes?
4. PRODUCT PLACEMENT: In which scenes should the product appear, and how?

Common video structures:
- "problem_solution": Hook → Problem → Solution (product) → Benefits → CTA
- "testimonial": Hook → Personal story → Product discovery → Transformation → CTA
- "product_demo": Hook → Product intro → Features → How to use → CTA
- "lifestyle": Aspirational scenes → Product integration → Benefits → CTA
- "before_after": Before state → Product use → After state → CTA"""

            # Build the user prompt
            user_prompt = f"""Analyze this advertising video's structure and plan content adaptation.

**ARTICLE CONTENT TO ADAPT:**
Title: {title if title else '[Not provided]'}
First Paragraph: {first_para[:500] if first_para else '[Not provided]'}
Rest of Content: {rest_content[:500] if rest_content else '[Not provided]'}
Free Text (if provided, use this instead): {free_text[:500] if free_text else '[Not provided]'}

**MANUAL INSTRUCTIONS (MUST FOLLOW):**
{manual_instructions if manual_instructions else '[No manual instructions]'}

{product_context if product_context else '**NO PRODUCT DETECTED**'}

**ANALYZE THE VIDEO FRAMES AND RETURN:**

1. Video structure type
2. For each scene (based on frames), determine:
   - Scene role in narrative (hook, problem, solution, etc.)
   - What content from article should appear (title, key benefit, CTA, etc.)
   - If product detected: should product appear here? How? (static, being_applied, in_hand, etc.)
   - Suggested visual elements

Return JSON:
{{
    "video_structure": "problem_solution" | "testimonial" | "product_demo" | "lifestyle" | "before_after" | "mixed",
    "narrative_summary": "Brief description of video's story arc",
    "scene_plan": [
        {{
            "scene_number": 1,
            "estimated_time_range": "0-3s",
            "narrative_role": "hook" | "problem" | "solution" | "benefit" | "cta" | "transition",
            "article_content_to_use": "Which part of article content fits here",
            "product_appearance": "static_display" | "being_applied" | "in_hand" | "lifestyle" | "not_visible" | null,
            "visual_suggestion": "Description of what this scene should show",
            "key_message": "The main point this scene communicates"
        }}
    ],
    "content_distribution": {{
        "title_usage": "Which scene(s) should feature the title",
        "key_benefits": ["Benefit 1 → Scene X", "Benefit 2 → Scene Y"],
        "cta_placement": "Which scene(s) for call-to-action"
    }},
    "product_integration_plan": {{
        "total_product_scenes": number,
        "primary_showcase_scene": number,
        "application_scenes": [scene numbers where product is being used],
        "lifestyle_scenes": [scene numbers with product in context]
    }}
}}"""

            # Build message content
            user_content = [{"type": "text", "text": user_prompt}] + image_contents
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=2500,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content is None:
                logger.warning("⚠️ [STRUCTURE] Analysis returned None")
                return {"video_structure": "unknown", "scene_plan": []}
            
            result = json.loads(content.strip())
            
            # Log results
            logger.info(f"✅ [STRUCTURE] Video type: {result.get('video_structure')}")
            logger.info(f"   Narrative: {result.get('narrative_summary', '')[:100]}...")
            scene_plan = result.get("scene_plan", [])
            logger.info(f"   Planned {len(scene_plan)} scenes")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [STRUCTURE] Analysis error: {e}")
            return {"video_structure": "unknown", "scene_plan": [], "error": str(e)}
    
    def analyze_video_style(
        self,
        frame_paths: List[str],
        video_duration: float = 0
    ) -> Dict[str, Any]:
        """Comprehensive video style analysis to match the original video's visual style.
        
        Analyzes the entire video to extract:
        - Color palette and grading
        - Lighting style (natural, studio, warm, cool)
        - Composition patterns (close-up, wide, etc.)
        - Camera movement tendencies
        - Overall mood and atmosphere
        - Scene transition styles
        - Subject framing preferences
        
        This creates a "style guide" used to generate videos that match the original.
        
        Args:
            frame_paths: List of frame paths from across the video.
            video_duration: Total video duration in seconds.
            
        Returns:
            Dict with comprehensive style analysis.
        """
        try:
            logger.info("🎨 [STYLE] Analyzing video visual style for matching...")
            
            # Sample frames evenly across the video for style analysis
            num_frames = min(8, len(frame_paths))
            if len(frame_paths) > num_frames:
                indices = [int(i * (len(frame_paths) - 1) / (num_frames - 1)) for i in range(num_frames)]
                sample_paths = [frame_paths[i] for i in indices]
            else:
                sample_paths = frame_paths
            
            # Encode frames
            image_contents = []
            for frame_path in sample_paths:
                try:
                    with open(frame_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                        image_contents.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}", "detail": "low"}
                        })
                except Exception as e:
                    logger.warning(f"Could not encode frame {frame_path}: {e}")
            
            if not image_contents:
                return {"error": "No frames to analyze"}
            
            system_prompt = """You are an expert cinematographer and visual style analyst.
Analyze these video frames to create a comprehensive VISUAL STYLE GUIDE that can be used to recreate videos with the EXACT SAME visual style.

Your analysis must capture every visual detail that makes this video unique, so AI-generated content will look like it belongs to the same video.

Focus on:
1. **Color Palette**: Exact dominant colors, color temperature, saturation levels, contrast
2. **Lighting**: Type (natural/studio/mixed), direction, intensity, shadows, highlights
3. **Composition**: Framing style, rule of thirds usage, negative space, subject placement
4. **Camera**: Typical angles, distances (close-up/medium/wide), movement patterns
5. **Mood/Atmosphere**: Overall feeling, energy level, professional vs casual
6. **Subjects**: How people/products are typically shown in this video
7. **Background Style**: Types of backgrounds, blur levels, environment style
8. **Quality/Finish**: Resolution feel, film grain, sharpness, professional polish level"""

            user_prompt = f"""Analyze these {len(image_contents)} frames from a video and create a DETAILED VISUAL STYLE GUIDE.

This style guide will be used to generate NEW images and videos that MUST look like they belong to the same video.

Return JSON:
{{
    "color_palette": {{
        "dominant_colors": ["#HEXCODE1", "#HEXCODE2", "..."],
        "color_temperature": "warm" | "cool" | "neutral",
        "saturation": "high" | "medium" | "low",
        "contrast": "high" | "medium" | "low",
        "color_description": "Detailed description of the color grading"
    }},
    "lighting": {{
        "type": "natural" | "studio" | "mixed" | "ambient",
        "direction": "front" | "side" | "back" | "overhead" | "mixed",
        "intensity": "bright" | "medium" | "dim" | "moody",
        "shadow_style": "soft" | "hard" | "minimal",
        "lighting_description": "Detailed description of lighting"
    }},
    "composition": {{
        "primary_framing": "close-up" | "medium" | "wide" | "extreme close-up" | "varied",
        "subject_placement": "centered" | "rule-of-thirds" | "off-center" | "varied",
        "negative_space": "minimal" | "balanced" | "abundant",
        "depth_of_field": "shallow" | "medium" | "deep",
        "composition_description": "Detailed description of composition style"
    }},
    "camera_style": {{
        "typical_angles": ["eye-level", "low-angle", "high-angle", "dutch"],
        "typical_distances": ["close-up", "medium", "wide"],
        "movement_tendency": "static" | "subtle" | "dynamic" | "handheld",
        "camera_description": "How the camera typically behaves"
    }},
    "mood_atmosphere": {{
        "overall_mood": "energetic" | "calm" | "professional" | "casual" | "intimate" | "dramatic",
        "energy_level": "high" | "medium" | "low",
        "style_category": "lifestyle" | "product-focused" | "testimonial" | "tutorial" | "artistic",
        "mood_description": "The emotional feeling of the video"
    }},
    "subject_presentation": {{
        "human_subjects": "present" | "hands-only" | "none",
        "human_style": "Description of how people appear (age, style, ethnicity, clothing)",
        "product_presentation": "in-use" | "displayed" | "both",
        "focus_subject": "product" | "person" | "balanced"
    }},
    "background_style": {{
        "environment": "indoor" | "outdoor" | "mixed" | "abstract",
        "background_type": "home" | "studio" | "nature" | "urban" | "minimal",
        "blur_level": "bokeh" | "slight" | "sharp",
        "background_description": "Typical background characteristics"
    }},
    "quality_finish": {{
        "resolution_feel": "cinematic" | "social-media" | "professional" | "amateur",
        "post_processing": "heavy" | "moderate" | "minimal" | "raw",
        "overall_polish": "highly-polished" | "natural" | "casual"
    }},
    "style_prompt_prefix": "A 50-word prompt prefix that captures the EXACT visual style to prepend to any image generation prompt",
    "style_prompt_suffix": "A 30-word prompt suffix with technical details to append to any image generation prompt",
    "motion_style_guide": "Description of how motion/animation should feel to match this video's style"
}}"""

            user_content = [{"type": "text", "text": user_prompt}] + image_contents
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=3000,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content is None:
                logger.warning("⚠️ [STYLE] Analysis returned None")
                return {}
            
            result = json.loads(content.strip())
            
            # Log key findings
            logger.info(f"✅ [STYLE] Analysis complete:")
            logger.info(f"   Color temp: {result.get('color_palette', {}).get('color_temperature', 'unknown')}")
            logger.info(f"   Lighting: {result.get('lighting', {}).get('type', 'unknown')}")
            logger.info(f"   Composition: {result.get('composition', {}).get('primary_framing', 'unknown')}")
            logger.info(f"   Mood: {result.get('mood_atmosphere', {}).get('overall_mood', 'unknown')}")
            
            # Log the style prompt that will be used
            style_prefix = result.get("style_prompt_prefix", "")
            if style_prefix:
                logger.info(f"   Style prefix: {style_prefix[:80]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [STYLE] Analysis error: {e}")
            return {"error": str(e)}
    
    def enhance_prompt_with_product(
        self,
        original_prompt: str,
        product_description: str,
        article_text: str = "",
        product_info: Dict[str, Any] = None,
        scene_context: str = None,
        video_style: Dict[str, Any] = None
    ) -> str:
        """Enhance an image generation prompt with product details, usage context, AND video style matching.
        
        This enhanced version understands:
        - HOW the product should appear in each scene (static, being used, lifestyle)
        - The VISUAL STYLE of the original video (colors, lighting, composition)
        - Generates prompts that create images matching the original video's look
        
        Args:
            original_prompt: The original image prompt from scene analysis.
            product_description: Detailed product description from detection.
            article_text: Optional new article content to adapt the scene to.
            product_info: Full product detection result with usage contexts.
            scene_context: Specific context for this scene (e.g., "being_applied", "static_display").
            video_style: Visual style analysis from analyze_video_style().
            
        Returns:
            Enhanced prompt string with product emphasis, context, and style matching.
        """
        try:
            logger.info("🎨 [PRODUCT] Enhancing prompt with product details and context...")
            
            # Extract additional context from product_info
            product_purpose = ""
            product_usage_method = ""
            usage_contexts = []
            scene_plan_info = None
            
            if product_info:
                product_purpose = product_info.get("product_purpose", "")
                product_usage_method = product_info.get("product_usage_method", "")
                usage_contexts = product_info.get("usage_contexts", [])
                scene_plan_info = product_info.get("scene_plan")  # From video structure analysis
            
            # Extract story context info if available (with type safety)
            story_context_info = product_info.get("story_context", {}) if product_info else {}
            # Ensure story_context_info is a dict, not a string
            if not isinstance(story_context_info, dict):
                story_context_info = {}
            story_type = story_context_info.get("story_type", "")
            story_summary = story_context_info.get("story_summary", "")
            scene_subject_appearance = story_context_info.get("scene_subject_appearance", "")
            has_visible_change = story_context_info.get("has_visible_change", False)
            start_state = story_context_info.get("start_state", "")
            end_state = story_context_info.get("end_state", "")
            essential_beats = story_context_info.get("essential_story_beats", [])
            must_preserve = story_context_info.get("must_preserve", [])
            
            # Scene-specific details (with type safety)
            scene_details = story_context_info.get("scene_details", {})
            if not isinstance(scene_details, dict):
                scene_details = {}
            scene_physical_state = scene_details.get("physical_state", "")
            scene_action = scene_details.get("action", "")
            scene_purpose = scene_details.get("purpose", "")
            scene_emotional_beat = scene_details.get("emotional_beat", "")
            
            # Determine the scene context from original prompt if not provided
            if not scene_context:
                # Try to infer context from original prompt
                prompt_lower = original_prompt.lower()
                if any(word in prompt_lower for word in ["apply", "applying", "putting", "placing", "stick", "press"]):
                    scene_context = "being_applied"
                elif any(word in prompt_lower for word in ["hold", "holding", "hand", "hands", "showing"]):
                    scene_context = "in_hand"
                elif any(word in prompt_lower for word in ["close", "detail", "zoom", "macro"]):
                    scene_context = "close_up"
                elif any(word in prompt_lower for word in ["before", "after", "result", "transform"]):
                    scene_context = "before_after"
                else:
                    scene_context = "static_display"
            
            # System prompt for enhancement with INTELLIGENT product usage logic
            system_prompt = """You are an expert visual director. Your job is to create prompts that show products EXACTLY as they appear in the original video, with LOGICAL real-world usage.

⚠️⚠️⚠️ TWO ABSOLUTE RULES - NEVER BREAK THESE ⚠️⚠️⚠️

RULE 1: PRODUCT MUST MATCH ORIGINAL EXACTLY
- Copy the EXACT visual description provided (color, shape, size, materials, patterns)
- Do NOT invent new designs or change the product's appearance
- The generated image must show the SAME product from the original video

RULE 2: USAGE MUST BE PHYSICALLY LOGICAL
- Think: "In real life, how would someone ACTUALLY use this product?"
- Apply common sense for each product type

========================================
ABSOLUTE LOGIC RULES (ZERO TOLERANCE):
========================================

🩹 **ADHESIVE PATCHES/STICKERS (weight loss, pain relief, etc.):**
   THEY STICK TO SKIN, NOT FABRIC!
   
   ✅ CORRECT PLACEMENT:
   - Bare stomach (shirt lifted or no shirt)
   - Bare upper arm
   - Bare thigh
   - Bare back/shoulder
   
   ❌ ABSOLUTELY FORBIDDEN:
   - On top of shirt/clothing
   - On pants/jeans
   - On shoes
   - On any fabric
   
   📝 HOW TO SHOW APPLICATION:
   - Person lifts shirt to expose bare stomach → applies patch to skin
   - Person in tank top → patch on bare shoulder
   - Close-up of bare skin with patch adhered to it

🐕 **PET PRODUCTS (toys, food, accessories):**
   THE PET MUST BE PRESENT AND INTERACTING!
   
   ✅ CORRECT:
   - Dog actively playing with/chewing the toy
   - Cat batting/chasing the toy
   - Pet eating from bowl
   
   ❌ FORBIDDEN:
   - Toy sitting alone on table/floor
   - Pet product without any pet visible

👐 **HANDHELD PRODUCTS:**
   - Natural grip, correct orientation
   - Size proportional to hand
   - Being used for its actual purpose

💍 **WEARABLES:**
   - On correct body part
   - Properly fitted, not floating

========================================
YOUR THOUGHT PROCESS:
========================================
Before writing, mentally simulate:
1. "I am holding this product. Where would I PUT it?"
2. "If this goes on skin, the skin must be VISIBLE and BARE"
3. "If this is for a pet, the PET must be VISIBLE"
4. "Does my prompt pass the 'common sense' test?"

========================================
🎬 VIDEO STORYTELLING AWARENESS:
========================================
Videos tell STORIES. Subjects may appear DIFFERENTLY in different scenes because:
- They change over time (weight loss, skin improvement, mood change)
- They're shown in different situations (before using product vs after)
- The story has an arc (problem → solution → result)

YOUR JOB: Follow the STORY CONTEXT provided for each scene.
- If told "subject appears overweight in this scene" → show them overweight
- If told "subject appears slim and confident" → show them slim and confident
- If told "subject is applying the product" → show that action

The same person CAN and SHOULD look different between scenes if the story requires it!

Keep prompt under 4000 characters."""

            # Determine product type for specific logic
            product_type = product_info.get('product_detected', 'unknown').lower() if product_info else 'unknown'
            is_patch = any(word in product_type for word in ['patch', 'sticker', 'adhesive', 'bandage'])
            is_cream = any(word in product_type for word in ['cream', 'lotion', 'gel', 'serum', 'ointment'])
            is_pet_product = any(word in product_type for word in ['dog', 'cat', 'pet', 'toy']) or (product_purpose and any(word in product_purpose.lower() for word in ['dog', 'cat', 'pet']))
            
            # Build specific warning based on product type
            product_specific_warning = ""
            if is_patch:
                product_specific_warning = """
⚠️⚠️⚠️ THIS IS AN ADHESIVE PATCH - CRITICAL RULES ⚠️⚠️⚠️
This product STICKS TO BARE SKIN. It CANNOT stick to fabric/clothing.

YOU MUST SHOW:
- BARE SKIN visible (stomach, arm, thigh, back)
- Patch applied DIRECTLY to skin surface
- If showing application: person lifts clothing to expose bare skin

YOU MUST NOT SHOW:
- Patch on top of shirt/clothing
- Patch on fabric of any kind
- Patch floating or not adhered to anything
"""
            elif is_cream:
                product_specific_warning = """
⚠️ THIS IS A CREAM/GEL - IT GOES ON BARE SKIN
Show application to visible bare skin (face, arms, body).
Do NOT show cream on clothing.
"""
            elif is_pet_product:
                product_specific_warning = """
⚠️ THIS IS A PET PRODUCT - THE PET MUST BE VISIBLE
Show a real dog/cat actively interacting with the product.
Do NOT show the product alone without the pet.
"""
            
            # Build story context instruction - DYNAMIC based on video analysis
            story_instruction = ""
            if story_context_info:
                story_instruction = f"""
🎬 VIDEO STORY CONTEXT:
Story Type: {story_type if story_type else 'commercial/advertisement'}
Story Summary: {story_summary if story_summary else 'Product advertisement'}

"""
                # Add scene-specific subject appearance if available
                if scene_subject_appearance:
                    story_instruction += f"""
⚠️⚠️⚠️ CRITICAL - SUBJECT APPEARANCE FOR THIS SCENE ⚠️⚠️⚠️
In THIS specific scene, the subject(s) MUST appear as:
{scene_subject_appearance}

This is EXACTLY how they should look - follow this description precisely!
"""
                elif scene_physical_state:
                    story_instruction += f"""
⚠️ SUBJECT STATE IN THIS SCENE:
Physical state: {scene_physical_state}
Action: {scene_action if scene_action else 'As shown in original'}
"""
                
                # Add scene purpose context
                if scene_purpose:
                    story_instruction += f"""
📍 SCENE PURPOSE: {scene_purpose}
Emotional beat: {scene_emotional_beat if scene_emotional_beat else 'Match the original mood'}
"""
                
                # If video has visible subject changes, note it
                if has_visible_change and (start_state or end_state):
                    story_instruction += f"""
📊 NOTE - Subject changes throughout video:
- Start of video: {start_state}
- End of video: {end_state}
Make sure this scene matches the CORRECT state for its position in the story!
"""
                
                # Add must-preserve elements
                if must_preserve:
                    story_instruction += f"""
🔒 MUST PRESERVE in this scene: {', '.join(must_preserve[:3])}
"""
            
            # Build the user prompt with STORY CONTEXT and LOGIC CHECK
            user_prompt = f"""Create a prompt that recreates this scene while maintaining the VIDEO'S STORY.

{story_instruction}
{product_specific_warning}

**THE PRODUCT (maintain visual consistency):**
Type: {product_type}
Visual Description: {product_description}
Purpose: {product_purpose if product_purpose else "Commercial product"}
How It's Used: {product_usage_method if product_usage_method else "Standard usage"}

**ORIGINAL SCENE TO RECREATE:**
{original_prompt}

**RECREATION RULES:**
1. SUBJECT APPEARANCE: Follow the story context above - subjects may look different in different scenes!
2. PRODUCT ACCURACY: Show product exactly as described
3. SCENE PURPOSE: This scene serves a specific purpose in the story - preserve that purpose
4. PHYSICAL LOGIC: Apply common sense (patches on bare skin, pets with pet products, etc.)

**LOGIC CHECK:**
- If it's a patch/sticker: Is it shown on BARE SKIN? If not, FIX IT.
- If it's a pet product: Is there a pet interacting? If not, ADD ONE.
- If subject appearance is specified above: Does the prompt match that appearance? If not, FIX IT.

**CREATE THE SCENE:**
Scene Context: {scene_context}

"""
            
            # Add specific context instructions WITH STRICT LOGIC
            if scene_context == "being_applied":
                if is_patch:
                    user_prompt += """
**APPLICATION SCENE FOR PATCH:**
🩹 REQUIRED: Show patch being applied to BARE SKIN
- Person lifts shirt → bare stomach visible → patch placed on bare stomach
- OR bare arm/shoulder visible → patch on bare arm
- The SKIN must be VISIBLE where the patch is placed
- ❌ NEVER show patch on clothing/shirt/fabric
"""
                elif is_pet_product:
                    user_prompt += """
**APPLICATION SCENE FOR PET PRODUCT:**
🐕 REQUIRED: Show pet actively interacting with the product
- Dog/cat must be visible in the scene
- Pet is playing with, chewing, or using the product
- ❌ NEVER show product alone without pet
"""
                else:
                    user_prompt += """
**APPLICATION SCENE:**
- Show product being used for its actual purpose
- Realistic hand/body positioning
- Logical, believable action
"""
            elif scene_context == "in_hand":
                user_prompt += """
**IN-HAND SCENE:**
- Natural hand grip, product clearly visible
- Correct scale relative to hand
"""
            elif scene_context == "static_display":
                user_prompt += """
**STATIC DISPLAY:**
- Product prominently displayed
- Clear, detailed view
"""
            elif scene_context == "close_up":
                user_prompt += """
**CLOSE-UP:**
- Detailed view of product features
- Match original product exactly
"""
            elif scene_context == "lifestyle":
                if is_patch:
                    user_prompt += """
**LIFESTYLE SCENE FOR PATCH:**
🩹 If patch is visible, it MUST be on BARE SKIN
- Person going about daily life with patch on bare stomach/arm
- Skin must be exposed where patch is shown
"""
                elif is_pet_product:
                    user_prompt += """
**LIFESTYLE SCENE FOR PET PRODUCT:**
🐕 Pet must be visible and happy with the product
"""
                else:
                    user_prompt += """
**LIFESTYLE SCENE:**
- Product in natural, everyday context
- Realistic usage scenario
"""
            
            # Add scene plan information if available (from video structure analysis)
            if scene_plan_info:
                user_prompt += f"""
**SCENE NARRATIVE ROLE:** {scene_plan_info.get('narrative_role', 'general')}
**KEY MESSAGE FOR THIS SCENE:** {scene_plan_info.get('key_message', 'Show the product')}
**VISUAL SUGGESTION:** {scene_plan_info.get('visual_suggestion', '')}
"""
            
            # Add video style matching instructions if available
            if video_style and not video_style.get("error"):
                style_prefix = video_style.get("style_prompt_prefix", "")
                style_suffix = video_style.get("style_prompt_suffix", "")
                color_info = video_style.get("color_palette", {})
                lighting_info = video_style.get("lighting", {})
                composition_info = video_style.get("composition", {})
                mood_info = video_style.get("mood_atmosphere", {})
                
                user_prompt += f"""

🎨 **CRITICAL: MATCH ORIGINAL VIDEO STYLE** 🎨
The generated image MUST match the visual style of the original video:

**COLOR STYLE:**
- Temperature: {color_info.get('color_temperature', 'neutral')}
- Saturation: {color_info.get('saturation', 'medium')}
- Contrast: {color_info.get('contrast', 'medium')}
- {color_info.get('color_description', '')}

**LIGHTING:**
- Type: {lighting_info.get('type', 'natural')}
- Direction: {lighting_info.get('direction', 'front')}
- Intensity: {lighting_info.get('intensity', 'medium')}
- {lighting_info.get('lighting_description', '')}

**COMPOSITION:**
- Framing: {composition_info.get('primary_framing', 'medium')}
- Subject placement: {composition_info.get('subject_placement', 'centered')}
- Depth of field: {composition_info.get('depth_of_field', 'medium')}
- {composition_info.get('composition_description', '')}

**MOOD:**
- Overall: {mood_info.get('overall_mood', 'professional')}
- Energy: {mood_info.get('energy_level', 'medium')}

**USE THIS STYLE PREFIX:** {style_prefix}
**USE THIS STYLE SUFFIX:** {style_suffix}

INCORPORATE these style elements into your enhanced prompt!
"""
            
            if article_text:
                article_summary = article_text[:800] + "..." if len(article_text) > 800 else article_text
                user_prompt += f"""
**NEW CONTEXT/ARTICLE TO ADAPT TO:**
{article_summary}
"""
            
            user_prompt += """
**FINAL INSTRUCTIONS:**
Generate a prompt that:
1. Shows LOGICAL, REALISTIC product usage (fix any illogical placements)
2. Keeps the product's EXACT visual appearance (colors, shape, materials)
3. Places product in correct context for {scene_context}
4. Makes physical and common sense

⚠️ REALITY CHECK before output:
- "Is this how a real person would use this product?" 
- "Does this placement make physical sense?"
- If NO → FIX IT to be realistic

OUTPUT ONLY the corrected, realistic prompt text.""".format(scene_context=scene_context)

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            if content is None:
                logger.warning("⚠️ [PRODUCT] Enhancement returned None, using original prompt")
                return original_prompt
            
            enhanced_prompt = content.strip()
            
            # Extract size and usage details from product_info
            size_info = ""
            usage_action = ""
            if product_info:
                details = product_info.get("product_details", {})
                if details:
                    dims = details.get("dimensions", "")
                    shape = details.get("shape", "")
                    if dims or shape:
                        size_info = f"SIZE: {shape}, {dims}"
                
                usage_method = product_info.get("product_usage_method", "")
                if usage_method:
                    usage_action = f"USAGE ACTION: {usage_method[:200]}"
            
            # Wrap with emphasis for image generator - with SIZE and USAGE emphasis
            final_prompt = f"""[CRITICAL - PRODUCT SIZE AND APPEARANCE MUST BE EXACT]
PRODUCT VISUAL: {product_description[:350]}
{size_info}
SCENE TYPE: {scene_context}
{usage_action if scene_context == "being_applied" else ""}

{enhanced_prompt}

[IMPORTANT: Product must be shown at CORRECT PROPORTIONAL SIZE relative to hands/body. If being applied, show the EXACT application action described above.]"""
            
            # Truncate if needed (Nano Banana limit is 4000 chars)
            if len(final_prompt) > 4000:
                final_prompt = final_prompt[:3997] + "..."
            
            logger.info(f"✅ [PRODUCT] Prompt enhanced ({len(final_prompt)} chars) - context: {scene_context}")
            return final_prompt
            
        except Exception as e:
            logger.error(f"❌ [PRODUCT] Enhancement error: {e}")
            # Return original prompt with basic product emphasis
            return f"[Product: {product_description[:200]}] {original_prompt}"
    
    def enhance_motion_prompt_with_product(
        self,
        original_motion_prompt: str,
        product_info: Dict[str, Any],
        scene_context: str = None,
        video_style: Dict[str, Any] = None
    ) -> str:
        """Enhance motion/animation prompt to accurately show product usage AND match video style.
        
        This ensures the video animation correctly shows:
        - How the product is being applied/used
        - The correct size and proportions in motion
        - The proper action sequence
        - Camera movements matching the original video's style
        
        Args:
            original_motion_prompt: The original motion prompt.
            product_info: Product detection results with usage details.
            scene_context: How the product appears in this scene.
            video_style: Visual style analysis from analyze_video_style().
            
        Returns:
            Enhanced motion prompt with accurate product usage and style matching.
        """
        if not product_info or not product_info.get("has_product"):
            return original_motion_prompt
        
        try:
            # Extract product details
            product_type = product_info.get("product_detected", "product")
            usage_method = product_info.get("product_usage_method", "")
            product_details = product_info.get("product_details", {})
            
            # Get size info
            size_info = ""
            if product_details:
                shape = product_details.get("shape", "")
                dims = product_details.get("dimensions", "")
                if shape or dims:
                    size_info = f"{shape}, {dims}"
            
            # Build context-specific motion instructions
            motion_instruction = ""
            
            if scene_context == "being_applied":
                # Describe the application action with ABSOLUTE LOGIC
                is_patch_product = "patch" in product_type.lower() or "sticker" in product_type.lower() or "adhesive" in product_type.lower()
                is_cream_product = "cream" in product_type.lower() or "lotion" in product_type.lower() or "gel" in product_type.lower()
                is_pet_toy = "toy" in product_type.lower() and ("dog" in product_type.lower() or "pet" in product_type.lower() or (usage_method and any(pet in usage_method.lower() for pet in ["dog", "cat", "pet"])))
                
                if is_patch_product:
                    motion_instruction = f"""MOTION: Patch application to BARE SKIN

⚠️⚠️⚠️ ABSOLUTE RULE: PATCH GOES ON BARE SKIN, NOT CLOTHING ⚠️⚠️⚠️

REQUIRED SEQUENCE:
1. Hands holding {product_type} ({size_info})
2. Person LIFTS SHIRT to expose BARE STOMACH (or bare arm/thigh visible)
3. Hands move patch toward BARE SKIN surface
4. Press patch onto BARE SKIN with gentle pressure
5. Smooth edges onto BARE SKIN
6. Patch is now ADHERED TO SKIN, not floating

❌ FORBIDDEN: Patch touching any fabric/clothing
✅ REQUIRED: Visible bare skin where patch is applied

Camera: Close-up showing bare skin clearly"""
                    
                elif is_cream_product:
                    motion_instruction = f"""MOTION: Cream application to BARE SKIN
1. Dispense product onto fingertips
2. Apply to BARE SKIN (face/arms/body - skin must be visible)
3. Gentle massage motion
Camera: Focus on bare skin and application"""
                    
                elif is_pet_toy:
                    motion_instruction = f"""MOTION: Dog/Pet playing with toy

⚠️ REQUIRED: A DOG/PET MUST BE VISIBLE AND INTERACTING ⚠️

SEQUENCE:
1. Dog sees the {product_type} ({size_info})
2. Dog excitedly approaches/grabs the toy
3. Dog plays - tugging, chewing, shaking
4. Joyful pet interaction throughout
5. Toy and dog move together dynamically

❌ FORBIDDEN: Toy alone without pet
✅ REQUIRED: Happy dog actively playing with toy

Camera: Follow dog and toy interaction"""
                    
                else:
                    motion_instruction = f"""MOTION: Product in realistic use
1. Product ({product_type}) held/used naturally
2. Show actual intended purpose
3. Logical, believable movement
USAGE: {usage_method[:150] if usage_method else 'Standard usage'}"""
                    
            elif scene_context == "in_hand":
                motion_instruction = f"""MOTION: Product showcase in hand:
1. Hand holding {product_type} ({size_info}) - product fills frame appropriately
2. Slight rotation or movement to show product details
3. Stable, professional presentation
Camera: Focus on product, slight movement for dynamism"""
                
            elif scene_context == "static_display":
                motion_instruction = f"""MOTION: Static product beauty shot:
1. {product_type} ({size_info}) displayed prominently
2. Subtle camera movement (slow zoom or pan)
3. Product remains centered and sharp
Camera: Smooth, cinematic movement around product"""
                
            elif scene_context == "lifestyle":
                motion_instruction = f"""MOTION: Lifestyle scene with product:
1. Natural environment movement
2. {product_type} visible and in-scale with surroundings
3. Organic camera movement
USAGE: {usage_method[:100] if usage_method else 'Product in natural context'}"""
            
            else:
                motion_instruction = f"""MOTION: Show {product_type} ({size_info}):
- Product clearly visible and correctly sized
- Smooth, professional camera movement
{f'USAGE: {usage_method[:100]}' if usage_method else ''}"""
            
            # Add video style matching if available
            style_motion_guide = ""
            if video_style and not video_style.get("error"):
                camera_style = video_style.get("camera_style", {})
                mood = video_style.get("mood_atmosphere", {})
                motion_guide = video_style.get("motion_style_guide", "")
                
                style_motion_guide = f"""
CAMERA STYLE TO MATCH:
- Movement: {camera_style.get('movement_tendency', 'subtle')}
- Typical angles: {', '.join(camera_style.get('typical_angles', ['eye-level']))}
- Energy: {mood.get('energy_level', 'medium')}
- {motion_guide if motion_guide else ''}
"""
            
            # Combine with original prompt
            enhanced_motion = f"""{motion_instruction}

ORIGINAL SCENE: {original_motion_prompt}
{style_motion_guide}
[CRITICAL: Product must be CORRECT SIZE relative to hands/body. {product_type} is {size_info}]"""
            
            # Truncate if too long
            if len(enhanced_motion) > 2500:
                enhanced_motion = enhanced_motion[:2497] + "..."
            
            logger.info(f"✅ [PRODUCT] Motion prompt enhanced for {scene_context}")
            return enhanced_motion
            
        except Exception as e:
            logger.error(f"❌ [PRODUCT] Motion enhancement error: {e}")
            return original_motion_prompt
    
    def _generate_image_prompt(
        self, 
        image_contents: List[Dict],
        manual_instructions: str = ""
    ) -> Dict[str, Any]:
        """Generate image recreation prompt using gpt-4o-mini.
        
        Args:
            image_contents: List of base64 encoded images.
            manual_instructions: Optional custom instructions.
            
        Returns:
            Dict with analysis, text_content, and first_prompt.
        """
        try:
            # Comprehensive analysis prompt for image recreation
            analysis_prompt = """Analyze this image in comprehensive detail. You must extract and describe ALL of the following:

**1. TEXT CONTENT (CRITICAL - TRANSCRIBE EXACTLY):**
- Transcribe ALL text visible in the image EXACTLY as written, preserving:
  - The exact wording (letter by letter)
  - The language it's written in
  - Line breaks and text positioning
  - Font style (bold, italic, etc.)
  - Text color
  - Text size (relative - large headline, medium body, small caption)
  - Text placement on the image (top, center, bottom, left, right)

**2. UI ELEMENTS (BUTTONS, LABELS, BADGES):**
- Describe any buttons: shape, color, text on button, position
- Describe any labels, badges, or tags
- Describe any call-to-action elements
- Note the exact text on each UI element

**3. VISUAL STYLE:**
- Overall style (photographic, illustrated, etc.)
- Color palette and mood
- Background description
- Lighting and atmosphere

**4. LAYOUT AND COMPOSITION:**
- How elements are arranged
- Text overlay positioning relative to background
- Any graphic elements (shapes, lines, icons)

**5. DESIGN ELEMENTS:**
- Any overlays, gradients, or effects on the background
- Shadow or glow effects on text
- Border or frame elements

**6. MOTION ANALYSIS (from frame sequence):**
- Identify camera movement (pan, zoom, tilt, static)
- Subject movement or animation
- Any transitions or effects between frames

**7. BACKGROUND:**
- Elements (buildings, nature, interior, abstract, etc.)
- Weather conditions (sunny, cloudy, rainy, foggy, night, etc.)
- Dominant colors and color scheme

**8. PEOPLE/CHARACTERS:**
- Estimated age range
- Gender
- Hair color and style
- Clothing description (type, color, style)
- Body position and movement

**9. OBJECTS:**
- Type of objects visible
- Purpose/use of each object
- Color of objects
- Size (small, medium, large, relative to frame)

**10. CAMERA:**
- Camera angle (eye-level, low angle, high angle, bird's eye, Dutch angle)
- Camera lens type (wide-angle, telephoto, macro, fisheye, standard)"""

            # Base system prompt for image recreation
            base_system_prompt = """You are an expert video content analyst and prompt engineer specializing in image recreation.

Based on your detailed analysis of the video frames, generate a text-to-image prompt to recreate the starting frame.

**FLEXIBLE GUIDELINES FOR first_prompt (adapt based on user instructions):**
The following are DEFAULT guidelines. User instructions may ask you to EXCLUDE or MODIFY certain elements. 
Always follow user instructions - they take priority over these defaults:

DEFAULT elements to include (unless user says otherwise):
- Text/typography if present (exact wording, positioning, font style, colors)
- Background and visual style
- UI elements (buttons, labels, badges)
- People/characters with accurate details
- Objects with their positions
- Camera angle and perspective

**CRITICAL RULES:**
1. If user instructions say to EXCLUDE something (e.g., "no text", "remove UI elements"), you MUST NOT include it in the first_prompt
2. Be CONSISTENT - apply user instructions to ALL aspects of your output
3. The first_prompt MUST be under 4000 characters maximum
4. Still do your analysis in the 'analysis' field, but the 'first_prompt' must respect user exclusions

Return your response as JSON with these keys:
- analysis: Your detailed analysis following the categories from the user prompt (for reference)
- text_content: {exact_text: string, language: string, position: string, style: string} (can be empty if user excludes text)
- first_prompt: Complete prompt to recreate the starting frame (for text-to-image) - MAX 4000 characters, respecting user exclusions"""

            # Prepend user's manual instructions to system prompt if provided
            if manual_instructions:
                system_prompt = f"""**🚨 USER INSTRUCTIONS (HIGHEST PRIORITY - MUST FOLLOW):**
{manual_instructions}

These instructions OVERRIDE any conflicting default guidelines below. Apply them consistently to your entire output, especially the 'first_prompt'.

---

{base_system_prompt}"""
            else:
                system_prompt = base_system_prompt

            # Build user content
            instruction_text = "Analyze these frames from a video scene and generate an image recreation prompt:"
            
            user_content = [
                {"type": "text", "text": instruction_text},
                {"type": "text", "text": analysis_prompt}
            ] + image_contents
            
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_completion_tokens=128000,  # gpt-5-mini supports up to 128k output tokens
                #temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Check for None content (can happen with API issues or content filtering)
            content = response.choices[0].message.content
            if content is None:
                logger.warning("⚠️ OpenAI returned None content for image prompt")
                return {"analysis": "", "text_content": {}, "first_prompt": ""}
            
            result_text = content.strip()
            result = json.loads(result_text)
            
            logger.info("✅ Image prompt generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating image prompt: {e}")
            return {
                "analysis": "",
                "text_content": {},
                "first_prompt": ""
            }
    
    def _generate_motion_prompt(
        self, 
        image_contents: List[Dict],
        manual_instructions: str = ""
    ) -> Dict[str, Any]:
        """Generate motion/animation prompt using gpt-4o.
        
        Args:
            image_contents: List of base64 encoded images.
            manual_instructions: Optional custom instructions.
            
        Returns:
            Dict with second_prompt (motion prompt).
        """
        try:
            # Motion-focused analysis prompt
            motion_analysis_prompt = """Analyze these video frames to understand the motion and animation.

Focus ONLY on:
1. Camera movement (pan left/right, tilt up/down, zoom in/out, dolly, crane, static)
2. Subject movement (walking, running, gesturing, facial expressions)
3. Object movement (falling, flying, rotating, scaling)
4. Speed and timing of movements
5. Direction of movement"""

            # System prompt for motion generation - camera-first, no hallucinations
            system_prompt = """You are an expert in video motion prompts for image-to-video AI (Runway/Kling).

**CRITICAL RULES - NO HALLUCINATIONS:**
- The motion prompt describes ONLY how the EXISTING image moves. Do NOT add new subjects, objects, actions, or story elements that are not clearly in the image.
- **Only describe motion for body parts or objects that are explicitly visible in the image.** If the image does not show hands, do NOT say "hand movement" or "hand move". If it does not show a bag, do NOT say "nudging the bag". Only mention movement of elements that appear in the image prompt.
- The generated video must look like the same scene in the image, just with motion. Nothing should appear, disappear, or change identity.
- Prefer subtle, natural motion over dramatic or creative motion to avoid weird AI artifacts.

**STRUCTURE OF second_prompt (in this order):**
1. **Camera movement first** (required): e.g. "Slow zoom in", "Subtle pan right", "Static shot", "Gentle dolly forward", "Slight tilt down".
2. **Then** (optional, only if that element is in the image): very brief subject motion - e.g. "subtle smile" only if face is visible, "slight head turn" only if head is in frame, "gentle hand movement" only if hands are visible in the image. Do NOT mention hands, arms, or objects that are not described in the image.
3. Do NOT invent new actions, new people, or new objects. Only describe motion that fits exactly what is already in the image.

**Keep second_prompt under 200 characters.** Camera movement is the most reliable; extra description can cause artifacts.

Return your response as JSON with this key:
- second_prompt: Motion prompt (camera first, then minimal subtle motion, no new elements)"""

            # Build user content
            instruction_text = "Analyze the motion in these frames and generate a Runway motion prompt:"
            if manual_instructions:
                instruction_text += f"\n\n**SPECIAL INSTRUCTIONS FROM USER:**\n{manual_instructions}"
            
            user_content = [
                {"type": "text", "text": instruction_text},
                {"type": "text", "text": motion_analysis_prompt}
            ] + image_contents
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=2000,  # Motion prompts are shorter
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Check for None content (can happen with API issues or content filtering)
            content = response.choices[0].message.content
            if content is None:
                logger.warning("⚠️ OpenAI returned None content for motion prompt")
                return {"second_prompt": ""}
            
            result_text = content.strip()
            result = json.loads(result_text)
            
            logger.info("✅ Motion prompt generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating motion prompt: {e}")
            return {
                "second_prompt": ""
            }
    
    def analyze_full_video(
        self,
        frame_paths_with_timestamps: List[Tuple[float, str]],
        pyscenedetect_timestamps: List[float],
        video_duration: float,
        manual_instructions: str = "",
        cta_button: bool = False,
        cta_text: str = "",
        row_num: int = 0,
        article_text: str = "",
        vertical: str = "",
        article_language: str = "",
        article_related_to_video: bool = True
    ) -> Dict[str, Any]:
        """Analyze entire video and generate scene timestamps + prompts in a single call.
        
        This unified approach sends all frames to OpenAI, which then:
        1. Validates/corrects PySceneDetect scene times
        2. Generates image prompts for each scene
        3. Generates motion prompts for each scene
        
        If article content is provided, prompts will be adapted to match the article's
        topic, location, characters, and cultural context.
        
        Args:
            frame_paths_with_timestamps: List of (timestamp, frame_path) tuples for entire video.
            pyscenedetect_timestamps: Initial scene start times from PySceneDetect.
            video_duration: Total video duration in seconds.
            manual_instructions: Optional custom instructions from user.
            cta_button: Whether to include a CTA button in image prompts.
            cta_text: Text for the CTA button.
            row_num: Row number for logging purposes.
            article_text: Optional article content to adapt prompts to.
            vertical: Optional vertical/offer name for content adaptation.
            article_language: Optional language code for content adaptation.
            article_related_to_video: True if article is similar to video (adapt), False if different (create new).
            
        Returns:
            Dict with corrected_scenes and scene_prompts.
        """
        row_prefix = f"[Row {row_num}] " if row_num > 0 else ""
        try:
            logger.info(f"🎬 {row_prefix}Analyzing full video with OpenAI (unified call)...")
            logger.info(f"   {row_prefix}Frames: {len(frame_paths_with_timestamps)}")
            logger.info(f"   {row_prefix}PySceneDetect scenes: {len(pyscenedetect_timestamps)}")
            logger.info(f"   {row_prefix}Video duration: {video_duration:.2f}s")
            
            # Encode images to base64 with timestamp labels
            image_contents = []
            for timestamp, frame_path in frame_paths_with_timestamps:
                try:
                    with open(frame_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                        # Add timestamp label before each image
                        image_contents.append({
                            "type": "text",
                            "text": f"[Frame at {timestamp:.1f}s]"
                        })
                        image_contents.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high"
                            }
                        })
                except Exception as e:
                    logger.warning(f"⚠️ Could not read frame at {timestamp:.1f}s: {e}")
            
            if not image_contents:
                logger.error("❌ No frames could be loaded")
                return self._empty_video_analysis_result(pyscenedetect_timestamps, video_duration)
            
            # Format PySceneDetect timestamps for the prompt
            pyscene_info = "PySceneDetect detected scene changes at these timestamps:\n"
            for i, ts in enumerate(pyscenedetect_timestamps):
                if i + 1 < len(pyscenedetect_timestamps):
                    end_ts = pyscenedetect_timestamps[i + 1]
                else:
                    end_ts = video_duration
                duration = end_ts - ts
                pyscene_info += f"  Scene {i+1}: {ts:.2f}s - {end_ts:.2f}s (duration: {duration:.2f}s)\n"
            
            # System prompt for unified video analysis
            system_prompt = f"""You are an expert video analyst and prompt engineer. You will analyze video frames and generate scene-based prompts.

**VIDEO INFO:**
- Total duration: {video_duration:.2f} seconds
- Frames provided: 1 per second (with timestamps)

**YOUR TASKS:**

1. **VALIDATE/CORRECT SCENE TIMESTAMPS:**
   Review the PySceneDetect timestamps against the actual frames. Adjust scene boundaries if needed.
   
   RULES:
   - Each scene MUST be between {config.PYSCENEDETECT_MIN_SCENE_DURATION}-{config.PYSCENEDETECT_MAX_SCENE_DURATION} seconds
   - Scenes should align with actual visual/content changes in the frames
   - **IMPORTANT: As long as it's the same character(s) on screen, it's the same scene** - Don't split a scene just because of camera angle changes or minor visual transitions if the same person/character remains the focus
   - A new scene starts when: the main character changes, the location completely changes, or there's a clear cut to different content
   - The first scene always starts at 0.0s
   - The last scene ends at {video_duration:.2f}s
   - If a scene is too long (>{config.PYSCENEDETECT_MAX_SCENE_DURATION}s), split it at a natural point
   - If a scene is too short (<{config.PYSCENEDETECT_MIN_SCENE_DURATION}s), merge it with adjacent scene
   - Maximum {config.MAX_SCENES} scenes total

2. **FOR EACH SCENE, ANALYZE THESE CATEGORIES IN DETAIL:**

   **VISUAL STYLE:**
   - Overall style (photographic, illustrated, cinematic, animated, 3D rendered, etc.)
   - Color palette and mood (warm, cool, vibrant, muted, etc.)
   - Lighting type and atmosphere (natural, studio, dramatic, soft, harsh, etc.)

   **BACKGROUND:**
   - Environment elements (buildings, nature, interior, abstract, urban, rural, etc.)
   - Weather/environment conditions (sunny, cloudy, rainy, foggy, night, day, etc.)
   - Dominant colors and color scheme

   **PEOPLE/CHARACTERS:**
   - Gender (male, female, non-binary, unclear)
   - Estimated age range (child, teen, young adult, middle-aged, elderly)
   - Ethnicity/skin tone (for accurate recreation)
   - Hair color, length, and style
   - Facial features and expressions
   - Clothing description (type, color, style, brand if visible)
   - Body position, pose, and movement
   - Number of people in frame

   **OBJECTS:**
   - Type of objects visible
   - Purpose/use of each object
   - Color, texture, and material of objects
   - Size (small, medium, large, relative to frame)
   - Position in frame

   **VISUAL CONTENT:**
   - Focus on describing the VISUAL CONTENT: people, places, objects, environments
   - Describe what you SEE in the frame: colors, shapes, lighting, mood, composition

   **CAMERA:**
   - Camera angle (eye-level, low angle, high angle, bird's eye, Dutch angle, worm's eye)
   - Camera lens type (wide-angle, telephoto, macro, fisheye, standard, anamorphic)
   - Depth of field (shallow/bokeh, deep focus)
   - Framing (close-up, medium shot, wide shot, extreme close-up, full body)

   **MOTION (from frame sequence):**
   - Camera movement type (pan left/right, tilt up/down, zoom in/out, dolly, truck, crane, handheld, static)
   - Speed of camera movement (slow, medium, fast)
   - Subject movement or animation
   - Direction of movement
   - Any effects or transitions visible

3. **FOR EACH SCENE, GENERATE:**

   **STORY COHERENCE (CRITICAL):**
   - Each scene serves a specific purpose in the narrative: hook → problem → agitation → solution_intro → demo → result → social_proof → cta.
   - The image_prompt MUST include the emotional state/action that fits the scene's story beat (e.g., frustration for "problem", relief for "result", excitement for "hook").
   - Scenes are NOT isolated - they must flow logically. Each scene connects to the previous one (why it follows) and leads to the next (transition_logic).
   - The story must feel natural and coherent when the scenes are played in sequence.

   a) **image_prompt** (for text-to-image generation):
      - Describe the first frame of the scene in comprehensive detail
      - Include ALL of the following:
        * Visual style, colors, lighting, and atmosphere
        * Background environment with specific details
        * People/characters with gender, age, ethnicity, hair, clothing, pose, and EMOTIONAL STATE matching the story beat
        * Objects with their positions, colors, and sizes
        * Camera angle, lens type, and framing
      - Make it detailed enough to recreate the image faithfully
      
      **LOGICAL & HUMAN (reduce hallucinations):**
      - Every scene must be LOGICAL and COHERENT: no surreal, impossible, or inconsistent elements.
      - People must look HUMAN and NATURAL: realistic poses, natural expressions, believable lighting and proportions.
      - When a product is visible, ALWAYS include a short verbal description of how it looks (shape, color, size, placement, and what it DOES) in the prompt, in addition to any reference - this keeps the model consistent.
      - Scenes should feel dynamic but believable; avoid exaggerated or artificial descriptions that cause AI artifacts.
      
      **YOUR IMAGE PROMPT MUST:**
      - Describe the visual scene: people, places, objects, environments, backgrounds
      - Include visual details: colors, shapes, lighting, mood, composition, camera angles
      - Include the EMOTIONAL STATE that matches this scene's story beat
      - Focus on what IS visible in the scene: natural visual elements, subjects, settings
      
      - Maximum 4000 characters
   
   b) **visible_elements** (inventory of what is in the image):
      - List every body part, person, object, and item visible in the scene (e.g. "face", "hands", "full body", "product", "table", "bag").
      - This list is the STRICT CONTRACT for what the motion_prompt can reference.

   c) **motion_prompt** (for image-to-video generation - MUST match the image, no hallucinations):
      - **Start with camera movement** (required): e.g. "Slow zoom in", "Subtle pan right", "Static shot", "Gentle dolly forward". Type: pan, tilt, zoom, dolly, truck, crane; direction and speed (slow, gradual).
      - **ONLY describe motion for items listed in visible_elements.** If "hands" is NOT in visible_elements, do NOT say "hand movement". If "bag" is NOT in visible_elements, do NOT say "nudging the bag". This is a strict rule.
      - **Then** optionally add only subtle motion that fits EXACTLY what is in visible_elements: e.g. "subtle smile" only if "face" is in visible_elements, "slight head turn" only if "head" or "face" is in visible_elements. Do NOT add new actions, people, or objects.
      - The video must be the SAME scene as the image with motion only. Nothing new may appear. Prefer conservative, natural motion to avoid weird AI artifacts.
      - If people are in the image: brief natural micro-movements only for what is in visible_elements (e.g. subtle smile, blink, slight nod only if "face"/"head" is listed). Do not invent dramatic expressions, gestures, or limb movement if those limbs are not in visible_elements.
      - If the image has text overlays, treat them as static (do not animate text).
      - Keep under 200 characters. Camera first, then minimal description.

**RETURN FORMAT (JSON):**
{{
  "corrected_scenes": [
    {{"scene_num": 1, "start": 0.0, "end": 3.5, "reason": "Original timing was accurate"}},
    {{"scene_num": 2, "start": 3.5, "end": 7.2, "reason": "Adjusted end to match visual change"}},
    ...
  ],
  "scene_prompts": [
    {{"scene_num": 1, "visible_elements": ["face", "upper body", "product in hand"], "story_beat": "hook", "transition_logic": "excited face -> next scene reveals the problem", "image_prompt": "...", "motion_prompt": "..."}},
    {{"scene_num": 2, "visible_elements": ["full body", "table", "coffee cup"], "story_beat": "problem", "transition_logic": "frustration shown -> next scene introduces solution", "image_prompt": "...", "motion_prompt": "..."}},
    ...
  ]
}}

**VISIBLE_ELEMENTS RULES:**
- List every body part, person, object, and item that is VISIBLE in the scene image (e.g. "face", "hands", "legs", "product", "table", "bag").
- The motion_prompt may ONLY reference items from visible_elements. If "hands" is not listed, motion_prompt MUST NOT mention hand movement.
- This is a STRICT CONTRACT: visible_elements is the single source of truth for what can move in the motion_prompt."""

            # NOTE: CTA button is now handled separately via overlay, not embedded in prompts
            
            # Add article adaptation instructions if provided
            if article_text:
                article_summary = article_text[:2000]  # Limit article length in prompt
                # Note: vertical_info removed - we only use article CONTENT, not metadata
                language_info = f"TARGET LANGUAGE: {article_language}" if article_language else ""
                
                if article_related_to_video:
                    # YES - Article is SIMILAR to video content
                    article_section = f"""
**🔗 ARTICLE-VIDEO RELATIONSHIP: SIMILAR CONTENT (Article IS related to Video)**
═══════════════════════════════════════════════════════════════════════════════

{language_info}

ARTICLE CONTENT:
{article_summary}

✅ ADAPTATION STRATEGY (SIMILAR CONTENT):
The article describes a SIMILAR offer/product to what's shown in the video.
Adapt the video for the new offer while keeping visuals SIMILAR to the original:

1. **KEEP THE SAME VISUAL STYLE** - The video's scenes, composition, and style should remain similar
2. **ADAPT THE PRODUCT/OFFER** - Replace the original product with the article's product (similar type)
3. **ADAPT THE MESSAGING** - Update text overlays and voiceover to match the article content
4. **ADAPT THE LANGUAGE** - ALL text must be in: {article_language}
5. **KEEP THE NARRATIVE STRUCTURE** - Same story flow (hook, problem, solution, CTA)

**🎭 PEOPLE & CULTURE (ALWAYS APPLY):**
{self._get_cultural_style_instructions(article_language)}

**In your image prompts:**
- Describe people with appropriate ethnicity for the target region
- Use culturally appropriate clothing and fashion
- Include environment details that match the target culture

---

"""
                else:
                    # NO - Article is FUNDAMENTALLY DIFFERENT from video content
                    article_section = f"""
**🔄 ARTICLE-VIDEO RELATIONSHIP: DIFFERENT CONTENT (Article is NOT related to Video)**
═══════════════════════════════════════════════════════════════════════════════════════

{language_info}

ARTICLE CONTENT (NEW TOPIC):
{article_summary}

⚠️⚠️⚠️ CRITICAL ADAPTATION STRATEGY (DIFFERENT CONTENT) ⚠️⚠️⚠️
The article describes a COMPLETELY DIFFERENT offer/product than what's shown in the video.
You must CREATE NEW content while KEEPING the video's STYLE and ATMOSPHERE:

1. **EXTRACT THE VIDEO'S STYLE** - Analyze: lighting, camera work, mood, energy, color palette
2. **KEEP THE VISUAL STYLE** - New video should FEEL like the original (same quality, mood, pacing)
3. **DO NOT USE THE ORIGINAL PRODUCT/OFFER** - The original video's product is IRRELEVANT
4. **CREATE NEW VISUALS FOR THE ARTICLE** - All scenes must be appropriate for the article's topic
5. **ALL TEXT AND VO IN TARGET LANGUAGE** - {article_language}

🎯 YOUR MISSION:
- Create prompts for a NEW video that LOOKS LIKE the original (same style/mood)
- But SHOWS content appropriate for the ARTICLE (NOT the original video's product)
- Features people, settings, and actions relevant to the ARTICLE
- Uses the same production quality and pacing as the original

Example: Original video = shoe advertisement → Article = work-from-home jobs
→ Keep: professional style, energetic mood, production quality
→ Create: scenes showing people working from home, home office setups
→ Do NOT: show shoes, running, sports themes

**🎭 PEOPLE & CULTURE (ALWAYS APPLY):**
{self._get_cultural_style_instructions(article_language)}

---

"""
                system_prompt = article_section + system_prompt
            
            # Add manual instructions if provided
            if manual_instructions:
                system_prompt = f"""**🚨 USER INSTRUCTIONS (HIGHEST PRIORITY - MUST FOLLOW):**
{manual_instructions}

Apply these instructions to ALL scene prompts consistently.

---

{system_prompt}"""

            # Build user content
            user_content = [
                {"type": "text", "text": pyscene_info},
                {"type": "text", "text": "\nHere are the video frames (1 per second):"},
            ] + image_contents
            
            logger.info(f"📡 {row_prefix}Sending unified request to gpt-5-mini...")
            
            import time as _time
            start_time = _time.time()
            
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_completion_tokens=128000,
                response_format={"type": "json_object"}
            )
            
            elapsed = _time.time() - start_time
            logger.info(f"✅ {row_prefix}OpenAI responded in {elapsed:.1f}s")
            
            # Check for None content
            content = response.choices[0].message.content
            if content is None:
                logger.warning(f"⚠️ {row_prefix}OpenAI returned None content")
                return self._empty_video_analysis_result(pyscenedetect_timestamps, video_duration)
            
            result = json.loads(content.strip())
            
            # Validate and log results
            corrected_scenes = result.get("corrected_scenes", [])
            scene_prompts = result.get("scene_prompts", [])
            
            logger.info(f"✅ {row_prefix}OpenAI analysis complete:")
            logger.info(f"   {row_prefix}Corrected scenes: {len(corrected_scenes)}")
            for scene in corrected_scenes:
                logger.info(f"     {row_prefix}Scene {scene.get('scene_num')}: {scene.get('start'):.2f}s - {scene.get('end'):.2f}s")
            logger.info(f"   {row_prefix}Prompts generated: {len(scene_prompts)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {row_prefix}Error in unified video analysis: {e}")
            logger.error(f"   {row_prefix}Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"   {row_prefix}Traceback:\n{traceback.format_exc()}")
            return self._empty_video_analysis_result(pyscenedetect_timestamps, video_duration)
    
    def _empty_video_analysis_result(
        self, 
        pyscenedetect_timestamps: List[float], 
        video_duration: float
    ) -> Dict[str, Any]:
        """Create empty/fallback result using original PySceneDetect timestamps."""
        corrected_scenes = []
        scene_prompts = []
        
        for i, ts in enumerate(pyscenedetect_timestamps):
            if i + 1 < len(pyscenedetect_timestamps):
                end_ts = pyscenedetect_timestamps[i + 1]
            else:
                end_ts = video_duration
            
            corrected_scenes.append({
                "scene_num": i + 1,
                "start": ts,
                "end": end_ts,
                "reason": "Fallback - using original PySceneDetect timing"
            })
            scene_prompts.append({
                "scene_num": i + 1,
                "image_prompt": "",
                "motion_prompt": ""
            })
        
        return {
            "corrected_scenes": corrected_scenes,
            "scene_prompts": scene_prompts
        }
    
    def generate_music_description(self, scene_prompts: List[Dict]) -> str:
        """Generate a music description based on video content analysis.
        
        Uses OpenAI to analyze the video scenes and describe appropriate
        background music that matches the video's mood, style, and content.
        
        Args:
            scene_prompts: List of scene prompts with image descriptions.
            
        Returns:
            A detailed music style description for Suno generation.
        """
        try:
            # Build context from scene prompts
            scenes_summary = "\n".join([
                f"Scene {i+1}: {sp.get('image_prompt', '')[:300]}"
                for i, sp in enumerate(scene_prompts[:6])  # First 6 scenes
            ])
            
            if not scenes_summary.strip():
                logger.warning("⚠️ No scene prompts available for music description")
                return "upbeat corporate background music, modern electronic, professional, energetic, inspirational, no vocals"
            
            prompt = f"""Based on this video content, describe the perfect background music for it.

VIDEO SCENES:
{scenes_summary}

Generate a detailed music description that includes:
1. Genre/style (e.g., corporate, upbeat electronic, ambient, cinematic, pop, indie, lo-fi)
2. Tempo (fast/medium/slow, approximate BPM)
3. Mood/emotion (energetic, calm, inspiring, dramatic, playful, sophisticated, warm)
4. Key instruments to feature (synths, piano, guitar, drums, strings, brass, etc.)
5. Overall vibe that matches the video content and would work as background music

The music MUST be:
- INSTRUMENTAL (absolutely no vocals or singing)
- Suitable as background music (not overpowering)
- Matching the energy and mood of the video scenes

Respond with ONLY the music description in 2-3 sentences, nothing else. Be specific and creative."""

            logger.info("🎵 Generating dynamic music description with OpenAI...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional music director who describes background music for videos. You always describe instrumental music only, no vocals."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250
            )
            
            description = response.choices[0].message.content.strip()
            logger.info(f"🎵 Generated music description: {description}")
            return description
            
        except Exception as e:
            logger.warning(f"⚠️ Could not generate music description: {e}")
            return "upbeat corporate background music, modern electronic synths, professional and energetic, inspirational mood, no vocals"
    
    def generate_music_description_from_text(self, content_text: str) -> str:
        """Generate a music description based on content text (for influencer mode).
        
        Args:
            content_text: The free text content describing the product/experience.
            
        Returns:
            A detailed music style description for Suno generation.
        """
        try:
            prompt = f"""Based on this content, describe the perfect background music for an influencer recommendation video.

CONTENT:
{content_text[:1500]}

Generate a detailed music description that includes:
1. Genre/style (e.g., upbeat pop, modern electronic, trendy, lifestyle, energetic)
2. Tempo (fast/medium/slow, appropriate for social media content)
3. Mood/emotion (exciting, positive, aspirational, fun, trendy)
4. Key instruments to feature (synths, drums, bass, guitar, etc.)
5. Overall vibe that would work for an influencer-style recommendation video

The music MUST be:
- INSTRUMENTAL (absolutely no vocals or singing)
- Suitable as background music (not overpowering)
- Trendy and engaging for social media
- Matching the energy of an enthusiastic recommendation

Respond with ONLY the music description in 2-3 sentences, nothing else. Be specific and creative."""

            logger.info("🎵 Generating music description for influencer mode...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional music director for social media content. You describe trendy, engaging background music for influencer videos. Always instrumental only, no vocals."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250
            )
            
            description = response.choices[0].message.content.strip()
            logger.info(f"🎵 Generated music description: {description}")
            return description
            
        except Exception as e:
            logger.warning(f"⚠️ Could not generate music description: {e}")
            return "upbeat trendy electronic music, modern synths with punchy drums, energetic and positive vibe, social media style, no vocals"
    
    def generate_opening_text(
        self, 
        article_text: str, 
        language: str = "en",
        video_description: str = None
    ) -> Optional[str]:
        """Generate a short, compelling opening text based on VIDEO content with cultural adaptation.
        
        Creates a brief, attention-grabbing headline that matches what's shown in the video
        AND is culturally appropriate for the target region/language.
        
        Args:
            article_text: Article content for context.
            language: Target language for the text.
            video_description: Description of what's shown in the video (from scene analysis).
            
        Returns:
            Short opening text string, or None if failed.
        """
        try:
            logger.info(f"📝 Generating opening text (language: {language})...")
            
            language_names = {
                # Major World Languages
                "en": "English", "de": "German", "es": "Spanish", "fr": "French",
                "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
                "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
                # European Languages
                "pl": "Polish", "hu": "Hungarian", "cs": "Czech", "sk": "Slovak",
                "ro": "Romanian", "bg": "Bulgarian", "uk": "Ukrainian", "hr": "Croatian",
                "sr": "Serbian", "sl": "Slovenian", "bs": "Bosnian", "mk": "Macedonian",
                "sq": "Albanian", "el": "Greek", "tr": "Turkish", "lt": "Lithuanian",
                "lv": "Latvian", "et": "Estonian", "fi": "Finnish", "sv": "Swedish",
                "no": "Norwegian", "da": "Danish", "is": "Icelandic", "ga": "Irish",
                "cy": "Welsh", "mt": "Maltese", "ca": "Catalan", "eu": "Basque",
                "gl": "Galician", "be": "Belarusian", "ka": "Georgian", "hy": "Armenian",
                "az": "Azerbaijani", "kk": "Kazakh", "uz": "Uzbek", "tg": "Tajik",
                "ky": "Kyrgyz", "tk": "Turkmen", "mn": "Mongolian",
                # Middle Eastern & South Asian Languages
                "he": "Hebrew", "fa": "Persian", "ur": "Urdu", "hi": "Hindi",
                "bn": "Bengali", "pa": "Punjabi", "gu": "Gujarati", "mr": "Marathi",
                "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
                "si": "Sinhala", "ne": "Nepali", "ps": "Pashto", "ku": "Kurdish",
                # Southeast Asian Languages
                "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
                "tl": "Filipino", "my": "Burmese", "km": "Khmer", "lo": "Lao",
                # African Languages
                "sw": "Swahili", "am": "Amharic", "ha": "Hausa", "yo": "Yoruba",
                "ig": "Igbo", "zu": "Zulu", "xh": "Xhosa", "af": "Afrikaans",
                # Regional Variants
                "pt-BR": "Brazilian Portuguese", "zh-CN": "Simplified Chinese",
                "zh-TW": "Traditional Chinese", "en-US": "American English",
                "en-GB": "British English", "es-MX": "Mexican Spanish",
                "fr-CA": "Canadian French"
            }
            lang_name = language_names.get(language, "English")
            
            # Get cultural region and hook style from config
            region = config.REGION_MAPPING.get(language, 'namer')
            hook_style = config.HOOK_STYLES.get(region, 'aspirational messaging, personal success')
            cultural_info = config.CULTURAL_STYLES.get(region, {})
            style_description = cultural_info.get('style', 'confident, aspirational')
            
            logger.info(f"   Region: {region}, Hook style: {hook_style[:50]}...")
            
            # Build context - prioritize video content
            context_parts = []
            if video_description:
                context_parts.append(f"VIDEO CONTENT (MOST IMPORTANT - text must match this!):\n{video_description}")
            if article_text:
                context_parts.append(f"Article context:\n{article_text[:300]}")
            
            context = "\n\n".join(context_parts) if context_parts else "General promotional content"
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert copywriter specializing in {lang_name} marketing content.
Generate a short, compelling opening headline in {lang_name}.

CULTURAL ADAPTATION (VERY IMPORTANT):
- Target language: {lang_name}
- Cultural region: {region}
- Preferred hook style for this region: {hook_style}
- Cultural tone: {style_description}

The headline should feel NATIVE to {lang_name} speakers - use idioms, expressions, and emotional triggers that resonate with this culture.

CRITICAL RULES:
1. The text MUST match what's shown in the VIDEO (not just the article)
2. 3-6 words maximum
3. Attention-grabbing and suitable for a video intro overlay
4. If video shows workers/jobs → text about THAT job type
5. If video shows products → text about THOSE products
6. If video shows a location/activity → text about THAT
7. DO NOT use generic text that doesn't match the video visuals
8. Use culturally appropriate hook style: {hook_style}
9. The text MUST be in {lang_name} - no English unless target is English"""
                    },
                    {
                        "role": "user",
                        "content": f"Generate a short opening headline (3-6 words only) in {lang_name} that MATCHES the video content and uses a {hook_style} approach:\n\n{context}"
                    }
                ],
                max_tokens=50,
                temperature=0.7
            )
            
            text = response.choices[0].message.content.strip()
            # Remove quotes if present
            text = text.strip('"\'')
            logger.info(f"✅ Generated opening text ({region} style): '{text}'")
            return text
            
        except Exception as e:
            logger.warning(f"⚠️ Could not generate opening text: {e}")
            return None
    
    def generate_vo_script_from_article(
        self,
        article_text: str,
        vertical: str,
        target_duration: float,
        target_language: str = "en",
        original_vo_transcript: str = None,
        scene_prompts: List[Dict] = None,
        gemini_vo_recommendations: Dict = None
    ) -> str:
        """Generate a new voice-over script based on article content and scene visuals.
        
        Creates a script suitable for TTS that matches the visuals AND content.
        The VO MUST match what's shown in the generated images.
        
        Args:
            article_text: Full article text content.
            vertical: The vertical/offer name (headline).
            target_duration: Target duration in seconds for the VO.
            target_language: ISO 639-1 language code for the script.
            original_vo_transcript: Optional original VO transcript for reference.
            scene_prompts: Optional list of scene prompts (image_prompt) to match VO with visuals.
            gemini_vo_recommendations: Optional Gemini analysis with VO recommendations including:
                - audio_analysis: voiceover_style, voiceover_tone, selling_approach
                - recommended_new_vo: style_to_match, tone_to_match, key_messages, avoid, structure
                - scene_breakdown: per-scene recommended_vo_for_new_video
            
        Returns:
            Voice-over script text suitable for TTS.
        """
        try:
            logger.info(f"📝 Generating VO script from article (target: {target_duration:.1f}s, language: {target_language})...")
            
            # Estimate word count based on duration
            # Average speaking rate: ~150 words per minute (2.5 words/second)
            target_words = int(target_duration * 2.5)
            
            # Language name mapping for the prompt
            language_names = {
                # Major World Languages
                "en": "English", "de": "German", "es": "Spanish", "fr": "French",
                "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
                "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
                # European Languages
                "pl": "Polish", "hu": "Hungarian", "cs": "Czech", "sk": "Slovak",
                "ro": "Romanian", "bg": "Bulgarian", "uk": "Ukrainian", "hr": "Croatian",
                "sr": "Serbian", "sl": "Slovenian", "bs": "Bosnian", "mk": "Macedonian",
                "sq": "Albanian", "el": "Greek", "tr": "Turkish", "lt": "Lithuanian",
                "lv": "Latvian", "et": "Estonian", "fi": "Finnish", "sv": "Swedish",
                "no": "Norwegian", "da": "Danish", "is": "Icelandic", "ga": "Irish",
                "cy": "Welsh", "mt": "Maltese", "ca": "Catalan", "eu": "Basque",
                "gl": "Galician", "be": "Belarusian", "ka": "Georgian", "hy": "Armenian",
                "az": "Azerbaijani", "kk": "Kazakh", "uz": "Uzbek", "tg": "Tajik",
                "ky": "Kyrgyz", "tk": "Turkmen", "mn": "Mongolian",
                # Middle Eastern & South Asian Languages
                "he": "Hebrew", "fa": "Persian", "ur": "Urdu", "hi": "Hindi",
                "bn": "Bengali", "pa": "Punjabi", "gu": "Gujarati", "mr": "Marathi",
                "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
                "si": "Sinhala", "ne": "Nepali", "ps": "Pashto", "ku": "Kurdish",
                # Southeast Asian Languages
                "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
                "tl": "Filipino", "my": "Burmese", "km": "Khmer", "lo": "Lao",
                # African Languages
                "sw": "Swahili", "am": "Amharic", "ha": "Hausa", "yo": "Yoruba",
                "ig": "Igbo", "zu": "Zulu", "xh": "Xhosa", "af": "Afrikaans",
                # Regional Variants
                "pt-BR": "Brazilian Portuguese", "zh-CN": "Simplified Chinese",
                "zh-TW": "Traditional Chinese", "en-US": "American English",
                "en-GB": "British English", "es-MX": "Mexican Spanish",
                "fr-CA": "Canadian French"
            }
            language_name = language_names.get(target_language, "English")
            
            # Build the prompt
            reference_section = ""
            if original_vo_transcript:
                reference_section = f"""
ORIGINAL VIDEO VOICE-OVER (for style reference only):
{original_vo_transcript[:500]}

Use the STRUCTURE and STYLE of the original VO as inspiration.
"""
            
            # Build scene visuals section from scene prompts
            visuals_section = ""
            if scene_prompts:
                visuals_section = """
🎬 VIDEO SCENES (THE VISUALS THE VIEWER WILL SEE):
The VO you write MUST match these visuals EXACTLY. The viewer will see these scenes while hearing your script.

"""
                for i, prompt in enumerate(scene_prompts[:8], 1):  # Max 8 scenes
                    image_prompt = prompt.get("image_prompt", "")[:300]
                    if image_prompt:
                        visuals_section += f"Scene {i}: {image_prompt}\n\n"
                
                visuals_section += """
🚨 ABSOLUTE RULE - VO MUST MATCH WHAT'S SHOWN!
- The scenes above show what the viewer will SEE
- Your VO must talk about EXACTLY what's shown
- If scenes show garbage collection worker → VO talks about waste collection jobs
- If scenes show office work → VO talks about office careers
- If scenes show delivery driver → VO talks about delivery/logistics jobs
- NEVER write VO about Topic A when the video shows Topic B!
- The article text is just REFERENCE - if it doesn't match the video, IGNORE IT!

Example of WRONG: Video shows nurse in hospital, VO talks about "babysitting jobs" ❌
Example of RIGHT: Video shows nurse in hospital, VO talks about "healthcare careers" ✅
"""
            
            # Determine style guidance based on original VO and Gemini recommendations
            style_guidance = ""
            
            # Add Gemini recommendations if available
            gemini_guidance = ""
            if gemini_vo_recommendations:
                audio_analysis = gemini_vo_recommendations.get("audio_analysis", {})
                recommended_vo = gemini_vo_recommendations.get("recommended_new_vo", {})
                
                if audio_analysis or recommended_vo:
                    gemini_guidance = """
🎯 AI ANALYSIS OF ORIGINAL VIDEO:
"""
                    if audio_analysis.get("voiceover_style"):
                        gemini_guidance += f"- VO Style: {audio_analysis.get('voiceover_style')}\n"
                    if audio_analysis.get("voiceover_tone"):
                        gemini_guidance += f"- VO Tone: {audio_analysis.get('voiceover_tone')}\n"
                    if audio_analysis.get("selling_approach"):
                        gemini_guidance += f"- Selling Approach: {audio_analysis.get('selling_approach')}\n"
                    if audio_analysis.get("speaking_pace"):
                        gemini_guidance += f"- Speaking Pace: {audio_analysis.get('speaking_pace')}\n"
                    if audio_analysis.get("key_phrases"):
                        key_phrases = audio_analysis.get("key_phrases", [])[:5]
                        if key_phrases:
                            gemini_guidance += f"- Key Phrases to Include: {', '.join(key_phrases)}\n"
                    
                    if recommended_vo:
                        gemini_guidance += "\n📋 RECOMMENDED NEW VO STRUCTURE:\n"
                        if recommended_vo.get("style_to_match"):
                            gemini_guidance += f"- Style: {recommended_vo.get('style_to_match')}\n"
                        if recommended_vo.get("tone_to_match"):
                            gemini_guidance += f"- Tone: {recommended_vo.get('tone_to_match')}\n"
                        if recommended_vo.get("key_messages_to_include"):
                            messages = recommended_vo.get("key_messages_to_include", [])[:3]
                            if messages:
                                gemini_guidance += f"- Key Messages: {', '.join(messages)}\n"
                        if recommended_vo.get("avoid"):
                            avoid_list = recommended_vo.get("avoid", [])[:3]
                            if avoid_list:
                                gemini_guidance += f"- AVOID: {', '.join(avoid_list)}\n"
                        if recommended_vo.get("suggested_structure"):
                            gemini_guidance += f"- Structure: {recommended_vo.get('suggested_structure')}\n"
                
                # Add per-scene VO recommendations
                scene_breakdown = gemini_vo_recommendations.get("scene_breakdown", [])
                if scene_breakdown:
                    gemini_guidance += "\n🎬 PER-SCENE VO SUGGESTIONS:\n"
                    for scene in scene_breakdown[:6]:
                        scene_num = scene.get("scene_number", "?")
                        rec_vo = scene.get("recommended_vo_for_new_video", "")
                        if rec_vo:
                            gemini_guidance += f"Scene {scene_num}: {rec_vo[:100]}...\n"
            
            if original_vo_transcript and len(original_vo_transcript) > 20:
                style_guidance = f"""
🎬 ORIGINAL VIDEO VO (MATCH THIS STYLE EXACTLY):
"{original_vo_transcript[:800]}"
{gemini_guidance}

ANALYZE THE ORIGINAL VO AND MATCH:
- Tone: Is it energetic? Calm? Urgent? Friendly? Professional?
- Structure: How does it flow? Hook → Story → CTA? Question → Answer?
- Pacing: Short punchy sentences? Longer flowing narrative?
- Voice: First person "I"? Second person "You"? Third person narrator?
- Language style: Casual? Formal? Conversational? Dramatic?

Your new VO MUST feel like it belongs to the SAME video. 
Same energy. Same rhythm. Same vibe. Just new content.
"""
            else:
                # No original transcript, but we might have Gemini guidance
                style_guidance = f"""
🎬 NO ORIGINAL VO DETECTED - USE AI-ANALYZED STYLE:
{gemini_guidance if gemini_guidance else '''
- Professional, engaging product advertisement tone
- Direct and benefit-focused
- Clear call-to-action at the end
'''}
"""
            
            prompt = f"""Create a voice-over script that matches the style of the original video.

{style_guidance}

{visuals_section}

CONTENT TO ADAPT (from article):
{article_text[:1500]}

YOUR TASK:
1. If original VO exists: MATCH its exact style, tone, energy, and structure
2. Adapt the CONTENT from the article above
3. Keep same length: ~{target_words} words ({target_duration:.0f} seconds)
4. Language: {language_name}
5. The VO must feel natural with the video visuals

IMPORTANT:
- Match the original's energy and pacing
- If original uses character names, you can too
- If original is casual, be casual
- If original is dramatic, be dramatic
- NO brackets or stage directions
- ONLY output the spoken words

Generate the voice-over:"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are an expert voice-over writer who can perfectly match any style. When given an original VO, you analyze its tone, rhythm, and structure and create new content that feels like it belongs to the same video. You write in {language_name} and your scripts sound natural when spoken."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            script = response.choices[0].message.content.strip()
            
            # Clean up any brackets or stage directions that might have slipped through
            script = re.sub(r'\[.*?\]', '', script)
            script = re.sub(r'\(.*?\)', '', script)
            script = script.strip()
            
            word_count = len(script.split())
            logger.info(f"✅ Generated VO script: {word_count} words (target: {target_words})")
            
            return script
            
        except Exception as e:
            logger.error(f"❌ Failed to generate VO script: {e}")
            # Return a simple fallback (don't use vertical - it's just metadata)
            return "Discover something amazing today. Click to learn more."

    def generate_influencer_prompts(
        self,
        free_text: str,
        reference_images: List[Dict[str, Any]],
        scene_count: int,
        manual_instructions: str = "",
        cta_text: str = "",
        language: str = "en"
    ) -> Dict[str, Any]:
        """Generate influencer-style video prompts for each scene.
        
        Creates prompts for an influencer recommendation video where:
        - Scene 1: Influencer with strong hook
        - Scene 4, 7, 10...: Influencer appears again (identical appearance)
        - Last scene: Influencer with CTA
        - Other scenes: Product/experience with cycling reference images
        
        Args:
            free_text: Content describing the product/experience to promote.
            reference_images: List of dicts with 'url' and optional 'base64' and 'analysis'.
            scene_count: Number of scenes to generate.
            manual_instructions: Optional custom instructions.
            cta_text: Call-to-action text for the last scene.
            language: ISO 639-1 language code.
            
        Returns:
            Dict with 'influencer_description', 'scene_prompts' list.
        """
        try:
            logger.info(f"🎭 Generating influencer prompts for {scene_count} scenes...")
            
            # Language name mapping
            language_names = {
                # Major World Languages
                "en": "English", "de": "German", "es": "Spanish", "fr": "French",
                "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
                "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
                # European Languages
                "pl": "Polish", "hu": "Hungarian", "cs": "Czech", "sk": "Slovak",
                "ro": "Romanian", "bg": "Bulgarian", "uk": "Ukrainian", "hr": "Croatian",
                "sr": "Serbian", "sl": "Slovenian", "bs": "Bosnian", "mk": "Macedonian",
                "sq": "Albanian", "el": "Greek", "tr": "Turkish", "lt": "Lithuanian",
                "lv": "Latvian", "et": "Estonian", "fi": "Finnish", "sv": "Swedish",
                "no": "Norwegian", "da": "Danish", "is": "Icelandic", "ga": "Irish",
                "cy": "Welsh", "mt": "Maltese", "ca": "Catalan", "eu": "Basque",
                "gl": "Galician", "be": "Belarusian", "ka": "Georgian", "hy": "Armenian",
                "az": "Azerbaijani", "kk": "Kazakh", "uz": "Uzbek", "tg": "Tajik",
                "ky": "Kyrgyz", "tk": "Turkmen", "mn": "Mongolian",
                # Middle Eastern & South Asian Languages
                "he": "Hebrew", "fa": "Persian", "ur": "Urdu", "hi": "Hindi",
                "bn": "Bengali", "pa": "Punjabi", "gu": "Gujarati", "mr": "Marathi",
                "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
                "si": "Sinhala", "ne": "Nepali", "ps": "Pashto", "ku": "Kurdish",
                # Southeast Asian Languages
                "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
                "tl": "Filipino", "my": "Burmese", "km": "Khmer", "lo": "Lao",
                # African Languages
                "sw": "Swahili", "am": "Amharic", "ha": "Hausa", "yo": "Yoruba",
                "ig": "Igbo", "zu": "Zulu", "xh": "Xhosa", "af": "Afrikaans",
                # Regional Variants
                "pt-BR": "Brazilian Portuguese", "zh-CN": "Simplified Chinese",
                "zh-TW": "Traditional Chinese", "en-US": "American English",
                "en-GB": "British English", "es-MX": "Mexican Spanish",
                "fr-CA": "Canadian French"
            }
            language_name = language_names.get(language, "English")
            
            # Determine which scenes show the influencer
            # Pattern: 1, 4, 7, 10... (every 3rd starting from 1) and always the last scene
            influencer_scenes = [1]
            for i in range(4, scene_count + 1, 3):
                if i not in influencer_scenes:
                    influencer_scenes.append(i)
            if scene_count not in influencer_scenes:
                influencer_scenes.append(scene_count)
            influencer_scenes = sorted(set(influencer_scenes))
            
            # Build reference image descriptions
            ref_image_descriptions = []
            for i, img in enumerate(reference_images):
                if img.get('analysis'):
                    ref_image_descriptions.append(f"Reference Image {i+1}: {img['analysis'][:500]}")
            
            ref_images_text = "\n".join(ref_image_descriptions) if ref_image_descriptions else "No reference images provided."
            
            # Build the prompt
            system_prompt = f"""You are an expert video content strategist who creates compelling influencer recommendation videos.
You will generate detailed image prompts for AI image generation (Nano Banana) and motion prompts for video generation (Runway).

CRITICAL IMAGE STYLE REQUIREMENTS (first_prompt):
- ALL images MUST be HYPER-REALISTIC and look like REAL PHOTOGRAPHS taken by a professional photographer
- ALWAYS start every first_prompt with: "Ultra photorealistic professional photograph, shot on Canon EOS R5, 85mm lens, natural studio lighting"
- Describe REAL skin textures, natural imperfections, genuine facial features
- Avoid any cartoonish, illustrated, AI-generated, or overly polished/synthetic looking styles
- Images should look INDISTINGUISHABLE from real photos
- Include realistic details: natural hair strands, skin pores, fabric textures, environmental reflections
- Use natural lighting setups common in professional photography

CRITICAL MOTION/ANIMATION REQUIREMENTS (second_prompt) - NO HALLUCINATIONS:
- ALWAYS start with CAMERA MOVEMENT: slow zoom in, dolly forward, subtle pan, static shot, gentle crane. This is the most reliable and avoids weird artifacts.
- Only describe motion for body parts or objects that appear IN THE IMAGE (first_prompt). If the image does not show hands, do NOT say "hand movement". If it does not show a bag or object, do NOT say "nudging the bag". Match the image exactly.
- Then optionally add only SUBTLE motion that matches what is visible: e.g. "subtle smile" only if face is in frame, "slight head turn" only if head is in frame. Do NOT add new actions, people, or story elements.
- The video must be the SAME scene as the image with motion only. Keep motions subtle and natural - avoid exaggerated or creative motion that can cause AI artifacts.

IMPORTANT RULES:
1. Generate prompts in {language_name} context but write the prompts themselves in English (for AI generation).
2. Create a DETAILED and CONSISTENT influencer appearance that must be IDENTICAL in all scenes where the influencer appears.
4. The influencer should be a relatable, attractive woman in her late 20s to early 30s with NATURAL, REALISTIC features.
5. All scenes should be vertical format (9:16 aspect ratio).
6. Use cinematic lighting and high-quality REALISTIC photography style."""

            # Manual instructions addition
            manual_section = ""
            if manual_instructions:
                manual_section = f"\n\nADDITIONAL INSTRUCTIONS FROM USER:\n{manual_instructions}"
            
            # CTA section
            cta_section = ""
            if cta_text:
                cta_section = f"\n\nCTA TEXT (for conceptual reference in last scene): {cta_text}"
            
            user_prompt = f"""Generate prompts for a {scene_count}-scene influencer recommendation video.

PRODUCT/EXPERIENCE TO PROMOTE:
{free_text[:3000]}

REFERENCE IMAGES FOR PRODUCT/EXPERIENCE:
{ref_images_text}

SCENE STRUCTURE:
- Scenes where INFLUENCER appears: {influencer_scenes}
- Other scenes: Show the product/experience only (influencer may be holding item or in location but not focal point)
- Scene 1: Start with a STRONG HOOK - influencer should show excitement/surprise
- Last scene (Scene {scene_count}): Include visual CTA concept (pointing, gesturing to click, etc.){manual_section}{cta_section}

FOR EACH SCENE, PROVIDE:
1. **visible_elements**: List of every body part, person, object visible in the scene (e.g. ["face", "hands", "product", "table"]). This is the strict contract for what second_prompt can reference.
2. **first_prompt**: Ultra-realistic photograph description. Start with "Ultra photorealistic professional photograph, shot on Canon EOS R5, 85mm lens". Describe the scene as if describing a real photo.
3. **second_prompt**: Motion prompt for image-to-video. MUST start with camera movement (e.g. Slow zoom in, Subtle pan right, Static shot). Then ONLY describe motion for items in visible_elements - if "hands" not listed do NOT say hand movement. Keep under 200 chars. Example: visible_elements=["face"] -> "Slow zoom in, subtle smile".
4. **story_beat**: One of: hook, problem, agitation, solution_intro, demo, result, social_proof, cta.
5. **transition_logic**: How this scene connects to the next scene (e.g. "close-up on excited face -> next scene reveals the product").

RESPONSE FORMAT (JSON):
{{
    "influencer_description": "DETAILED description of the influencer's appearance (face, hair, body type, skin tone, distinctive features) - this MUST be used identically in all influencer scenes",
    "scene_prompts": [
        {{
            "scene_number": 1,
            "shows_influencer": true,
            "reference_image_index": null,
            "visible_elements": ["face", "upper body", "hands"],
            "first_prompt": "detailed image prompt here",
            "second_prompt": "motion/animation prompt here",
            "story_beat": "hook",
            "transition_logic": "excited face reveal -> next scene shows product"
        }},
        ...
    ]
}}

Generate {scene_count} scenes now:"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            # Validate and enhance prompts with influencer description
            influencer_desc = result.get("influencer_description", "")
            scene_prompts = result.get("scene_prompts", [])
            
            # Ensure influencer description is embedded in all influencer scenes
            for prompt in scene_prompts:
                if prompt.get("shows_influencer", False) and influencer_desc:
                    # Prepend influencer description to first_prompt
                    original_prompt = prompt.get("first_prompt", "")
                    prompt["first_prompt"] = f"INFLUENCER: {influencer_desc}. SCENE: {original_prompt}"
            
            logger.info(f"✅ Generated {len(scene_prompts)} influencer scene prompts")
            return {
                "influencer_description": influencer_desc,
                "scene_prompts": scene_prompts
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate influencer prompts: {e}")
            return {
                "influencer_description": "",
                "scene_prompts": []
            }

    def generate_influencer_vo_script(
        self,
        free_text: str,
        scene_count: int,
        target_duration: float,
        manual_instructions: str = "",
        language: str = "en",
        original_vo_transcript: str = ""
    ) -> str:
        """Generate a first-person voice-over script for an influencer video.
        
        Creates a compelling VO script that matches the original video's style.
        If original VO exists, the new VO will match its tone, rhythm and energy.
        
        Args:
            free_text: Content describing the product/experience.
            scene_count: Number of scenes for pacing reference.
            target_duration: Target duration in seconds.
            manual_instructions: Optional custom instructions.
            language: ISO 639-1 language code.
            original_vo_transcript: The original video's VO for style matching.
            
        Returns:
            Voice-over script text suitable for TTS.
        """
        try:
            logger.info(f"🎤 Generating influencer VO script (target: {target_duration:.1f}s, language: {language})...")
            
            # Estimate word count based on duration
            target_words = int(target_duration * 2.5)
            
            # Language name mapping
            language_names = {
                # Major World Languages
                "en": "English", "de": "German", "es": "Spanish", "fr": "French",
                "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
                "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
                # European Languages
                "pl": "Polish", "hu": "Hungarian", "cs": "Czech", "sk": "Slovak",
                "ro": "Romanian", "bg": "Bulgarian", "uk": "Ukrainian", "hr": "Croatian",
                "sr": "Serbian", "sl": "Slovenian", "bs": "Bosnian", "mk": "Macedonian",
                "sq": "Albanian", "el": "Greek", "tr": "Turkish", "lt": "Lithuanian",
                "lv": "Latvian", "et": "Estonian", "fi": "Finnish", "sv": "Swedish",
                "no": "Norwegian", "da": "Danish", "is": "Icelandic", "ga": "Irish",
                "cy": "Welsh", "mt": "Maltese", "ca": "Catalan", "eu": "Basque",
                "gl": "Galician", "be": "Belarusian", "ka": "Georgian", "hy": "Armenian",
                "az": "Azerbaijani", "kk": "Kazakh", "uz": "Uzbek", "tg": "Tajik",
                "ky": "Kyrgyz", "tk": "Turkmen", "mn": "Mongolian",
                # Middle Eastern & South Asian Languages
                "he": "Hebrew", "fa": "Persian", "ur": "Urdu", "hi": "Hindi",
                "bn": "Bengali", "pa": "Punjabi", "gu": "Gujarati", "mr": "Marathi",
                "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
                "si": "Sinhala", "ne": "Nepali", "ps": "Pashto", "ku": "Kurdish",
                # Southeast Asian Languages
                "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
                "tl": "Filipino", "my": "Burmese", "km": "Khmer", "lo": "Lao",
                # African Languages
                "sw": "Swahili", "am": "Amharic", "ha": "Hausa", "yo": "Yoruba",
                "ig": "Igbo", "zu": "Zulu", "xh": "Xhosa", "af": "Afrikaans",
                # Regional Variants
                "pt-BR": "Brazilian Portuguese", "zh-CN": "Simplified Chinese",
                "zh-TW": "Traditional Chinese", "en-US": "American English",
                "en-GB": "British English", "es-MX": "Mexican Spanish",
                "fr-CA": "Canadian French"
            }
            language_name = language_names.get(language, "English")
            
            # Determine style based on original VO if available
            style_guidance = ""
            if original_vo_transcript and len(original_vo_transcript) > 20:
                style_guidance = f"""
🎬 ORIGINAL VIDEO VO - MATCH THIS STYLE EXACTLY:
"{original_vo_transcript[:1000]}"

ANALYZE AND MATCH:
- Tone: Casual? Professional? Excited? Calm? Urgent?
- Voice: First person "I"? Second person "You"? Narrator style?
- Energy: High energy hype? Chill recommendation? Dramatic reveal?
- Pacing: Quick punchy lines? Flowing narrative? Emotional build-up?
- Language: Slang and casual? Formal? Conversational?
- Structure: How does it open? How does it close? Story arc?

Your new VO must feel like the SAME person in the SAME video.
Match the vibe exactly. Just new content.
"""
            else:
                style_guidance = """
🎬 NO ORIGINAL VO - USE AUTHENTIC INFLUENCER STYLE:
- Genuine, relatable voice
- As if personally recommending to a friend
- Natural flow, conversational
"""
            
            # Check for special instructions about voice style
            voice_style = ""
            if manual_instructions:
                if "third person" in manual_instructions.lower():
                    voice_style = "Override: Use third person narrator style."
                elif "narrator" in manual_instructions.lower():
                    voice_style = "Override: Use professional narrator style."
            
            prompt = f"""Create an influencer voice-over script that matches the original video's style.

{style_guidance}

PRODUCT/CONTENT INFO:
{free_text[:2000]}

YOUR TASK:
1. If original VO exists: MATCH its exact style, tone, energy, and structure
2. Create NEW content about the product/experience above
3. Keep the same energy and feeling as the original
4. Length: ~{target_words} words ({target_duration:.0f} seconds)
5. Language: {language_name}

{f"SPECIAL INSTRUCTIONS: {manual_instructions}" if manual_instructions else ""}
{voice_style}

IMPORTANT:
- If original uses names, you can use names
- If original is casual, be casual
- If original is hype, be hype
- If original is calm, be calm
- NO brackets or stage directions
- ONLY output the spoken words

Generate the voice-over:"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are an expert at matching voice-over styles. When given an original VO, you create new content that feels like it belongs to the same video - same energy, same rhythm, same personality. You write naturally in {language_name}."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            script = response.choices[0].message.content.strip()
            
            # Clean up any brackets or stage directions
            script = re.sub(r'\[.*?\]', '', script)
            script = re.sub(r'\(.*?\)', '', script)
            script = script.strip()
            
            word_count = len(script.split())
            logger.info(f"✅ Generated influencer VO script: {word_count} words (target: {target_words})")
            logger.info(f"📝 VO Script preview: {script[:200]}...")
            
            return script
            
        except Exception as e:
            logger.error(f"❌ Failed to generate influencer VO script: {e}")
            return "Check this out! I recently discovered something amazing and I had to share it with you. Click below to learn more!"


# =============================================================================
# KIE.AI SERVICE (NANO BANANA + RUNWAY)
# =============================================================================
class KieAIService:
    """Service for Kie.ai API interactions (Nano Banana and Runway)."""
    
    def __init__(self, api_key: str, s3_service=None):
        """Initialize Kie.ai service.
        
        Args:
            api_key: Kie.ai API key.
            s3_service: Optional S3 service for uploading CTA buttons.
        """
        self.api_key = api_key
        self.base_url = config.KIE_BASE_URL
        self.s3_service = s3_service
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.info("✅ Kie.ai client initialized")
    
    # Nano Banana max prompt length
    NANO_BANANA_MAX_PROMPT_LENGTH: int = 4000
    
    def _sanitize_prompt_remove_brands(self, text: str, target_language: str = "en", article_text: str = "") -> str:
        """Replace brand names with generic terms while keeping product/scene descriptions accurate.
        
        This replaces specific brand names with language-appropriate generic alternatives,
        but keeps all product physical descriptions (shape, color, size, etc.) intact.
        
        The replacement text is context-aware based on the article content:
        - Sale/Discount offers → "SALE" or localized version
        - Job/Career offers → "APPLY"
        - Health/Wellness → "TRY NOW"
        - Learning/Course → "LEARN MORE"
        - Other/Unclear → Remove branding completely (no replacement)
        
        Args:
            text: The prompt text to sanitize.
            target_language: Target language code (e.g., 'en', 'he', 'da', 'tr').
            article_text: The article content to determine context-appropriate text.
            
        Returns:
            Sanitized text with brand names replaced by context-appropriate terms.
        """
        if not text:
            return text
        
        import re
        
        # Determine context-appropriate text based on article content
        article_lower = article_text.lower() if article_text else ""
        
        # Context detection patterns
        is_sale_offer = any(word in article_lower for word in [
            'sale', 'discount', 'off', '%', 'מבצע', 'הנחה', 'rabat', 'indirim', 'promo', 'offer', 'deal'
        ])
        is_job_offer = any(word in article_lower for word in [
            'job', 'career', 'work', 'hiring', 'employment', 'עבודה', 'משרה', 'קריירה', 'job opening'
        ])
        is_health_product = any(word in article_lower for word in [
            'health', 'wellness', 'supplement', 'vitamin', 'weight', 'slim', 'בריאות', 'דיאטה'
        ])
        is_learning = any(word in article_lower for word in [
            'learn', 'course', 'training', 'education', 'tutorial', 'קורס', 'לימוד'
        ])
        
        # Language-specific text translations
        text_translations = {
            "sale": {"en": "SALE", "he": "מבצע", "da": "TILBUD", "tr": "İNDİRİM", "de": "ANGEBOT", "fr": "PROMO", "es": "OFERTA", "it": "OFFERTA", "nl": "AANBIEDING", "pt": "PROMOÇÃO", "ru": "СКИДКА", "ar": "عرض", "ja": "セール", "ko": "할인", "zh": "促销"},
            "apply": {"en": "APPLY", "he": "הגש מועמדות", "da": "ANSØG", "tr": "BAŞVUR", "de": "BEWERBEN", "fr": "POSTULER", "es": "APLICAR", "it": "CANDIDATI", "nl": "SOLLICITEER", "pt": "APLICAR", "ru": "ПОДАТЬ", "ar": "تقدم", "ja": "応募", "ko": "지원", "zh": "申请"},
            "try": {"en": "TRY NOW", "he": "נסה עכשיו", "da": "PRØV NU", "tr": "ŞİMDİ DENE", "de": "JETZT TESTEN", "fr": "ESSAYER", "es": "PROBAR", "it": "PROVA ORA", "nl": "PROBEER NU", "pt": "EXPERIMENTE", "ru": "ПОПРОБУЙ", "ar": "جرب الآن", "ja": "今すぐ試す", "ko": "지금 시도", "zh": "立即尝试"},
            "learn": {"en": "LEARN MORE", "he": "למד עוד", "da": "LÆR MERE", "tr": "DAHA FAZLA", "de": "MEHR ERFAHREN", "fr": "EN SAVOIR PLUS", "es": "SABER MÁS", "it": "SCOPRI DI PIÙ", "nl": "MEER INFO", "pt": "SAIBA MAIS", "ru": "УЗНАТЬ БОЛЬШЕ", "ar": "اعرف المزيد", "ja": "詳しく見る", "ko": "더 알아보기", "zh": "了解更多"},
        }
        
        # Select appropriate text based on context
        lang_key = target_language.lower()
        if is_sale_offer:
            replacement_text = text_translations["sale"].get(lang_key, "SALE")
        elif is_job_offer:
            replacement_text = text_translations["apply"].get(lang_key, "APPLY")
        elif is_health_product:
            replacement_text = text_translations["try"].get(lang_key, "TRY NOW")
        elif is_learning:
            replacement_text = text_translations["learn"].get(lang_key, "LEARN MORE")
        else:
            # No clear context - remove branding without replacement
            replacement_text = ""
        
        # Only replace specific brand-related patterns with generic alternatives
        # Keep product descriptions intact!
        brand_patterns = [
            # Replace "brand: X" or "brand name: X" with generic
            (r'\bbrand\s*(?:name)?\s*[:=]\s*["\']?[\w\s\-\.]+["\']?', ''),
            # Replace "logo" with "design element"
            (r'\b(?:brand\s+)?logo\b', 'design element'),
            # Remove trademark/registered symbols
            (r'[™®©]', ''),
        ]
        
        # Only add text replacement patterns if we have a replacement text
        if replacement_text:
            brand_patterns.extend([
                # Replace brand text on screens/packaging with contextual text
                (r'\btext\s+(?:saying|showing|displaying)\s*["\'][\w\s\-\.%]+["\']', f'text showing "{replacement_text}"'),
                # Replace specific percentages like "20% off" with contextual text (only for sale offers)
                (r'\b\d+%\s*(?:off|discount|rabat|indirim|הנחה)\b', replacement_text if is_sale_offer else ''),
            ])
        else:
            # No replacement - just remove the branding text
            brand_patterns.extend([
                (r'\btext\s+(?:saying|showing|displaying)\s*["\'][\w\s\-\.%]+["\']', ''),
                (r'\b\d+%\s*(?:off|discount|rabat|indirim|הנחה)\b', ''),
            ])
        
        sanitized = text
        for pattern, replacement in brand_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def generate_image_nano_banana(
        self, 
        prompt: str, 
        reference_image_url: Optional[str] = None,
        reference_description: Optional[str] = None,
        target_language: str = "en",
        article_text: str = ""
    ) -> Optional[str]:
        """Generate an image using Nano Banana.
        
        Args:
            prompt: Text prompt for image generation (max 5000 characters).
            reference_image_url: Optional URL of a reference image for style/content guidance.
            reference_description: Optional description of the reference image to include in prompt.
            target_language: Target language for any text on the image (e.g., 'en', 'he', 'da').
            article_text: Article content for context-aware text replacement.
            
        Returns:
            URL of the generated image, or None if failed.
        """
        try:
            # Lightly sanitize prompt to replace brand names with context-appropriate terms
            # Keep product and scene descriptions accurate!
            prompt = self._sanitize_prompt_remove_brands(prompt, target_language, article_text)
            if reference_description:
                reference_description = self._sanitize_prompt_remove_brands(reference_description, target_language, article_text)
            
            # If we have a reference description, prepend it to the prompt with detailed instructions
            if reference_description:
                # Enhance reference description - focus on accurate physical recreation
                enhanced_ref = f"""REFERENCE PRODUCT (match physical appearance exactly):
{reference_description}

ACCURACY REQUIREMENTS:
- Match EXACT shape, colors, size, materials from reference
- Recreate the product's physical appearance precisely
- If text/branding appears on product, REMOVE it completely - product must be clean
- For text overlays on screen (not on product): only add if specifically instructed

SCENE:"""
                prompt = f"{enhanced_ref}\n\n{prompt}"
            
            # Truncate prompt if it exceeds max length (5000 chars for Nano Banana)
            if len(prompt) > self.NANO_BANANA_MAX_PROMPT_LENGTH:
                logger.warning(
                    f"⚠️ Prompt too long ({len(prompt)} chars), "
                    f"truncating to {self.NANO_BANANA_MAX_PROMPT_LENGTH} chars"
                )
                prompt = prompt[:self.NANO_BANANA_MAX_PROMPT_LENGTH - 3] + "..."
            
            # Log reference information
            if reference_image_url:
                logger.info(f"📸 Using product reference image: {reference_image_url[:80]}...")
            if reference_description:
                logger.info(f"📝 Using product reference description: {reference_description[:100]}...")
            
            ref_info = f" (with reference image + description)" if reference_image_url and reference_description else \
                       f" (with reference image)" if reference_image_url else \
                       f" (with reference description)" if reference_description else ""
            logger.info(f"🍌 Generating image with Nano Banana ({len(prompt)} chars){ref_info}...")
            
            url = f"{self.base_url}/api/v1/jobs/createTask"
            
            # Choose model based on whether reference image is available
            # nano-banana-edit accepts image_urls for reference images
            # nano-banana is text-only
            if reference_image_url:
                model = "google/nano-banana-edit"
                # Use reference image + text description for consistent, logical output. Product must be clean.
                clean_prompt = f"""Logical, natural, human scene. No surreal or inconsistent elements.
Use the reference image for the product's EXACT physical appearance (shape, colors, materials, size).
CRITICAL: The product must be COMPLETELY CLEAN - NO text, NO logos, NO branding on the product surface itself.
If the reference product has text/logos, REMOVE them completely. The product surface should be plain/clean.

{prompt}"""
                input_params = {
                    "prompt": clean_prompt,
                    "image_urls": [reference_image_url],  # Pass as array per API docs
                    "output_format": "png",
                    "image_size": "9:16"
                }
                logger.info(f"🎯 Using nano-banana-edit with reference image: {reference_image_url[:60]}...")
            else:
                model = "google/nano-banana"
                input_params = {
                    "prompt": prompt,
                    "output_format": "png",
                    "image_size": "9:16"
                }
                logger.warning(f"⚠️ No reference image - using nano-banana (text-only)")
            
            payload = {
                "model": model,
                "input": input_params
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 200 and "taskId" in result.get("data", {}):
                task_id = result["data"]["taskId"]
                logger.info(f"✅ Nano Banana task created: {task_id}")
                
                # Poll for completion
                return self._wait_for_image_task(task_id)
            else:
                logger.error(f"❌ Nano Banana API error: {result}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating image with Nano Banana: {e}")
            return None
    
    def generate_clean_product_image(
        self, 
        reference_image_urls: List[str],
        product_description: str
    ) -> Optional[str]:
        """Generate a clean, isolated product image from reference frames.
        
        Takes multiple reference frames from the video and generates a clean,
        professional product image with no background, text, or overlays.
        This clean image is then used as the reference for scene generations.
        
        Args:
            reference_image_urls: List of reference image URLs (up to 3).
            product_description: Detailed description of the product.
            
        Returns:
            URL of the clean product image, or None if failed.
        """
        try:
            if not reference_image_urls:
                logger.warning("⚠️ No reference images provided for clean product generation")
                return None
            
            # Use up to 3 reference images
            ref_urls = reference_image_urls[:3]
            logger.info(f"🧹 Generating clean product image from {len(ref_urls)} reference frames...")
            
            # Lightly sanitize product description - keep physical details accurate
            sanitized_description = self._sanitize_prompt_remove_brands(product_description) if product_description else ""
            
            # Build a detailed prompt for clean product extraction
            prompt = f"""Generate a clean, isolated product photograph.

TASK: Extract the product from the reference images and create a professional product photo.

PRODUCT DESCRIPTION:
{sanitized_description}

REQUIREMENTS:
- Show ONLY the product itself on a white background
- No people, hands, or body parts
- No scene elements or environment
- Professional product photography lighting
- Sharp focus on the product
- Match the EXACT physical appearance from reference images: shape, colors, materials, textures, size
- CRITICAL: If text/logos appear on product, REMOVE them completely - the product surface must be CLEAN with no text
- The product must have a plain, clean surface with no branding, logos, or text of any kind

OUTPUT: A clean, isolated product image with no text or branding."""

            # Truncate if needed
            if len(prompt) > self.NANO_BANANA_MAX_PROMPT_LENGTH:
                prompt = prompt[:self.NANO_BANANA_MAX_PROMPT_LENGTH - 3] + "..."
            
            url = f"{self.base_url}/api/v1/jobs/createTask"
            
            # Use nano-banana-edit with multiple reference images
            model = "google/nano-banana-edit"
            input_params = {
                "prompt": prompt,
                "image_urls": ref_urls,  # Pass all reference URLs
                "output_format": "png",
                "image_size": "1:1"  # Square for product image
            }
            
            logger.info(f"🎯 Using nano-banana-edit with {len(ref_urls)} reference images")
            for i, ref_url in enumerate(ref_urls):
                logger.info(f"   Reference {i+1}: {ref_url[:60]}...")
            
            payload = {
                "model": model,
                "input": input_params
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 200 and "taskId" in result.get("data", {}):
                task_id = result["data"]["taskId"]
                logger.info(f"✅ Clean product image task created: {task_id}")
                
                # Poll for completion
                clean_url = self._wait_for_image_task(task_id)
                if clean_url:
                    logger.info(f"✅ Clean product image generated: {clean_url[:60]}...")
                return clean_url
            else:
                logger.error(f"❌ Clean product image API error: {result}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating clean product image: {e}")
            return None
    
    def _wait_for_image_task(self, task_id: str, timeout: int = 300) -> Optional[str]:
        """Wait for image generation task to complete.
        
        Args:
            task_id: Task ID to poll.
            timeout: Maximum wait time in seconds.
            
        Returns:
            URL of the generated image, or None if failed/timeout.
        """
        start_time = time.time()
        check_interval = 5
        
        while time.time() - start_time < timeout:
            try:
                url = f"{self.base_url}/api/v1/jobs/recordInfo"
                response = requests.get(
                    f"{url}?taskId={task_id}",
                    headers=self.headers,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                
                if result.get("code") == 200:
                    data = result.get("data", {})
                    state = data.get("state", "").lower()
                    
                    if state == "success":
                        result_json_str = data.get("resultJson")
                        if result_json_str:
                            result_json = json.loads(result_json_str)
                            result_urls = result_json.get("resultUrls", [])
                            if result_urls:
                                logger.info(f"✅ Image generated successfully")
                                return result_urls[0]
                    
                    elif state == "fail":
                        logger.error(f"❌ Image generation failed")
                        return None
                    
                    # Still processing, continue polling
                    
            except Exception as e:
                logger.error(f"❌ Error polling task status: {e}")
            
            time.sleep(check_interval)
        
        logger.error("❌ Image generation timeout")
        return None
    
    def generate_video_runway(
        self, 
        prompt: str, 
        image_url: str,
        duration: float = 5.0
    ) -> Optional[str]:
        """Generate a video using Runway.
        
        Args:
            prompt: Motion/animation prompt.
            image_url: URL of the source image.
            duration: Video duration in seconds (will be rounded to 5 or 10).
            
        Returns:
            URL of the generated video, or None if failed.
        """
        try:
            # Runway supports 5 or 10 second durations
            # Round to nearest supported duration
            runway_duration = 5 if duration <= 7.5 else 10
            logger.info(f"🎬 Generating video with Runway (duration: {runway_duration}s, original scene: {duration:.1f}s)...")
            
            url = f"{self.base_url}/api/v1/runway/generate"
            
            payload = {
                "prompt": prompt,
                "imageUrl": image_url,
                "duration": runway_duration,
                "quality": "720p",
                "waterMark": ""
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 200 and "taskId" in result.get("data", {}):
                task_id = result["data"]["taskId"]
                logger.info(f"✅ Runway task created: {task_id}")
                
                # Poll for completion
                return self._wait_for_video_task(task_id)
            else:
                logger.error(f"❌ Runway API error: {result}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating video with Runway: {e}")
            return None
    
    def _wait_for_video_task(self, task_id: str, timeout: int = 600) -> Optional[str]:
        """Wait for Runway video generation task to complete.
        
        Uses the dedicated Runway record-detail endpoint for accurate status.
        
        Args:
            task_id: Task ID to poll.
            timeout: Maximum wait time in seconds.
            
        Returns:
            URL of the generated video, or None if failed/timeout.
        """
        start_time = time.time()
        check_interval = 10
        
        while time.time() - start_time < timeout:
            try:
                # Use the dedicated Runway record-detail endpoint
                url = f"{self.base_url}/api/v1/runway/record-detail"
                response = requests.get(
                    f"{url}?taskId={task_id}",
                    headers=self.headers,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                
                if result.get("code") == 200:
                    data = result.get("data", {})
                    state = data.get("state", "").lower()
                    
                    logger.debug(f"Runway task {task_id} status: {state}")
                    
                    if state == "success":
                        # Get video URL from videoInfo object
                        video_info = data.get("videoInfo", {})
                        video_url = video_info.get("videoUrl")
                        if video_url:
                            logger.info(f"✅ Video generated successfully")
                            return video_url
                    
                    elif state == "fail":
                        fail_msg = data.get("failMsg", "Unknown error")
                        logger.error(f"❌ Video generation failed: {fail_msg}")
                        return None
                    
                    # States: wait, queueing, generating - continue polling
                    
            except Exception as e:
                logger.error(f"❌ Error polling Runway task status: {e}")
            
            time.sleep(check_interval)
        
        logger.error("❌ Video generation timeout")
        return None
    
    def generate_video_kling(
        self,
        prompt: str,
        image_url: str,
        duration: float = 5.0,
        negative_prompt: str = "blur, distort, and low quality",
        cfg_scale: float = 0.5
    ) -> Optional[str]:
        """Generate a video using Kling V2.5 Turbo Image-to-Video Pro.
        
        Args:
            prompt: Motion/animation prompt.
            image_url: URL of the source image.
            duration: Video duration in seconds (5 or 10).
            negative_prompt: What to avoid in the video.
            cfg_scale: Config scale (0.0-1.0, default 0.5).
            
        Returns:
            URL of the generated video, or None if failed.
        """
        try:
            # Kling supports 5 or 10 second durations
            # Request based on target scene duration + buffer for slow motion
            # If scene needs > 5s (accounting for 2x slow motion max), request 10s
            kling_duration = "5" if duration <= 10.0 else "10"  # 5s video can extend to 10s with 2x slowmo
            logger.info(f"🎬 Generating video with Kling V2.5 (duration: {kling_duration}s, original scene: {duration:.1f}s)...")
            
            url = f"{self.base_url}/api/v1/jobs/createTask"
            
            payload = {
                "model": "kling/v2-5-turbo-image-to-video-pro",
                "input": {
                    "prompt": prompt,
                    "image_url": image_url,
                    "duration": kling_duration,
                    "negative_prompt": negative_prompt,
                    "cfg_scale": cfg_scale
                }
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 200 and "taskId" in result.get("data", {}):
                task_id = result["data"]["taskId"]
                logger.info(f"✅ Kling task created: {task_id}")
                
                # Poll for completion using generic endpoint
                return self._wait_for_kling_task(task_id)
            else:
                logger.error(f"❌ Kling API error: {result}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating video with Kling: {e}")
            return None
    
    def _wait_for_kling_task(self, task_id: str, timeout: int = 600) -> Optional[str]:
        """Wait for Kling video generation task to complete.
        
        Uses the generic jobs/recordInfo endpoint for status.
        
        Args:
            task_id: Task ID to poll.
            timeout: Maximum wait time in seconds.
            
        Returns:
            URL of the generated video, or None if failed/timeout.
        """
        start_time = time.time()
        check_interval = 10
        
        while time.time() - start_time < timeout:
            try:
                url = f"{self.base_url}/api/v1/jobs/recordInfo"
                response = requests.get(
                    f"{url}?taskId={task_id}",
                    headers=self.headers,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                
                if result.get("code") == 200:
                    data = result.get("data", {})
                    state = data.get("state", "").lower()
                    
                    logger.debug(f"Kling task {task_id} status: {state}")
                    
                    if state == "success":
                        # Get video URL from resultJson
                        result_json_str = data.get("resultJson", "{}")
                        try:
                            result_json = json.loads(result_json_str) if isinstance(result_json_str, str) else result_json_str
                            result_urls = result_json.get("resultUrls", [])
                            if result_urls:
                                video_url = result_urls[0]
                                logger.info(f"✅ Kling video generated successfully")
                                return video_url
                        except json.JSONDecodeError:
                            logger.error(f"❌ Failed to parse Kling resultJson: {result_json_str}")
                            return None
                    
                    elif state == "fail":
                        fail_msg = data.get("failMsg", "Unknown error")
                        logger.error(f"❌ Kling video generation failed: {fail_msg}")
                        return None
                    
                    # States: waiting, queuing, generating - continue polling
                    
            except Exception as e:
                logger.error(f"❌ Error polling Kling task status: {e}")
            
            time.sleep(check_interval)
        
        logger.error("❌ Kling video generation timeout")
        return None
    
    def generate_video_kling_30(
        self,
        prompt: str,
        image_url: str,
        duration: float = 5.0,
        mode: str = "std"
    ) -> Optional[str]:
        """Generate a video using Kling 3.0 (model: kling-3.0/video).
        
        Uses Kie.ai createTask + recordInfo. API expects image_urls (array),
        sound, multi_shots, multi_prompt. Single-shot: first frame = image_urls[0].
        
        Args:
            prompt: Motion/animation prompt.
            image_url: URL of the source image (first frame).
            duration: Video duration in seconds (3-15).
            mode: "std" (standard resolution) or "pro" (higher resolution).
            
        Returns:
            URL of the generated video, or None if failed.
        """
        try:
            logger.info(f"🎬 Generating video with Kling 3.0 (mode={mode}, scene duration: {duration:.1f}s)...")
            
            url = f"{self.base_url}/api/v1/jobs/createTask"
            
            duration_str = "10" if duration > 7 else "5"
            if duration <= 4:
                duration_str = "3"
            elif duration <= 5:
                duration_str = "5"
            elif duration <= 10:
                duration_str = "10"
            else:
                duration_str = "15"
            
            payload = {
                "model": "kling-3.0/video",
                "input": {
                    "prompt": prompt,
                    "image_urls": [image_url],
                    "sound": False,
                    "duration": duration_str,
                    "aspect_ratio": "9:16",
                    "mode": mode,
                    "multi_shots": False,
                    "multi_prompt": [],
                }
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 200 and "taskId" in result.get("data", {}):
                task_id = result["data"]["taskId"]
                logger.info(f"✅ Kling 3.0 task created: {task_id}")
                return self._wait_for_kling_task(task_id)
            else:
                logger.error(f"❌ Kling 3.0 API error: {result}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating video with Kling 3.0: {e}")
            return None
    
    def generate_cta_button(self, cta_text: str) -> Optional[str]:
        """Generate a CTA button image using Nano Banana AI model.
        
        Creates a beautiful CTA button with the specified text using the Kie.ai Nano Banana model.
        
        Args:
            cta_text: Text to display on the button.
            
        Returns:
            URL of the generated button image, or None if failed.
        """
        try:
            logger.info(f"🔘 Generating CTA button with Nano Banana: '{cta_text}'...")
            
            # Create the prompt for a beautiful CTA button with green screen background
            # Green screen color: #00FF00 (pure chroma key green) for easy removal
            prompt = f"""Create a beautiful, modern call-to-action button with the exact text "{cta_text}" prominently displayed.

CRITICAL REQUIREMENTS:
- The ENTIRE background MUST be a perfectly flat, solid, uniform bright green color (hex #00FF00)
- The green background must be completely smooth and uniform - NO gradients, NO textures, NO variations
- NO shadows anywhere in the image - the button must have NO drop shadow, NO glow, NO blur effects
- The green must extend to all edges of the image with zero variation

Button design:
- A vibrant gradient background on the button itself (blue to purple or orange - NO GREEN)
- Sharp, clean rounded corners
- The text "{cta_text}" clearly visible in white color
- NO shadow effects on the button or text
- NO glow effects
- Flat design style - clean and simple
- The button should be centered in the image

IMPORTANT: The background must be 100% flat solid green (#00FF00) with absolutely no shadows, gradients, or effects.
The text on the button must be exactly: {cta_text}"""

            # Create task with Nano Banana model
            url = f"{self.base_url}/api/v1/jobs/createTask"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "nano-banana-pro",
                "input": {
                    "prompt": prompt,
                    "aspect_ratio": "16:9",
                    "resolution": "1K",
                    "output_format": "png"
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if result.get("code") != 200:
                logger.error(f"❌ Nano Banana API error: {result.get('message', 'Unknown error')}")
                return None
            
            task_id = result.get("data", {}).get("taskId")
            if not task_id:
                logger.error("❌ No task ID returned from Nano Banana")
                return None
            
            logger.info(f"📋 Nano Banana CTA task created: {task_id}")
            
            # Poll for completion
            query_url = f"{self.base_url}/api/v1/jobs/recordInfo"
            max_attempts = 60  # 5 minutes max
            
            for attempt in range(max_attempts):
                time.sleep(5)
                
                query_response = requests.get(
                    query_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    params={"taskId": task_id},
                    timeout=30
                )
                query_response.raise_for_status()
                status_result = query_response.json()
                
                if status_result.get("code") != 200:
                    continue
                
                data = status_result.get("data", {})
                state = data.get("state", "")
                
                if state == "success":
                    result_json = data.get("resultJson", "{}")
                    try:
                        result_data = json.loads(result_json)
                        result_urls = result_data.get("resultUrls", [])
                        if result_urls:
                            image_url = result_urls[0]
                            logger.info(f"✅ CTA button generated: {image_url}")
                            return image_url
                    except json.JSONDecodeError:
                        logger.error(f"❌ Failed to parse result JSON: {result_json}")
                    break
                    
                elif state == "fail":
                    fail_msg = data.get("failMsg", "Unknown error")
                    logger.error(f"❌ Nano Banana CTA generation failed: {fail_msg}")
                    break
                    
                elif state in ["waiting", "queuing", "generating"]:
                    if attempt % 6 == 0:  # Log every 30 seconds
                        logger.info(f"⏳ CTA generation status: {state}...")
                    continue
            
            logger.error("❌ CTA button generation timeout")
            return None
                
        except Exception as e:
            logger.error(f"❌ Error generating CTA button with Nano Banana: {e}")
            return None
    
    def generate_opening_text(self, text: str) -> Optional[str]:
        """Generate an opening text overlay image using Nano Banana AI model.
        
        Creates a stylish text overlay for the opening scene with green screen background.
        
        Args:
            text: Text to display in the opening.
            
        Returns:
            URL of the generated text image, or None if failed.
        """
        try:
            logger.info(f"🎬 Generating opening text with Nano Banana: '{text}'...")
            
            # Create the prompt for opening text with green screen background
            prompt = f"""Create a stylish, modern text overlay with the exact text "{text}" prominently displayed.

CRITICAL REQUIREMENTS:
- The ENTIRE background MUST be a perfectly flat, solid, uniform bright green color (hex #00FF00)
- The green background must be completely smooth and uniform - NO gradients, NO textures, NO variations
- NO shadows anywhere in the image - NO drop shadow, NO glow, NO blur effects
- The green must extend to all edges of the image with zero variation

Text design:
- Large, bold, eye-catching typography
- White or light colored text for visibility
- Modern sans-serif font style
- The text should be centered in the image
- Can have a subtle gradient or bold style on the text itself (NOT the background)
- NO shadow effects on the text
- NO glow effects
- Clean, professional look suitable for video opening

IMPORTANT: The background must be 100% flat solid green (#00FF00) with absolutely no shadows, gradients, or effects.
The text must be exactly: {text}"""

            # Create task with Nano Banana model
            url = f"{self.base_url}/api/v1/jobs/createTask"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "nano-banana-pro",
                "input": {
                    "prompt": prompt,
                    "aspect_ratio": "16:9",
                    "resolution": "1K",
                    "output_format": "png"
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if result.get("code") != 200:
                logger.error(f"❌ Nano Banana API error: {result.get('message', 'Unknown error')}")
                return None
            
            task_id = result.get("data", {}).get("taskId")
            if not task_id:
                logger.error("❌ No task ID returned from Nano Banana")
                return None
            
            logger.info(f"📋 Nano Banana opening text task created: {task_id}")
            
            # Poll for completion
            query_url = f"{self.base_url}/api/v1/jobs/recordInfo"
            max_attempts = 60  # 5 minutes max
            
            for attempt in range(max_attempts):
                time.sleep(5)
                
                query_response = requests.get(
                    query_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    params={"taskId": task_id},
                    timeout=30
                )
                query_response.raise_for_status()
                status_result = query_response.json()
                
                if status_result.get("code") != 200:
                    continue
                
                data = status_result.get("data", {})
                state = data.get("state", "")
                
                if state == "success":
                    result_json = data.get("resultJson", "{}")
                    try:
                        result_data = json.loads(result_json)
                        result_urls = result_data.get("resultUrls", [])
                        if result_urls:
                            image_url = result_urls[0]
                            logger.info(f"✅ Opening text generated: {image_url}")
                            return image_url
                    except json.JSONDecodeError:
                        logger.error(f"❌ Failed to parse result JSON: {result_json}")
                    break
                    
                elif state == "fail":
                    fail_msg = data.get("failMsg", "Unknown error")
                    logger.error(f"❌ Nano Banana opening text generation failed: {fail_msg}")
                    break
                    
                elif state in ["waiting", "queuing", "generating"]:
                    if attempt % 6 == 0:  # Log every 30 seconds
                        logger.info(f"⏳ Opening text generation status: {state}...")
                    continue
            
            logger.error("❌ Opening text generation timeout")
            return None
                
        except Exception as e:
            logger.error(f"❌ Error generating opening text with Nano Banana: {e}")
            return None


# =============================================================================
# IMAGE PROCESSING UTILITIES (for CTA button)
# =============================================================================

def remove_green_background(image_path: str, output_path: str) -> bool:
    """Remove green background from an image and save with transparency.
    
    Samples the actual green color from the image corners for accurate removal.
    
    Args:
        image_path: Path to the input image.
        output_path: Path to save the output PNG with transparency.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        logger.info("🎨 Removing green background from CTA button...")
        
        # Open the image
        img = Image.open(image_path).convert("RGBA")
        pixels = img.load()
        
        width, height = img.size
        
        # Sample the green color from the corners (where background should be)
        corner_samples = []
        sample_size = 10  # Sample a small area from each corner
        
        # Top-left corner
        for y in range(min(sample_size, height)):
            for x in range(min(sample_size, width)):
                r, g, b, a = pixels[x, y]
                corner_samples.append((r, g, b))
        
        # Top-right corner
        for y in range(min(sample_size, height)):
            for x in range(max(0, width - sample_size), width):
                r, g, b, a = pixels[x, y]
                corner_samples.append((r, g, b))
        
        # Bottom-left corner
        for y in range(max(0, height - sample_size), height):
            for x in range(min(sample_size, width)):
                r, g, b, a = pixels[x, y]
                corner_samples.append((r, g, b))
        
        # Bottom-right corner
        for y in range(max(0, height - sample_size), height):
            for x in range(max(0, width - sample_size), width):
                r, g, b, a = pixels[x, y]
                corner_samples.append((r, g, b))
        
        # Calculate average green color from samples
        if corner_samples:
            avg_r = sum(s[0] for s in corner_samples) // len(corner_samples)
            avg_g = sum(s[1] for s in corner_samples) // len(corner_samples)
            avg_b = sum(s[2] for s in corner_samples) // len(corner_samples)
            sampled_green = (avg_r, avg_g, avg_b)
            logger.info(f"🎨 Sampled background color: RGB{sampled_green}")
        else:
            sampled_green = (46, 204, 113)  # Default fallback
            logger.info("🎨 Using default green color")
        
        # More strict tolerance to avoid removing the button itself
        # Only remove pixels that are clearly green (high G, low R and B relative to G)
        tolerance = 40
        
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                
                # Check if pixel is close to the sampled green color
                # AND the pixel is predominantly green (g > r and g > b)
                is_green_bg = (
                    abs(r - sampled_green[0]) < tolerance and 
                    abs(g - sampled_green[1]) < tolerance and 
                    abs(b - sampled_green[2]) < tolerance and
                    g > r * 0.8 and  # Green channel should be significant
                    g > b * 0.8
                )
                
                if is_green_bg:
                    # Make pixel transparent
                    pixels[x, y] = (r, g, b, 0)
        
        # Save with transparency
        img.save(output_path, "PNG")
        logger.info(f"✅ Green background removed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error removing green background: {e}")
        return False


def add_glow_effect(image_path: str, output_path: str, glow_color: tuple = (255, 255, 255), glow_radius: int = 15) -> bool:
    """Add a glow effect around the non-transparent parts of an image.
    
    Args:
        image_path: Path to the input PNG with transparency.
        output_path: Path to save the output image with glow.
        glow_color: RGB tuple for the glow color (default: white).
        glow_radius: Blur radius for the glow effect.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        logger.info("✨ Adding glow effect to CTA button...")
        
        # Open the image with transparency
        img = Image.open(image_path).convert("RGBA")
        
        # Create a copy for the glow
        # Extract alpha channel
        alpha = img.split()[3]
        
        # Create a solid color image for glow
        glow_base = Image.new("RGBA", img.size, glow_color + (255,))
        
        # Apply alpha mask to glow base
        glow_base.putalpha(alpha)
        
        # Blur the glow
        glow_blurred = glow_base.filter(ImageFilter.GaussianBlur(radius=glow_radius))
        
        # Enhance the glow brightness
        enhancer = ImageEnhance.Brightness(glow_blurred)
        glow_blurred = enhancer.enhance(1.5)
        
        # Create final image with glow behind the button
        final_size = (img.width + glow_radius * 2, img.height + glow_radius * 2)
        final_img = Image.new("RGBA", final_size, (0, 0, 0, 0))
        
        # Paste glow (centered)
        final_img.paste(glow_blurred, (glow_radius, glow_radius), glow_blurred)
        
        # Paste original button on top
        final_img.paste(img, (glow_radius, glow_radius), img)
        
        # Save result
        final_img.save(output_path, "PNG")
        logger.info(f"✅ Glow effect added: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error adding glow effect: {e}")
        return False


# =============================================================================
# RENDI.DEV SERVICE
# =============================================================================
class RendiService:
    """Service for Rendi.dev API interactions. Falls back to local FFmpeg when Rendi API fails (e.g. 403)."""
    
    def __init__(self, api_key: str, s3_service=None):
        """Initialize Rendi service.
        
        Args:
            api_key: Rendi.dev API key.
            s3_service: Optional S3 service for uploading results when using local FFmpeg fallback.
        """
        self.api_key = api_key
        self.base_url = config.RENDI_BASE_URL
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        self.s3_service = s3_service
        logger.info("✅ Rendi.dev client initialized")
    
    @staticmethod
    def _ffmpeg_available() -> bool:
        """Check if local FFmpeg is available for fallback."""
        return FFmpegProcessor.check_ffmpeg_installed()
    
    def _get_duration_ffprobe(self, media_url: str, is_video: bool = True) -> Optional[float]:
        """Get duration of video or audio using local ffprobe (works on URLs).
        
        Returns:
            Duration in seconds, or None if failed.
        """
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", media_url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as e:
            logger.debug(f"ffprobe duration failed: {e}")
        return None
    
    def _trim_video_local(self, video_url: str, duration: float) -> Optional[str]:
        """Trim video to duration using local FFmpeg. Downloads, trims, uploads to S3 if available."""
        if not self._ffmpeg_available():
            return None
        if not self.s3_service:
            logger.warning("⚠️ Local trim fallback requires S3 service for upload")
            return None
        tmp_dir = tempfile.mkdtemp()
        try:
            inp = os.path.join(tmp_dir, "in.mp4")
            out = os.path.join(tmp_dir, "out.mp4")
            r = requests.get(video_url, timeout=120, stream=True)
            r.raise_for_status()
            with open(inp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            subprocess.run([
                "ffmpeg", "-y", "-i", inp, "-t", str(duration),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out
            ], check=True, capture_output=True, timeout=300)
            if not os.path.isfile(out):
                return None
            key = f"rendi_fallback/trim_{int(time.time())}_{os.getpid()}.mp4"
            return self.s3_service.upload_video_from_path(out, key)
        except Exception as e:
            logger.warning(f"⚠️ Local trim fallback failed: {e}")
            return None
        finally:
            try:
                for f in ["in.mp4", "out.mp4"]:
                    p = os.path.join(tmp_dir, f)
                    if os.path.isfile(p):
                        os.remove(p)
                os.rmdir(tmp_dir)
            except Exception:
                pass
    
    def _concatenate_videos_local(self, video_urls: List[str]) -> Optional[str]:
        """Concatenate videos using local FFmpeg. Downloads, concats, uploads to S3 if available."""
        if not self._ffmpeg_available() or not video_urls:
            return None
        if not self.s3_service:
            logger.warning("⚠️ Local concat fallback requires S3 service for upload")
            return None
        tmp_dir = tempfile.mkdtemp()
        try:
            inputs = []
            for i, url in enumerate(video_urls):
                p = os.path.join(tmp_dir, f"in_{i}.mp4")
                r = requests.get(url, timeout=120, stream=True)
                r.raise_for_status()
                with open(p, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                inputs.append(p)
            list_file = os.path.join(tmp_dir, "list.txt")
            with open(list_file, "w") as f:
                for p in inputs:
                    f.write(f"file '{p}'\n")
            out = os.path.join(tmp_dir, "out.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                "-c", "copy", "-movflags", "+faststart", out
            ], check=True, capture_output=True, timeout=600)
            if not os.path.isfile(out):
                return None
            key = f"rendi_fallback/concat_{int(time.time())}_{os.getpid()}.mp4"
            return self.s3_service.upload_video_from_path(out, key)
        except Exception as e:
            logger.warning(f"⚠️ Local concat fallback failed: {e}")
            return None
        finally:
            try:
                for f in os.listdir(tmp_dir):
                    os.remove(os.path.join(tmp_dir, f))
                os.rmdir(tmp_dir)
            except Exception:
                pass
    
    def _add_audio_to_video_local(self, video_url: str, audio_url: str) -> Optional[str]:
        """Add audio to video using local FFmpeg."""
        if not self._ffmpeg_available() or not self.s3_service:
            return None
        tmp_dir = tempfile.mkdtemp()
        try:
            vid_path = os.path.join(tmp_dir, "video.mp4")
            aud_path = os.path.join(tmp_dir, "audio.mp3")
            out_path = os.path.join(tmp_dir, "out.mp4")
            for url, path in [(video_url, vid_path), (audio_url, aud_path)]:
                r = requests.get(url, timeout=120, stream=True)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            subprocess.run([
                "ffmpeg", "-y", "-i", vid_path, "-i", aud_path,
                "-c:v", "copy", "-c:a", "aac", "-shortest", out_path
            ], check=True, capture_output=True, timeout=300)
            if not os.path.isfile(out_path):
                return None
            key = f"rendi_fallback/audio_{int(time.time())}_{os.getpid()}.mp4"
            return self.s3_service.upload_video_from_path(out_path, key)
        except Exception as e:
            logger.warning(f"⚠️ Local add_audio fallback failed: {e}")
            return None
        finally:
            try:
                for f in ["video.mp4", "audio.mp3", "out.mp4"]:
                    p = os.path.join(tmp_dir, f)
                    if os.path.isfile(p):
                        os.remove(p)
                os.rmdir(tmp_dir)
            except Exception:
                pass
    
    def detect_scenes_cloud(self, video_url: str, threshold: float = 0.1) -> List[float]:
        """Detect scene changes using Rendi.dev cloud FFmpeg.
        
        Args:
            video_url: URL of the video to analyze.
            threshold: Scene change detection threshold (0-1, lower = more sensitive).
            
        Returns:
            List of timestamps (in seconds) where scenes start.
        """
        try:
            logger.info(f"🌐 Detecting scenes via Rendi.dev cloud (threshold={threshold})...")
            
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            
            # Use FFmpeg's scene detection filter with metadata output
            # Write scene timestamps to a text file using the metadata filter
            # Note: comma needs to be escaped in the select filter
            ffmpeg_command = (
                f"-i {{{{in_1}}}} "
                f"-vf \"select=gt(scene\\,{threshold}),metadata=mode=print:file={{{{out_1}}}}\" "
                f"-an -f null -"
            )
            
            payload = {
                "input_files": {"in_1": video_url},
                "output_files": {"out_1": "scene_metadata.txt"},
                "ffmpeg_command": ffmpeg_command,
                "max_command_run_seconds": 300,
                "vcpu_count": 4
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "command_id" in result:
                command_id = result["command_id"]
                logger.info(f"✅ Scene detection task created: {command_id}")
                
                # Wait for completion and get scene data
                return self._wait_for_scene_detection(command_id)
            else:
                logger.error(f"❌ Rendi scene detection error: {result}")
                return [0.0]
                
        except Exception as e:
            logger.error(f"❌ Error detecting scenes via cloud: {e}")
            return [0.0]
    
    def _wait_for_scene_detection(self, command_id: str, timeout: int = 300) -> List[float]:
        """Wait for scene detection to complete and parse results.
        
        Args:
            command_id: Command ID to poll.
            timeout: Maximum wait time in seconds.
            
        Returns:
            List of scene timestamps.
        """
        import re
        start_time = time.time()
        check_interval = 5
        
        while time.time() - start_time < timeout:
            try:
                url = f"{self.base_url}/v1/commands/{command_id}"
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                status = result.get("status", "").upper()
                
                if status == "SUCCESS":
                    timestamps = [0.0]  # Always start with 0
                    
                    # Get the output file URL (scene_metadata.txt)
                    output_files = result.get("output_files", {})
                    scene_info_url = None
                    
                    if "out_1" in output_files:
                        scene_info_url = output_files["out_1"].get("storage_url")
                    
                    # Download and parse the scene_metadata.txt file
                    if scene_info_url:
                        try:
                            logger.info(f"📥 Downloading scene detection results...")
                            file_response = requests.get(scene_info_url, timeout=60)
                            file_response.raise_for_status()
                            scene_info_content = file_response.text
                            
                            logger.info(f"📄 Scene metadata file content length: {len(scene_info_content)} chars")
                            
                            # Parse metadata print output for pts_time values
                            # metadata=print outputs: frame:0    pts:0    pts_time:0.000000
                            # Also handles showinfo format: [Parsed...] n:0 pts:0 pts_time:0.000000
                            pts_pattern = r"pts_time[=:\s]+(\d+\.?\d*)"
                            matches = re.findall(pts_pattern, scene_info_content)
                            
                            for match in matches:
                                ts = float(match)
                                if ts not in timestamps and ts > 0:
                                    timestamps.append(ts)
                            
                            logger.info(f"📊 Found {len(matches)} scene change entries in metadata file")
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Could not download/parse scene info file: {e}")
                    
                    timestamps.sort()
                    logger.info(f"✅ Cloud scene detection complete: {len(timestamps)} scenes detected")
                    return timestamps[:config.MAX_SCENES]
                    
                elif status == "FAILED":
                    error_msg = result.get("error_message", "Unknown error")
                    logger.error(f"❌ Scene detection failed: {error_msg}")
                    return [0.0]
                    
            except Exception as e:
                logger.error(f"❌ Error polling scene detection: {e}")
            
            time.sleep(check_interval)
        
        logger.error("❌ Scene detection timeout")
        return [0.0]
    
    def get_video_duration_cloud(self, video_url: str) -> float:
        """Get video duration using Rendi.dev cloud FFmpeg. Falls back to local ffprobe on failure."""
        try:
            logger.info("🌐 Getting video duration via Rendi.dev cloud...")
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            ffmpeg_command = "-i {{in_1}} -c copy {{out_1}}"
            payload = {
                "input_files": {"in_1": video_url},
                "output_files": {"out_1": "duration_check.mp4"},
                "ffmpeg_command": ffmpeg_command,
                "max_command_run_seconds": 120,
                "vcpu_count": 2
            }
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if "command_id" in result:
                command_id = result["command_id"]
                start_time = time.time()
                while time.time() - start_time < 120:
                    check_url = f"{self.base_url}/v1/commands/{command_id}"
                    check_response = requests.get(check_url, headers=self.headers, timeout=30)
                    check_response.raise_for_status()
                    check_result = check_response.json()
                    status = check_result.get("status", "").upper()
                    if status == "SUCCESS":
                        output_files = check_result.get("output_files", {})
                        if "out_1" in output_files:
                            duration = output_files["out_1"].get("duration")
                            if duration:
                                logger.info(f"✅ Video duration: {duration:.2f}s")
                                return float(duration)
                        break
                    elif status == "FAILED":
                        break
                    time.sleep(3)
        except Exception as e:
            logger.warning(f"⚠️ Rendi video duration failed ({e}), trying local ffprobe...")
            if self._ffmpeg_available():
                dur = self._get_duration_ffprobe(video_url, is_video=True)
                if dur is not None:
                    logger.info(f"✅ Video duration (local): {dur:.2f}s")
                    return dur
            logger.error(f"❌ Error getting video duration: {e}")
        logger.warning("⚠️ Could not determine video duration, using default 30s")
        return 30.0
    
    def get_audio_duration_cloud(self, audio_url: str) -> float:
        """Get audio duration using Rendi.dev cloud FFmpeg. Falls back to local ffprobe on failure."""
        try:
            logger.info("🌐 Getting audio duration via Rendi.dev cloud...")
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            ffmpeg_command = "-i {{in_1}} -c copy {{out_1}}"
            payload = {
                "input_files": {"in_1": audio_url},
                "output_files": {"out_1": "duration_check.mp3"},
                "ffmpeg_command": ffmpeg_command,
                "max_command_run_seconds": 60,
                "vcpu_count": 2
            }
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if "command_id" in result:
                command_id = result["command_id"]
                start_time = time.time()
                while time.time() - start_time < 60:
                    check_url = f"{self.base_url}/v1/commands/{command_id}"
                    check_response = requests.get(check_url, headers=self.headers, timeout=30)
                    check_response.raise_for_status()
                    check_result = check_response.json()
                    status = check_result.get("status", "").upper()
                    if status == "SUCCESS":
                        output_files = check_result.get("output_files", {})
                        if "out_1" in output_files:
                            duration = output_files["out_1"].get("duration")
                            if duration:
                                logger.info(f"✅ Audio duration: {duration:.2f}s")
                                return float(duration)
                        break
                    elif status == "FAILED":
                        break
                    time.sleep(2)
        except Exception as e:
            logger.warning(f"⚠️ Rendi audio duration failed ({e}), trying local ffprobe...")
            if self._ffmpeg_available():
                dur = self._get_duration_ffprobe(audio_url, is_video=False)
                if dur is not None:
                    logger.info(f"✅ Audio duration (local): {dur:.2f}s")
                    return dur
            logger.error(f"❌ Error getting audio duration: {e}")
        return 0.0
    
    def loop_video_to_duration(self, video_url: str, target_duration: float) -> Optional[str]:
        """Loop a video to reach a target duration using Rendi.dev.
        
        The video will be looped as many times as needed to reach/exceed the target duration.
        
        Args:
            video_url: URL of the video to loop.
            target_duration: Target duration in seconds.
            
        Returns:
            URL of the looped video, or None if failed.
        """
        try:
            logger.info(f"🔁 Looping video to reach {target_duration:.2f}s...")
            
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            
            # Use stream_loop to loop the video, then trim to exact duration
            # -stream_loop -1 loops indefinitely, -t limits to target duration
            ffmpeg_command = (
                f"-stream_loop -1 -i {{{{in_1}}}} -t {target_duration:.3f} "
                f"-c:v libx264 -preset fast -crf 18 "
                f"-c:a aac -b:a 128k "
                f"-movflags +faststart "
                f"{{{{out_1}}}}"
            )
            
            payload = {
                "input_files": {"in_1": video_url},
                "output_files": {"out_1": "looped_video.mp4"},
                "ffmpeg_command": ffmpeg_command,
                "max_command_run_seconds": 300,
                "vcpu_count": 4
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "command_id" in result:
                command_id = result["command_id"]
                logger.info(f"✅ Loop task created: {command_id}")
                return self._wait_for_command(command_id)
            else:
                logger.error(f"❌ Rendi loop error: {result}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error looping video: {e}")
            return None
    
    def trim_video(self, video_url: str, duration: float) -> Optional[str]:
        """Trim a video to a specific duration using Rendi.dev. Falls back to local FFmpeg on failure."""
        try:
            logger.info(f"✂️ Trimming video to {duration:.2f}s (with re-encode for clean cuts)...")
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            ffmpeg_command = (
                f"-i {{{{in_1}}}} -t {duration:.3f} "
                f"-c:v libx264 -preset fast -crf 18 "
                f"-c:a aac -b:a 128k "
                f"-movflags +faststart "
                f"{{{{out_1}}}}"
            )
            payload = {
                "input_files": {"in_1": video_url},
                "output_files": {"out_1": "trimmed_video.mp4"},
                "ffmpeg_command": ffmpeg_command,
                "max_command_run_seconds": 180,
                "vcpu_count": 4
            }
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if "command_id" in result:
                command_id = result["command_id"]
                logger.info(f"✅ Trim task created: {command_id}")
                return self._wait_for_command(command_id)
            return None
        except Exception as e:
            logger.warning(f"⚠️ Rendi trim failed ({e}), trying local FFmpeg...")
            out = self._trim_video_local(video_url, duration)
            if out:
                logger.info("✅ Trim completed with local FFmpeg")
            else:
                logger.error(f"❌ Error trimming video: {e}")
            return out
    
    def slow_motion_video(self, video_url: str, speed_factor: float, target_duration: float = None) -> Optional[str]:
        """Apply slow motion to a video using Rendi.dev.
        
        Slows down the video by the given factor to extend its duration.
        Uses setpts filter for smooth slow motion effect.
        Handles videos with or without audio tracks.
        
        Args:
            video_url: URL of the video to slow down.
            speed_factor: Speed multiplier (0.5 = half speed/2x duration, 0.8 = 80% speed/1.25x duration).
                         Should be between 0.5 and 1.0 for subtle slow motion.
            target_duration: Optional target duration. If provided, will trim to exact duration after slowing.
            
        Returns:
            URL of the slowed video, or None if failed.
        """
        try:
            # Clamp speed factor to reasonable range (0.5 to 1.0 for slow motion)
            speed_factor = max(0.5, min(1.0, speed_factor))
            
            if speed_factor >= 0.99:
                logger.info(f"⏸️ Speed factor {speed_factor:.2f} is too close to 1.0, skipping slow motion")
                return video_url
            
            slowdown_percent = (1.0 - speed_factor) * 100
            logger.info(f"⏸️ Applying slow motion ({slowdown_percent:.0f}% slower, speed={speed_factor:.2f}x)...")
            
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            
            # Use setpts filter for smooth slow motion
            # setpts=PTS/speed_factor slows down video (e.g., PTS/0.8 = 25% slower)
            pts_factor = 1.0 / speed_factor  # e.g., 0.8 speed -> 1.25 PTS multiplier
            
            # Video-only slow motion (no audio processing - audio will be added later)
            # This avoids the "Stream specifier ':a' matches no streams" error
            if target_duration:
                # Slow down and trim to exact target duration
                ffmpeg_command = (
                    f"-i {{{{in_1}}}} "
                    f"-vf \"setpts={pts_factor:.4f}*PTS\" "
                    f"-t {target_duration:.3f} "
                    f"-an "  # Remove audio (will be added back in next step)
                    f"-c:v libx264 -preset fast -crf 18 "
                    f"-movflags +faststart "
                    f"{{{{out_1}}}}"
                )
            else:
                # Just slow down without trimming
                ffmpeg_command = (
                    f"-i {{{{in_1}}}} "
                    f"-vf \"setpts={pts_factor:.4f}*PTS\" "
                    f"-an "  # Remove audio (will be added back in next step)
                    f"-c:v libx264 -preset fast -crf 18 "
                    f"-movflags +faststart "
                    f"{{{{out_1}}}}"
                )
            
            payload = {
                "input_files": {"in_1": video_url},
                "output_files": {"out_1": "slowmo_video.mp4"},
                "ffmpeg_command": ffmpeg_command,
                "max_command_run_seconds": 300,
                "vcpu_count": 4
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "command_id" in result:
                command_id = result["command_id"]
                logger.info(f"✅ Slow motion task created: {command_id}")
                return self._wait_for_command(command_id)
            else:
                logger.error(f"❌ Rendi slow motion error: {result}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error applying slow motion: {e}")
            return None
    
    def trim_videos_batch(self, video_durations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trim multiple videos to their target durations in PARALLEL.
        
        Args:
            video_durations: List of dicts with 'video_url' and 'duration' keys.
            
        Returns:
            List of dicts with 'video_url' and 'duration' keys (trimmed URLs with their durations).
        """
        if not video_durations:
            return []
        
        # Filter out any items that have image_url but no video_url (only trim videos, not images)
        filtered_video_durations = []
        for item in video_durations:
            if isinstance(item, dict):
                # Only include if it has video_url (not just image_url)
                if item.get("video_url"):
                    filtered_video_durations.append(item)
                elif item.get("image_url"):
                    # Skip images - we only trim videos
                    logger.warning(f"⚠️ Skipping image in trim_videos_batch (only videos should be trimmed): {item.get('image_url', '')[:60]}...")
            else:
                # If it's a string, assume it's a video URL
                filtered_video_durations.append(item)
        
        if not filtered_video_durations:
            logger.warning("⚠️ No video URLs found for trimming (only images were provided)")
            return []
        
        logger.info(f"✂️ Trimming {len(filtered_video_durations)} videos in PARALLEL...")
        
        # Add index to track original order
        items_with_index = [
            {"index": i, "video_url": item.get("video_url"), "duration": item.get("duration", 5.0)}
            for i, item in enumerate(filtered_video_durations)
        ]
        
        def trim_single(item: Dict[str, Any]) -> Dict[str, Any]:
            """Trim or loop a single video to match target duration."""
            video_url = item.get("video_url")
            target_duration = item.get("duration", 5.0)
            idx = item.get("index", 0)
            
            if not video_url:
                return {
                    "video_url": None,
                    "duration": target_duration,
                    "index": idx,
                    "success": False
                }
            
            try:
                # Get actual video duration
                actual_duration = self.get_video_duration_cloud(video_url)
                
                if actual_duration <= 0:
                    # Fallback: assume video is 10 seconds (typical Runway/Kling output)
                    actual_duration = 10.0
                    logger.warning(f"⚠️ Scene {idx + 1}: Could not get video duration, assuming {actual_duration}s")
                
                logger.info(f"   Scene {idx + 1}: actual={actual_duration:.2f}s, target={target_duration:.2f}s")
                
                # IMPORTANT: Never loop videos - looping causes jarring jumps/repeats
                # Use slow motion to extend if needed (up to 1.5x duration increase)
                # Only trim if video is longer than needed
                if target_duration < actual_duration:
                    # Video is longer than needed - trim it
                    logger.info(f"   Scene {idx + 1}: Trimming video from {actual_duration:.2f}s to {target_duration:.2f}s")
                    final_url = self.trim_video(video_url, target_duration)
                    final_duration = target_duration
                elif target_duration > actual_duration:
                    # Video is shorter than target - use slow motion if within reasonable bounds
                    # Max slow motion: 2x duration (speed factor 0.5)
                    duration_ratio = target_duration / actual_duration
                    max_slowdown = 2.0  # Maximum 100% slower (2x duration) (1.5x duration)
                    
                    if duration_ratio <= max_slowdown:
                        # Apply slow motion to match target duration
                        speed_factor = actual_duration / target_duration  # e.g., 5s/7s = 0.71
                        logger.info(f"   Scene {idx + 1}: Applying slow motion ({(1-speed_factor)*100:.0f}% slower) to extend from {actual_duration:.2f}s to {target_duration:.2f}s")
                        
                        slowmo_url = self.slow_motion_video(
                            video_url=video_url,
                            speed_factor=speed_factor,
                            target_duration=target_duration
                        )
                        
                        if slowmo_url:
                            final_url = slowmo_url
                            final_duration = target_duration
                        else:
                            # Slow motion failed, use original
                            logger.warning(f"   ⚠️ Scene {idx + 1}: Slow motion failed, using original video")
                            final_url = video_url
                            final_duration = actual_duration
                    else:
                        # Too much slowdown needed (would look unnatural), use as-is
                        logger.info(f"   Scene {idx + 1}: Video too short for slow motion ({duration_ratio:.2f}x needed > {max_slowdown:.1f}x max), using as-is ({actual_duration:.2f}s)")
                        final_url = video_url
                        final_duration = actual_duration
                else:
                    # Duration matches exactly
                    logger.info(f"   Scene {idx + 1}: Duration matches ({actual_duration:.2f}s), using as-is")
                    final_url = video_url
                    final_duration = actual_duration
                
                if final_url:
                    return {
                        "video_url": final_url,
                        "duration": final_duration,  # Return actual duration used
                        "index": idx,
                        "success": True
                    }
                else:
                    logger.warning(f"⚠️ Scene {idx + 1}: Processing failed, using original")
                    return {
                        "video_url": video_url,
                        "duration": actual_duration,  # Use actual duration of original video
                        "index": idx,
                        "success": False
                    }
            except Exception as e:
                logger.error(f"❌ Scene {idx + 1}: Error processing video: {e}")
                # Use original if processing fails - assume 10s (typical Runway/Kling output)
                return {
                    "video_url": video_url,
                    "duration": 10.0,  # Fallback to typical Runway/Kling output duration
                    "index": idx,
                    "success": False
                }
        
        trimmed_results = []
        
        # Use ThreadPoolExecutor for parallel trimming
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(trim_single, item): item for item in items_with_index}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result and result.get("video_url"):
                        trimmed_results.append(result)
                        status = "✅" if result.get("success") else "⚠️"
                        logger.info(f"   {status} Scene {result['index'] + 1}: trimmed to {result['duration']:.2f}s")
                except Exception as e:
                    item = futures[future]
                    logger.error(f"   ❌ Scene {item['index'] + 1}: trim failed - {e}")
                    # Use original on error
                    trimmed_results.append({
                        "video_url": item.get("video_url"),
                        "duration": item.get("duration", 5.0),
                        "index": item.get("index", 0)
                    })
        
        # Sort by original index to maintain order
        trimmed_results.sort(key=lambda x: x.get("index", 0))
        
        # Remove duplicates by video_url to prevent same video appearing twice
        seen_urls = set()
        unique_trimmed_results = []
        for v in trimmed_results:
            video_url = v.get("video_url")
            if video_url and video_url not in seen_urls:
                seen_urls.add(video_url)
                unique_trimmed_results.append(v)
            elif video_url in seen_urls:
                logger.warning(f"⚠️ Duplicate video URL in trimmed results (index {v.get('index', 0)}): {video_url[:60]}... - removing duplicate")
        
        if len(unique_trimmed_results) < len(trimmed_results):
            logger.warning(f"⚠️ Removed {len(trimmed_results) - len(unique_trimmed_results)} duplicate videos from trimmed results")
        
        # Remove index from final output
        trimmed_videos = [
            {"video_url": v["video_url"], "duration": v["duration"]} 
            for v in unique_trimmed_results
        ]
        
        logger.info(f"✅ Parallel trimming complete: {len(trimmed_videos)} videos")
        
        return trimmed_videos
    
    def concatenate_videos(
        self, 
        video_data: List[Dict[str, Any]], 
        use_transitions: bool = False
    ) -> Optional[str]:
        """Concatenate multiple videos into one.
        
        Args:
            video_data: List of dicts with 'video_url' and 'duration' keys.
            use_transitions: If True, use xfade transitions (can cause issues).
                           If False, use simple concat (more reliable).
            
        Returns:
            URL of the concatenated video, or None if failed.
        """
        try:
            if not video_data:
                logger.error("❌ No video data provided for concatenation")
                return None
            
            # Extract URLs and durations - ONLY videos, NOT images
            # Filter out any items that have image_url but no video_url
            filtered_video_data = []
            for item in video_data:
                if isinstance(item, dict):
                    # Only include if it has video_url (not just image_url)
                    if item.get("video_url"):
                        filtered_video_data.append(item)
                    elif item.get("image_url"):
                        # Skip images - we only want videos for concatenation
                        logger.warning(f"⚠️ Skipping image URL in concatenation (only videos should be concatenated): {item.get('image_url', '')[:60]}...")
                else:
                    # If it's a string, assume it's a video URL
                    filtered_video_data.append(item)
            
            if not filtered_video_data:
                logger.error("❌ No video URLs found for concatenation (only images were provided)")
                return None
            
            video_urls = [item["video_url"] if isinstance(item, dict) else item for item in filtered_video_data]
            durations = [item.get("duration", 5.0) if isinstance(item, dict) else 5.0 for item in filtered_video_data]
            
            num_videos = len(video_urls)
            logger.info(f"🎬 Concatenating {num_videos} videos with Rendi (transitions={use_transitions})...")
            for i, dur in enumerate(durations):
                logger.info(f"   Video {i+1}: {dur:.2f}s")
            
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            
            # Build input files dict
            input_files = {f"in_{i+1}": video_url for i, video_url in enumerate(video_urls)}
            
            if use_transitions and num_videos > 1:
                # Use xfade transitions with correct offsets based on actual durations
                return self._concatenate_with_transitions(video_urls, durations)
            else:
                # Use simple concat demuxer (more reliable, no freezing issues)
                return self._concatenate_simple(video_urls)
                
        except Exception as e:
            logger.error(f"❌ Error concatenating videos: {e}")
            return None
    
    def _concatenate_simple(self, video_urls: List[str]) -> Optional[str]:
        """Concatenate videos using concat filter (simple, reliable, no repetition).
        
        This method normalizes all videos to the same format before concatenating.
        Ensures clean cuts without repetition or weird transitions.
        """
        try:
            # Remove duplicates to prevent same video appearing twice
            unique_urls = []
            seen_urls = set()
            for url in video_urls:
                if url and url not in seen_urls:
                    unique_urls.append(url)
                    seen_urls.add(url)
                elif url in seen_urls:
                    logger.warning(f"⚠️ Duplicate video URL detected and removed: {url[:60]}...")
            
            num_videos = len(unique_urls)
            if num_videos != len(video_urls):
                logger.warning(f"⚠️ Removed {len(video_urls) - num_videos} duplicate videos from concatenation")
            
            logger.info(f"🎬 Using simple concat method for {num_videos} videos (clean cuts, no repetition)...")
            
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            input_files = {f"in_{i+1}": video_url for i, video_url in enumerate(unique_urls)}
            
            # Normalize all videos to same format, then concatenate
            # This ensures clean cuts without repetition
            filter_parts = []
            
            # First, normalize each video (fps, resolution, pixel format)
            # This ensures all videos are in the same format before concatenation
            for i in range(num_videos):
                filter_parts.append(
                    f"[{i}:v]fps=30,scale=1080:1920:force_original_aspect_ratio=decrease,"
                    f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v{i}]"
                )
            
            # Then concat them together - this creates clean cuts without repetition
            concat_inputs = "".join([f"[v{i}]" for i in range(num_videos)])
            filter_parts.append(f"{concat_inputs}concat=n={num_videos}:v=1:a=0[outv]")
            
            filter_complex = ";".join(filter_parts)
            
            # Build FFmpeg command
            input_args = " ".join([f"-i {{{{in_{i+1}}}}}" for i in range(num_videos)])
            ffmpeg_command = (
                f"{input_args} -filter_complex \"{filter_complex}\" "
                f"-map \"[outv]\" -c:v libx264 -preset fast -crf 18 "
                f"-movflags +faststart {{{{out_1}}}}"
            )
            
            payload = {
                "input_files": input_files,
                "output_files": {"out_1": "concatenated_video.mp4"},
                "ffmpeg_command": ffmpeg_command,
                "max_command_run_seconds": 600,
                "vcpu_count": 8
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if "command_id" in result:
                command_id = result["command_id"]
                logger.info(f"✅ Rendi concat task created: {command_id}")
                return self._wait_for_command(command_id)
            return None
        except Exception as e:
            logger.warning(f"⚠️ Rendi simple concat failed ({e}), trying local FFmpeg...")
            out = self._concatenate_videos_local(unique_urls)
            if out:
                logger.info("✅ Concat completed with local FFmpeg")
            else:
                logger.error(f"❌ Error in simple concat: {e}")
            return out
    
    def _concatenate_with_transitions(
        self, 
        video_urls: List[str], 
        durations: List[float]
    ) -> Optional[str]:
        """Concatenate videos with xfade transitions.
        
        Calculates correct offsets based on actual video durations.
        """
        try:
            num_videos = len(video_urls)
            logger.info(f"🎬 Using xfade transitions for {num_videos} videos...")
            
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            input_files = {f"in_{i+1}": video_url for i, video_url in enumerate(video_urls)}
            
            filter_parts = []
            transition_duration = 0.2  # Very short transition to avoid repetition and smooth cuts
            
            # Normalize each video
            for i in range(num_videos):
                filter_parts.append(
                    f"[{i}:v]fps=30,scale=1080:1920:force_original_aspect_ratio=decrease,"
                    f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v{i}]"
                )
            
            if num_videos == 1:
                filter_parts.append(f"[v0]copy[outv]")
            else:
                # Calculate correct xfade offsets based on actual durations
                # For xfade, offset is the time in the FIRST video where transition starts
                # Transition 0->1: starts at duration[0] - transition_duration
                # Transition 1->2: starts at (duration[0] - transition_duration) + (duration[1] - transition_duration)
                # etc.
                
                # First transition: offset = duration of first video - transition_duration
                cumulative_time = durations[0] - transition_duration
                
                xfade_expr = f"[v0][v1]xfade=transition=fade:duration={transition_duration}:offset={cumulative_time:.3f}[tmp1]"
                filter_parts.append(xfade_expr)
                
                # For subsequent transitions, we need to calculate the offset in the accumulated video
                # Each transition starts at: previous cumulative time + (previous video duration - transition_duration)
                for i in range(2, num_videos):
                    # The offset is the time in the accumulated video (tmp[i-1]) where transition starts
                    # This is: cumulative time of all previous videos minus all previous transitions
                    cumulative_time += durations[i-1] - transition_duration
                    
                    if i == num_videos - 1:
                        xfade_expr = f"[tmp{i-1}][v{i}]xfade=transition=fade:duration={transition_duration}:offset={cumulative_time:.3f}[outv]"
                    else:
                        xfade_expr = f"[tmp{i-1}][v{i}]xfade=transition=fade:duration={transition_duration}:offset={cumulative_time:.3f}[tmp{i}]"
                    filter_parts.append(xfade_expr)
            
            filter_complex = ";".join(filter_parts)
            
            input_args = " ".join([f"-i {{{{in_{i+1}}}}}" for i in range(num_videos)])
            ffmpeg_command = (
                f"{input_args} -filter_complex \"{filter_complex}\" "
                f"-map \"[outv]\" -c:v libx264 -preset fast -crf 18 "
                f"-movflags +faststart {{{{out_1}}}}"
            )
            
            payload = {
                "input_files": input_files,
                "output_files": {"out_1": "concatenated_video.mp4"},
                "ffmpeg_command": ffmpeg_command,
                "max_command_run_seconds": 600,
                "vcpu_count": 8
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if "command_id" in result:
                command_id = result["command_id"]
                logger.info(f"✅ Rendi xfade task created: {command_id}")
                return self._wait_for_command(command_id)
            return None
        except Exception as e:
            logger.warning(f"⚠️ Rendi xfade concat failed ({e}), trying local FFmpeg...")
            out = self._concatenate_videos_local(video_urls)
            if out:
                logger.info("✅ Concat (xfade fallback) completed with local FFmpeg")
            else:
                logger.error(f"❌ Error in xfade concat: {e}")
            return out
    
    def add_audio_to_video(self, video_url: str, audio_url: str) -> Optional[str]:
        """Add audio track to a video. Falls back to local FFmpeg on Rendi failure."""
        try:
            logger.info(f"🎬 Adding audio to video with Rendi...")
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            ffmpeg_command = "-i {{in_1}} -i {{in_2}} -c:v copy -c:a aac -shortest {{out_1}}"
            payload = {
                "ffmpeg_command": ffmpeg_command,
                "input_files": {"in_1": video_url, "in_2": audio_url},
                "output_files": {"out_1": "video_with_audio.mp4"},
                "vcpu_count": 4,
                "max_command_run_seconds": 300
            }
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if "command_id" in result:
                command_id = result["command_id"]
                logger.info(f"✅ Rendi audio task created: {command_id}")
                return self._wait_for_command(command_id)
            return None
        except Exception as e:
            logger.warning(f"⚠️ Rendi add_audio failed ({e}), trying local FFmpeg...")
            out = self._add_audio_to_video_local(video_url, audio_url)
            if out:
                logger.info("✅ Add audio completed with local FFmpeg")
            else:
                logger.error(f"❌ Error adding audio to video: {e}")
            return out
    
    def validate_video_has_audio(self, video_url: str) -> bool:
        """Check if a video has an audio track using FFprobe via Rendi.
        
        Args:
            video_url: URL of the video to check.
            
        Returns:
            True if video has audio track, False otherwise.
        """
        try:
            logger.info(f"🔍 Validating video has audio track...")
            
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            
            # Use ffprobe to check for audio streams
            # This command outputs audio stream info if present
            ffprobe_command = (
                "-v quiet -select_streams a:0 -show_entries stream=codec_type "
                "-of default=noprint_wrappers=1:nokey=1 {{in_1}}"
            )
            
            payload = {
                "ffmpeg_command": f"ffprobe {ffprobe_command}",
                "input_files": {
                    "in_1": video_url
                },
                "output_files": {},
                "vcpu_count": 1,
                "max_command_run_seconds": 30
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            # If the command succeeds and returns "audio", the video has audio
            if response.status_code == 200:
                result = response.json()
                # Check if command completed successfully
                if result.get("status") == "completed" or "audio" in str(result).lower():
                    logger.info("✅ Video has audio track")
                    return True
            
            # Alternative check: Try to get video duration with audio info
            # If we can't probe directly, assume video has audio if it was created from audio combination
            logger.info("✅ Assuming video has audio (probe inconclusive)")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Could not validate audio track: {e}, assuming video has audio")
            # In case of error, assume video has audio to avoid blocking
            return True
    
    def add_background_music_to_video(
        self, 
        video_url: str, 
        music_url: str, 
        music_volume: float = 0.35
    ) -> Optional[str]:
        """Add background music to a video that already has audio (voice-over).
        
        This overlays the music track on top of existing video audio.
        
        Args:
            video_url: URL of the video (already has voice audio).
            music_url: URL of the background music.
            music_volume: Volume level for music (0.0 to 1.0). Default 0.25.
            
        Returns:
            URL of the video with mixed audio, or None if failed.
        """
        try:
            logger.info(f"🎵 Adding background music to video (music_volume={music_volume})...")
            
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            
            # FFmpeg command to overlay background music on video's existing audio
            # The video audio (voice) stays at full volume, music is reduced
            ffmpeg_command = (
                f"-i {{{{in_1}}}} -i {{{{in_2}}}} "
                f"-filter_complex \"[0:a]volume=1.0[voice];[1:a]volume={music_volume}[music];[voice][music]amix=inputs=2:duration=first:dropout_transition=2[mixed]\" "
                f"-map 0:v -map \"[mixed]\" -c:v copy -c:a aac -b:a 192k -shortest {{{{out_1}}}}"
            )
            
            payload = {
                "ffmpeg_command": ffmpeg_command,
                "input_files": {
                    "in_1": video_url,  # Video with voice
                    "in_2": music_url   # Background music
                },
                "output_files": {
                    "out_1": "video_with_music.mp4"
                },
                "vcpu_count": 4,
                "max_command_run_seconds": 300
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "command_id" in result:
                command_id = result["command_id"]
                logger.info(f"✅ Rendi background music task created: {command_id}")
                return self._wait_for_command(command_id)
            else:
                logger.error(f"❌ Rendi API error: {result}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error adding background music: {e}")
            return None
    
    def overlay_cta_on_video(
        self,
        video_url: str,
        cta_image_url: str,
        position: str = "bottom_center"
    ) -> Optional[str]:
        """Overlay a CTA button image on a video with floating animation.
        
        Args:
            video_url: URL of the video to overlay on.
            cta_image_url: URL of the CTA button image (PNG with transparency).
            position: Position of the overlay ('bottom_center', 'center', etc.).
            
        Returns:
            URL of the video with CTA overlay, or None if failed.
        """
        try:
            logger.info(f"🔘 Overlaying CTA button on video with floating effect (position: {position})...")
            
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            
            # Floating animation: button moves up and down using sine wave
            # Base Y position at bottom-center (10% from bottom)
            # sin(t*3) creates smooth oscillation, *10 is the amplitude (10 pixels up/down)
            base_y = "main_h-overlay_h-main_h*0.10"
            float_offset = "sin(t*3)*10"  # 3 cycles per second, 10 pixels amplitude
            
            if position == "bottom_center":
                overlay_filter = f"overlay=x=(main_w-overlay_w)/2:y={base_y}+{float_offset}"
            elif position == "center":
                overlay_filter = f"overlay=x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2+{float_offset}"
            elif position == "bottom_right":
                overlay_filter = f"overlay=x=main_w-overlay_w-main_w*0.05:y={base_y}+{float_offset}"
            else:
                overlay_filter = f"overlay=x=(main_w-overlay_w)/2:y={base_y}+{float_offset}"
            
            # FFmpeg command to overlay PNG image on video with floating animation
            # Scale the overlay to be ~50% of video width while maintaining aspect ratio
            ffmpeg_command = (
                f"-i {{{{in_1}}}} -i {{{{in_2}}}} "
                f"-filter_complex \"[1:v]scale=iw*0.5:-1[scaled];[0:v][scaled]{overlay_filter}[out]\" "
                f"-map \"[out]\" -map 0:a? -c:v libx264 -preset fast -crf 18 -c:a copy {{{{out_1}}}}"
            )
            
            payload = {
                "ffmpeg_command": ffmpeg_command,
                "input_files": {
                    "in_1": video_url,      # Video
                    "in_2": cta_image_url   # CTA button image
                },
                "output_files": {
                    "out_1": "video_with_cta.mp4"
                },
                "vcpu_count": 4,
                "max_command_run_seconds": 300
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "command_id" in result:
                command_id = result["command_id"]
                logger.info(f"✅ Rendi CTA overlay task created: {command_id}")
                return self._wait_for_command(command_id)
            else:
                logger.error(f"❌ Rendi API error: {result}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error overlaying CTA on video: {e}")
            return None
    
    def overlay_cta_on_video_timed(
        self,
        video_url: str,
        cta_image_url: str,
        position: str = "bottom_center",
        start_time: float = 0.0,
        end_time: float = None
    ) -> Optional[str]:
        """Overlay a CTA button image on a video only during a specific time range.
        
        Args:
            video_url: URL of the video to overlay on.
            cta_image_url: URL of the CTA button image (PNG with transparency).
            position: Position of the overlay ('bottom_center', 'center', etc.).
            start_time: Start time in seconds when CTA should appear.
            end_time: End time in seconds when CTA should disappear (None = end of video).
            
        Returns:
            URL of the video with CTA overlay, or None if failed.
        """
        try:
            logger.info(f"🔘 Overlaying CTA button on video (position: {position}, start: {start_time:.1f}s)...")
            
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            
            # Floating animation: button moves up and down using sine wave
            base_y = "main_h-overlay_h-main_h*0.10"
            float_offset = "sin(t*3)*10"
            
            if position == "bottom_center":
                overlay_filter = f"overlay=x=(main_w-overlay_w)/2:y={base_y}+{float_offset}"
            elif position == "center":
                overlay_filter = f"overlay=x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2+{float_offset}"
            elif position == "bottom_right":
                overlay_filter = f"overlay=x=main_w-overlay_w-main_w*0.05:y={base_y}+{float_offset}"
            else:
                overlay_filter = f"overlay=x=(main_w-overlay_w)/2:y={base_y}+{float_offset}"
            
            # Add timing condition: enable overlay only between start_time and end_time
            # Using 'between(t,start,end)' function in FFmpeg
            if end_time is not None:
                timing_condition = f":enable='between(t,{start_time},{end_time})'"
            else:
                timing_condition = f":enable='gte(t,{start_time})'"
            
            overlay_filter += timing_condition
            
            # FFmpeg command with timed overlay
            ffmpeg_command = (
                f"-i {{{{in_1}}}} -i {{{{in_2}}}} "
                f"-filter_complex \"[1:v]scale=iw*0.5:-1[scaled];[0:v][scaled]{overlay_filter}[out]\" "
                f"-map \"[out]\" -map 0:a? -c:v libx264 -preset fast -crf 18 -c:a copy {{{{out_1}}}}"
            )
            
            payload = {
                "ffmpeg_command": ffmpeg_command,
                "input_files": {
                    "in_1": video_url,
                    "in_2": cta_image_url
                },
                "output_files": {
                    "out_1": "video_with_cta_timed.mp4"
                },
                "vcpu_count": 4,
                "max_command_run_seconds": 300
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "command_id" in result:
                command_id = result["command_id"]
                logger.info(f"✅ Rendi CTA timed overlay task created: {command_id}")
                return self._wait_for_command(command_id)
            else:
                logger.error(f"❌ Rendi API error: {result}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error overlaying timed CTA on video: {e}")
            return None
    
    def add_text_overlay_to_video(
        self,
        video_url: str,
        text: str,
        start_time: float = 0.0,
        end_time: float = None,
        position: str = "bottom_center",
        font_size: int = 60,
        font_color: str = "white",
        background_color: str = "black@0.5"
    ) -> Optional[str]:
        """Add text overlay to video using FFmpeg drawtext filter.
        
        Args:
            video_url: URL of the video to add text to.
            text: Text to display.
            start_time: Start time in seconds when text should appear.
            end_time: End time in seconds when text should disappear (None = end of video).
            position: Position of the text ('bottom_center', 'center', 'top_center', etc.).
            font_size: Font size in pixels.
            font_color: Font color (e.g., 'white', 'black', '#FFFFFF').
            background_color: Background color for text box (e.g., 'black@0.5' for semi-transparent black).
            
        Returns:
            URL of the video with text overlay, or None if failed.
        """
        try:
            logger.info(f"📝 Adding text overlay to video: '{text[:50]}...' (position: {position}, start: {start_time:.1f}s)")
            
            url = f"{self.base_url}/v1/run-ffmpeg-command"
            
            # Escape special characters in text for FFmpeg
            escaped_text = text.replace("'", "\\'").replace(":", "\\:").replace("\\", "\\\\")
            
            # Position calculation
            if position == "bottom_center":
                x_expr = "(main_w-text_w)/2"
                y_expr = "main_h-text_h-50"
            elif position == "center":
                x_expr = "(main_w-text_w)/2"
                y_expr = "(main_h-text_h)/2"
            elif position == "top_center":
                x_expr = "(main_w-text_w)/2"
                y_expr = "50"
            elif position == "bottom_left":
                x_expr = "50"
                y_expr = "main_h-text_h-50"
            elif position == "bottom_right":
                x_expr = "main_w-text_w-50"
                y_expr = "main_h-text_h-50"
            else:
                x_expr = "(main_w-text_w)/2"
                y_expr = "main_h-text_h-50"
            
            # Build drawtext filter
            # Add background box for better readability
            drawtext_filter = (
                f"drawtext=text='{escaped_text}'"
                f":fontsize={font_size}"
                f":fontcolor={font_color}"
                f":box=1:boxcolor={background_color}:boxborderw=10"
                f":x={x_expr}"
                f":y={y_expr}"
            )
            
            # Add timing if specified
            if end_time is not None:
                drawtext_filter += f":enable='between(t,{start_time},{end_time})'"
            elif start_time > 0:
                drawtext_filter += f":enable='gte(t,{start_time})'"
            
            ffmpeg_command = (
                f"-i {{{{in_1}}}} "
                f"-vf \"{drawtext_filter}\" "
                f"-c:v libx264 -preset fast -crf 18 "
                f"-c:a copy "
                f"-movflags +faststart "
                f"{{{{out_1}}}}"
            )
            
            payload = {
                "ffmpeg_command": ffmpeg_command,
                "input_files": {
                    "in_1": video_url
                },
                "output_files": {
                    "out_1": "video_with_text_overlay.mp4"
                },
                "vcpu_count": 4,
                "max_command_run_seconds": 300
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "command_id" in result:
                command_id = result["command_id"]
                logger.info(f"✅ Rendi text overlay task created: {command_id}")
                return self._wait_for_command(command_id)
            else:
                logger.error(f"❌ Rendi API error: {result}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error adding text overlay to video: {e}")
            return None
    
    def _wait_for_command(self, command_id: str, timeout: int = 600) -> Optional[str]:
        """Wait for Rendi command to complete.
        
        Args:
            command_id: Command ID to poll.
            timeout: Maximum wait time in seconds.
            
        Returns:
            URL of the output video, or None if failed/timeout.
        """
        start_time = time.time()
        check_interval = 10
        
        while time.time() - start_time < timeout:
            try:
                url = f"{self.base_url}/v1/commands/{command_id}"
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                status = result.get("status", "").lower()
                
                if status in ["completed", "success"]:
                    output_files = result.get("output_files", {})
                    if "out_1" in output_files and "storage_url" in output_files["out_1"]:
                        video_url = output_files["out_1"]["storage_url"]
                        logger.info(f"✅ Rendi command completed")
                        return video_url
                
                elif status == "failed":
                    error_msg = result.get("error_message", "Unknown error")
                    logger.error(f"❌ Rendi command failed: {error_msg}")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Error polling Rendi command: {e}")
            
            time.sleep(check_interval)
        
        logger.error("❌ Rendi command timeout")
        return None


# =============================================================================
# ELEVENLABS SERVICE
# =============================================================================
class ElevenLabsService:
    """Service for ElevenLabs API interactions."""
    
    def __init__(self, api_key: str, openai_client: OpenAI = None):
        """Initialize ElevenLabs service.
        
        Args:
            api_key: ElevenLabs API key.
            openai_client: Optional OpenAI client for speech detection.
        """
        self.api_key = api_key
        self.base_url = config.ELEVENLABS_BASE_URL
        self.openai_client = openai_client
        logger.info("✅ ElevenLabs client initialized")
    
    def detect_speech_in_audio(self, audio_path: str) -> bool:
        """Detect if audio contains speech/voice-over using OpenAI Whisper.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            True if speech is detected, False if only music/silence.
        """
        try:
            if not self.openai_client:
                logger.warning("⚠️ No OpenAI client for speech detection, assuming speech present")
                return True
            
            logger.info("🔍 Detecting speech in audio...")
            
            with open(audio_path, 'rb') as audio_file:
                # Use Whisper to transcribe - if there's text, there's speech
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                
                # Check if meaningful text was transcribed
                transcript_text = transcript.strip() if transcript else ""
                
                # If transcript is mostly empty or just noise artifacts, no speech
                if len(transcript_text) < 10:  # Less than 10 chars = likely no speech
                    logger.info("🎵 No speech detected - audio is music/ambient only")
                    return False
                else:
                    logger.info(f"🎤 Speech detected: '{transcript_text[:50]}...'")
                    return True
                    
        except Exception as e:
            logger.warning(f"⚠️ Speech detection failed: {e}, assuming speech present")
            return True  # Default to processing voice if detection fails
    
    def voice_changer(
        self, 
        audio_path: str, 
        voice_id: str = None
    ) -> Optional[bytes]:
        """Change voice in audio using speech-to-speech.
        
        Args:
            audio_path: Path to the audio file.
            voice_id: ElevenLabs voice ID (uses default if None).
            
        Returns:
            Audio data as bytes, or None if failed.
        """
        try:
            # Validate voice_id (handles #N/A, empty, etc.)
            voice_id = get_validated_voice_id(voice_id, config.DEFAULT_VOICE_ID)
            logger.info(f"🎤 Changing voice with ElevenLabs (voice_id: {voice_id})...")
            
            url = f"{self.base_url}/speech-to-speech/{voice_id}"
            
            headers = {
                "xi-api-key": self.api_key
            }
            
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'audio': ('audio.mp3', audio_file, 'audio/mpeg')
                }
                
                data = {
                    'model_id': 'eleven_multilingual_sts_v2',
                    'remove_background_noise': 'false'
                }
                
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                    params={'output_format': 'mp3_44100_128'},
                    timeout=300
                )
                response.raise_for_status()
                
                audio_data = response.content
                
                if len(audio_data) > 1000:
                    logger.info(f"✅ Voice changed: {len(audio_data)} bytes")
                    return audio_data
                else:
                    logger.error(f"❌ Voice change failed: audio too small")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error changing voice: {e}")
            return None
    
    def separate_stems(
        self,
        audio_path: str,
        output_dir: str
    ) -> Optional[str]:
        """Separate audio into stems and extract clean vocals.
        
        Uses ElevenLabs Stem Separation API to separate vocals from music.
        
        Args:
            audio_path: Path to the audio file.
            output_dir: Directory to save the extracted vocals.
            
        Returns:
            Path to the clean vocals file, or None if failed.
        """
        try:
            logger.info("🎵 Separating stems with ElevenLabs...")
            
            url = f"{self.base_url}/music/stem-separation"
            
            headers = {
                "xi-api-key": self.api_key
            }
            
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'file': ('audio.mp3', audio_file, 'audio/mpeg')
                }
                
                # Use two_stems_v1 to get vocals and instrumental only
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    params={
                        'output_format': 'mp3_44100_128',
                        'stem_variation_id': 'two_stems_v1'
                    },
                    timeout=600  # Stem separation can take a while
                )
                response.raise_for_status()
                
                # Response is a ZIP file containing the stems
                zip_data = io.BytesIO(response.content)
                
                vocals_path = None
                
                with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                    # List files in the ZIP
                    file_list = zip_ref.namelist()
                    logger.info(f"📦 Stems in ZIP: {file_list}")
                    
                    # Find the vocals stem (usually named 'vocals' or similar)
                    vocals_file = None
                    for filename in file_list:
                        lower_name = filename.lower()
                        if 'vocal' in lower_name or 'voice' in lower_name:
                            vocals_file = filename
                            break
                    
                    if not vocals_file:
                        # If no explicit vocals file, take the first one that's not instrumental
                        for filename in file_list:
                            lower_name = filename.lower()
                            if 'instrumental' not in lower_name and 'music' not in lower_name and 'accomp' not in lower_name:
                                vocals_file = filename
                                break
                    
                    if vocals_file:
                        # Extract vocals to output directory
                        vocals_path = os.path.join(output_dir, "clean_vocals.mp3")
                        with zip_ref.open(vocals_file) as src:
                            with open(vocals_path, 'wb') as dst:
                                dst.write(src.read())
                        logger.info(f"✅ Vocals extracted: {vocals_path}")
                    else:
                        logger.error("❌ Could not find vocals stem in ZIP")
                        return None
                
                return vocals_path
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ Stem separation HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"❌ Error separating stems: {e}")
            return None
    
    def text_to_speech(
        self,
        text: str,
        voice_id: str = None,
        language: str = "en"
    ) -> Optional[bytes]:
        """Generate speech from text using ElevenLabs TTS API.
        
        Args:
            text: Text to convert to speech.
            voice_id: ElevenLabs voice ID (uses default if None).
            language: ISO 639-1 language code for voice selection.
            
        Returns:
            Audio data as bytes, or None if failed.
        """
        try:
            # Validate voice_id (handles #N/A, empty, etc.)
            voice_id = get_validated_voice_id(voice_id, config.DEFAULT_VOICE_ID)
            logger.info(f"🔊 Generating speech with ElevenLabs TTS (voice_id: {voice_id}, language: {language})...")
            
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Use Eleven v3 for all languages (best quality, supports Hebrew)
            model_id = "eleven_v3"
            
            # Voice settings optimized for natural speech
            payload = {
                "text": text,
                "model_id": model_id,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            logger.info(f"🔊 Using ElevenLabs model: {model_id}")
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            audio_bytes = response.content
            logger.info(f"✅ TTS generated: {len(audio_bytes)} bytes")
            return audio_bytes
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ TTS HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"❌ Error generating TTS: {e}")
            return None
    
    def text_to_speech_with_timestamps(
        self,
        text: str,
        voice_id: str = None,
        language: str = "en"
    ) -> Optional[Tuple[bytes, List[Dict]]]:
        """Generate speech from text with word-level timestamps using ElevenLabs.
        
        Uses the /with-timestamps endpoint to get precise character-level timing,
        then converts to word-level segments for subtitle synchronization.
        
        Args:
            text: Text to convert to speech.
            voice_id: ElevenLabs voice ID (uses default if None).
            language: ISO 639-1 language code for voice selection.
            
        Returns:
            Tuple of (audio_bytes, word_segments) or None if failed.
            word_segments is a list of dicts with:
                - text: the word
                - type: "word"
                - start_time: start time in seconds
                - end_time: end time in seconds
        """
        try:
            # Validate voice_id (handles #N/A, empty, etc.)
            voice_id = get_validated_voice_id(voice_id, config.DEFAULT_VOICE_ID)
            logger.info(f"🔊 Generating speech with timestamps (voice_id: {voice_id}, language: {language})...")
            
            url = f"{self.base_url}/text-to-speech/{voice_id}/with-timestamps"
            
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Use Eleven v3 for all languages (best quality, supports Hebrew)
            model_id = "eleven_v3"
            
            # Voice settings optimized for natural speech
            payload = {
                "text": text,
                "model_id": model_id,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            logger.info(f"🔊 Using ElevenLabs model: {model_id} with timestamps")
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            # Parse JSON response with audio and alignment
            data = response.json()
            
            # Decode audio from base64
            audio_base64 = data.get("audio_base64", "")
            audio_bytes = base64.b64decode(audio_base64)
            logger.info(f"✅ TTS generated: {len(audio_bytes)} bytes")
            
            # Extract alignment data
            alignment = data.get("alignment", {})
            characters = alignment.get("characters", [])
            char_starts = alignment.get("character_start_times_seconds", [])
            char_ends = alignment.get("character_end_times_seconds", [])
            
            if not characters or not char_starts or not char_ends:
                logger.warning("⚠️ No alignment data in response, returning audio only")
                return audio_bytes, []
            
            # Convert character-level alignment to word-level segments
            word_segments = self._convert_alignment_to_word_segments(
                characters, char_starts, char_ends
            )
            
            logger.info(f"✅ Extracted {len(word_segments)} word segments from timestamps")
            return audio_bytes, word_segments
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ TTS with timestamps HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"❌ Error generating TTS with timestamps: {e}")
            return None
    
    def _convert_alignment_to_word_segments(
        self,
        characters: List[str],
        char_starts: List[float],
        char_ends: List[float]
    ) -> List[Dict]:
        """Convert character-level alignment to word-level segments.
        
        Groups characters into words by splitting on spaces and punctuation,
        and calculates word start/end times from the character timing.
        
        Args:
            characters: List of characters from ElevenLabs alignment.
            char_starts: List of start times for each character.
            char_ends: List of end times for each character.
            
        Returns:
            List of word segments with text, type, start_time, end_time.
        """
        if len(characters) != len(char_starts) or len(characters) != len(char_ends):
            logger.warning("⚠️ Mismatched alignment arrays")
            return []
        
        word_segments = []
        current_word = ""
        word_start = None
        word_end = None
        
        for i, char in enumerate(characters):
            # Skip if this character is a space or common punctuation that separates words
            if char in " \t\n\r":
                # If we have a word accumulated, save it
                if current_word and word_start is not None:
                    word_segments.append({
                        "text": current_word,
                        "type": "word",
                        "start_time": round(word_start, 3),
                        "end_time": round(word_end, 3)
                    })
                # Reset for next word
                current_word = ""
                word_start = None
                word_end = None
            else:
                # Add character to current word
                if word_start is None:
                    word_start = char_starts[i]
                current_word += char
                word_end = char_ends[i]
        
        # Don't forget the last word
        if current_word and word_start is not None:
            word_segments.append({
                "text": current_word,
                "type": "word",
                "start_time": round(word_start, 3),
                "end_time": round(word_end, 3)
            })
        
        logger.info(f"📝 Converted {len(characters)} characters to {len(word_segments)} words")
        return word_segments
    
    def get_transcript_from_audio(self, audio_path: str) -> Optional[str]:
        """Get transcript from audio using OpenAI Whisper.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Transcript text, or None if failed.
        """
        try:
            if not self.openai_client:
                logger.warning("⚠️ No OpenAI client for transcription")
                return None
            
            logger.info("📝 Transcribing audio with Whisper...")
            
            with open(audio_path, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                
                transcript_text = transcript.strip() if transcript else ""
                
                if transcript_text:
                    logger.info(f"✅ Transcribed: {len(transcript_text)} characters")
                    return transcript_text
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return None
    
    def detect_vo_presence(self, audio_path: str) -> Tuple[bool, Optional[str]]:
        """Detect if audio has meaningful voice-over (speech).
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Tuple of (has_vo: bool, transcript: Optional[str])
            - has_vo: True if meaningful speech detected, False otherwise
            - transcript: The transcript text if speech found, None otherwise
        """
        try:
            transcript = self.get_transcript_from_audio(audio_path)
            
            if not transcript:
                logger.info("🔇 No speech detected in audio (empty transcription)")
                return False, None
            
            # Check if transcript has meaningful content
            # Only filter out if truly no speech (0-1 words might be noise)
            word_count = len(transcript.split())
            
            if word_count < 2:
                logger.info(f"🔇 No meaningful speech detected ({word_count} words) - treating as no VO")
                return False, None
            
            logger.info(f"🎤 Voice-over detected: {word_count} words")
            return True, transcript
            
        except Exception as e:
            logger.error(f"❌ VO detection failed: {e}")
            return False, None
    
    def detect_vo_gender(self, audio_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect the gender of the narrator from audio.
        
        Uses Whisper for transcription and OpenAI to analyze voice characteristics.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Tuple of (gender: Optional[str], transcript: Optional[str])
            - gender: 'm' for male, 'f' for female, None if no VO detected
            - transcript: The transcript text if speech found
        """
        try:
            # First check if there's VO
            has_vo, transcript = self.detect_vo_presence(audio_path)
            
            if not has_vo:
                return None, None
            
            # Use OpenAI to analyze the audio for gender
            # We need to send the audio file directly for voice analysis
            if not self.openai_client:
                logger.warning("⚠️ No OpenAI client for gender detection")
                return None, transcript
            
            logger.info("🔍 Detecting narrator gender with OpenAI...")
            
            # Read audio file and encode to base64
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Determine audio format from file extension
            audio_format = "mp3"
            if audio_path.lower().endswith('.wav'):
                audio_format = "wav"
            elif audio_path.lower().endswith('.m4a'):
                audio_format = "m4a"
            
            # Use GPT-4o-audio to analyze the voice
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-audio-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at identifying voice characteristics. Listen to the audio and determine the gender of the main speaker/narrator. Respond with ONLY a single letter: 'm' for male voice or 'f' for female voice. If you cannot determine the gender or there's no clear narrator, respond with 'u' for unknown."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": audio_base64,
                                        "format": audio_format
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": "What is the gender of the narrator/speaker in this audio? Reply with only 'm' for male or 'f' for female."
                                }
                            ]
                        }
                    ],
                    max_tokens=10
                )
                
                gender_response = response.choices[0].message.content.strip().lower()
                
                # Parse the response
                if 'm' in gender_response and 'f' not in gender_response:
                    gender = 'm'
                elif 'f' in gender_response and 'm' not in gender_response:
                    gender = 'f'
                elif gender_response in ['m', 'f']:
                    gender = gender_response
                else:
                    # Try to extract from longer response
                    if 'male' in gender_response and 'female' not in gender_response:
                        gender = 'm'
                    elif 'female' in gender_response:
                        gender = 'f'
                    else:
                        logger.warning(f"⚠️ Could not parse gender from response: {gender_response}")
                        gender = None
                
                if gender:
                    gender_name = "male" if gender == 'm' else "female"
                    logger.info(f"✅ Detected narrator gender: {gender_name} ({gender})")
                
                return gender, transcript
                
            except Exception as e:
                # Fallback: try using text analysis of transcript
                logger.warning(f"⚠️ Audio gender detection failed: {e}, trying text-based fallback...")
                
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "Based on the transcript and context, try to determine if the speaker is likely male or female. This is a voice-over transcript. Respond with ONLY 'm' for male or 'f' for female. If unsure, respond with 'm' as default."
                            },
                            {
                                "role": "user",
                                "content": f"Transcript: {transcript[:500]}"
                            }
                        ],
                        max_tokens=5
                    )
                    
                    gender = response.choices[0].message.content.strip().lower()
                    if gender not in ['m', 'f']:
                        gender = 'm'  # Default to male if unclear
                    
                    gender_name = "male" if gender == 'm' else "female"
                    logger.info(f"✅ Detected narrator gender (from text): {gender_name} ({gender})")
                    return gender, transcript
                    
                except Exception as e2:
                    logger.error(f"❌ Fallback gender detection also failed: {e2}")
                    return None, transcript
                    
        except Exception as e:
            logger.error(f"❌ Gender detection failed: {e}")
            return None, None


# =============================================================================
# AWS S3 SERVICE
# =============================================================================
class S3Service:
    """Service for AWS S3 operations."""
    
    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        region: str,
        folder_path: str
    ):
        """Initialize S3 service.
        
        Args:
            access_key_id: AWS access key ID.
            secret_access_key: AWS secret access key.
            bucket_name: S3 bucket name.
            region: AWS region.
            folder_path: Folder path within the bucket.
        """
        self.bucket = bucket_name
        self.region = region
        
        # Normalize folder path
        folder_path = folder_path.strip().lstrip('/')
        if folder_path and not folder_path.endswith('/'):
            folder_path += '/'
        self.folder = folder_path
        
        session = boto3.session.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region
        )
        self.client = session.client('s3')
        logger.info("✅ S3 client initialized")
    
    def upload_video_from_url(
        self, 
        source_url: str, 
        key_name: str,
        make_public: bool = True
    ) -> Optional[str]:
        """Upload a video from URL to S3.
        
        Args:
            source_url: URL of the video to upload.
            key_name: Name for the S3 object.
            make_public: Whether to make the object publicly readable.
            
        Returns:
            Public URL of the uploaded video, or None if failed.
        """
        try:
            logger.info(f"📤 Uploading video to S3...")
            
            # Download video
            with requests.get(source_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                body = io.BytesIO()
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        body.write(chunk)
                body.seek(0)
            
            # Build S3 key
            key = f"{self.folder}{key_name}"
            
            # Upload to S3
            extra_args = {"ContentType": "video/mp4"}
            self.client.upload_fileobj(body, self.bucket, key, ExtraArgs=extra_args)
            
            # Set public ACL if requested
            if make_public:
                try:
                    self.client.put_object_acl(
                        Bucket=self.bucket,
                        Key=key,
                        ACL="public-read"
                    )
                except (BotoCoreError, ClientError) as e:
                    logger.warning(f"⚠️ Could not set public ACL: {e}")
            
            # Build public URL
            url = f"https://{self.bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"
            logger.info(f"✅ Uploaded to S3: {url}")
            return url
            
        except Exception as e:
            logger.error(f"❌ S3 upload failed: {e}")
            return None
    
    def upload_video_from_path(
        self,
        local_path: str,
        key_name: str,
        make_public: bool = True
    ) -> Optional[str]:
        """Upload a video from local file path to S3.
        
        Args:
            local_path: Path to the local video file.
            key_name: Name for the S3 object.
            make_public: Whether to make the object publicly readable.
            
        Returns:
            Public URL of the uploaded video, or None if failed.
        """
        try:
            if not os.path.isfile(local_path):
                logger.error(f"❌ Local file not found: {local_path}")
                return None
            logger.info(f"📤 Uploading local video to S3...")
            key = f"{self.folder}{key_name}"
            self.client.upload_file(local_path, self.bucket, key, ExtraArgs={"ContentType": "video/mp4"})
            if make_public:
                try:
                    self.client.put_object_acl(Bucket=self.bucket, Key=key, ACL="public-read")
                except (BotoCoreError, ClientError) as acl_err:
                    logger.warning(f"⚠️ Could not set public ACL: {acl_err}")
            url = f"https://{self.bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"
            logger.info(f"✅ Uploaded to S3: {url[:60]}...")
            return url
        except (BotoCoreError, ClientError, Exception) as e:
            logger.error(f"❌ S3 upload from path failed: {e}")
            return None
    
    def upload_audio_bytes(
        self, 
        audio_data: bytes, 
        key_name: str,
        make_public: bool = True
    ) -> Optional[str]:
        """Upload audio bytes to S3.
        
        Args:
            audio_data: Audio data as bytes.
            key_name: Name for the S3 object.
            make_public: Whether to make the object publicly readable.
            
        Returns:
            Public URL of the uploaded audio, or None if failed.
        """
        try:
            logger.info(f"📤 Uploading audio to S3...")
            
            body = io.BytesIO(audio_data)
            key = f"{self.folder}{key_name}"
            
            extra_args = {"ContentType": "audio/mpeg"}
            self.client.upload_fileobj(body, self.bucket, key, ExtraArgs=extra_args)
            
            if make_public:
                try:
                    self.client.put_object_acl(
                        Bucket=self.bucket,
                        Key=key,
                        ACL="public-read"
                    )
                except (BotoCoreError, ClientError) as e:
                    logger.warning(f"⚠️ Could not set public ACL: {e}")
            
            url = f"https://{self.bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"
            logger.info(f"✅ Uploaded audio to S3: {url}")
            return url
            
        except Exception as e:
            logger.error(f"❌ S3 audio upload failed: {e}")
            return None
    
    def upload_image_bytes(
        self, 
        image_data: bytes, 
        key_name: str,
        make_public: bool = True
    ) -> Optional[str]:
        """Upload image bytes to S3.
        
        Args:
            image_data: Image data as bytes.
            key_name: Name for the S3 object.
            make_public: Whether to make the object publicly readable.
            
        Returns:
            Public URL of the uploaded image, or None if failed.
        """
        try:
            logger.info(f"📤 Uploading image to S3...")
            
            body = io.BytesIO(image_data)
            key = f"{self.folder}{key_name}"
            
            extra_args = {"ContentType": "image/png"}
            self.client.upload_fileobj(body, self.bucket, key, ExtraArgs=extra_args)
            
            if make_public:
                try:
                    self.client.put_object_acl(
                        Bucket=self.bucket,
                        Key=key,
                        ACL="public-read"
                    )
                except (BotoCoreError, ClientError) as e:
                    logger.warning(f"⚠️ Could not set public ACL: {e}")
            
            url = f"https://{self.bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"
            logger.info(f"✅ Uploaded image to S3: {url}")
            return url
            
        except Exception as e:
            logger.error(f"❌ S3 image upload failed: {e}")
            return None


# =============================================================================
# ZAPCAP SERVICE (Subtitles)
# =============================================================================
class ZapCapService:
    """Service for adding subtitles via ZapCap API."""
    
    # List of available template IDs for random selection
    TEMPLATE_IDS = [
        "your-zapcap-template-id",
        "50cdfac1-0a7a-48dd-af14-4d24971e213a",
        "55267be2-9eec-4d06-aff8-edcb401b112e",
        "5de632e7-0b02-4d15-8137-e004871e861b",
        "7b946549-ae16-4085-9dd3-c20c82504daa",
        "982ad276-a76f-4d80-a4e2-b8fae0038464",
        "a51c5222-47a7-4c37-b052-7b9853d66bf6",
        "ca050348-e2d0-49a7-9c75-7a5e8335c67d",
        "d46bb0da-cce0-4507-909d-fa8904fb8ed7",
        "dfe027d9-bd9d-4e55-a94f-d57ed368a060",
        "e659ee0c-53bb-497e-869c-90f8ec0a921f",
        "d2018215-2125-41c1-940e-f13b411fff5c",
        "1c0c9b65-47c4-41bf-a187-25a8305fd0dd",
        "a104df87-5b1a-4490-8cca-62e504a84615",
        "6255949c-4a52-4255-8a67-39ebccfaa3ef",
        "a6760d82-72c1-4190-bfdb-7d9c908732f1"
    ]
    
    def __init__(self, api_key: str, template_id: str = None):
        """Initialize ZapCap service.
        
        Args:
            api_key: ZapCap API key.
            template_id: Optional template ID for caption styling (if not provided, will be randomly selected per request).
        """
        self.api_key = api_key
        self.base_url = config.ZAPCAP_BASE_URL
        self.default_template_id = template_id or config.ZAPCAP_TEMPLATE_ID
    
    def _get_random_template_id(self) -> str:
        """Get a random template ID from the available list.
        
        Returns:
            A random template ID string.
        """
        return random.choice(self.TEMPLATE_IDS)
    
    def add_subtitles(
        self, 
        video_url: str, 
        language: str = "en",
        transcript: List[Dict] = None
    ) -> Optional[str]:
        """Add subtitles to video using ZapCap.
        
        Args:
            video_url: URL of the video to add subtitles to.
            language: Language code for subtitles (default: "en").
            transcript: Optional list of word segments with timing for "Bring Your Own Transcript".
                       Each segment: {"text": "word", "type": "word", "start_time": 0.0, "end_time": 0.5}
                       When provided, ZapCap skips auto-transcription and uses these values.
            
        Returns:
            URL of the captioned video, or None if failed.
        """
        try:
            if transcript:
                logger.info(f"📝 Adding subtitles via ZapCap with custom transcript ({len(transcript)} words, language: {language})...")
            else:
                logger.info(f"📝 Adding subtitles via ZapCap with auto-transcription (language: {language})...")
            
            # Step 1: Download video
            video_data = self._download_video(video_url)
            if not video_data:
                logger.error("❌ Failed to download video for subtitles")
                return None
            
            logger.info(f"   Downloaded video: {len(video_data)} bytes")
            
            # Step 2: Upload to ZapCap
            headers = {"x-api-key": self.api_key}
            
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_file.write(video_data)
                temp_path = temp_file.name
            
            try:
                with open(temp_path, "rb") as f:
                    files = {"file": (f"video_{int(time.time())}.mp4", f, "video/mp4")}
                    response = requests.post(
                        f"{self.base_url}/videos",
                        headers=headers,
                        files=files,
                        timeout=300  # Increased from 120 for larger videos
                    )
                
                if response.status_code != 201:
                    logger.error(f"❌ ZapCap upload failed: {response.status_code} - {response.text}")
                    return None
                
                video_id = response.json().get("id")
                logger.info(f"   Uploaded to ZapCap: video_id={video_id}")
                
                # Step 3: Create caption task
                # Select a random template ID for each request
                selected_template_id = self._get_random_template_id()
                logger.info(f"   Using random ZapCap template: {selected_template_id}")
                
                task_body = {
                    "templateId": selected_template_id,
                    "language": language.lower(),
                    "autoApprove": True,
                    "renderOptions": {
                        "subsOptions": {
                            "emoji": True,
                            "emojiAnimation": True,
                            "emphasizeKeywords": True
                        },
                        "styleOptions": {
                            "top": 28,
                            "fontUppercase": False,
                            "fontSize": 42,
                            "fontWeight": 700,
                            "fontColor": "#FFFFFF",
                            "fontShadow": "m",
                            "stroke": "s",
                            "strokeColor": "#000000"
                        }
                    }
                }
                
                # Add custom transcript if provided (Bring Your Own Transcript)
                if transcript:
                    task_body["transcript"] = transcript
                    logger.info(f"   Using custom transcript with {len(transcript)} words")
                
                task_response = requests.post(
                    f"{self.base_url}/videos/{video_id}/task",
                    headers=headers,
                    json=task_body,
                    timeout=60
                )
                
                if task_response.status_code not in [200, 201]:
                    logger.error(f"❌ ZapCap task creation failed: {task_response.text}")
                    return None
                
                task_result = task_response.json()
                task_id = task_result.get("taskId") or task_result.get("id")
                logger.info(f"   Task created: task_id={task_id}")
                
                # Step 4: Wait for completion
                return self._wait_for_completion(video_id, task_id)
                
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
        except Exception as e:
            logger.error(f"❌ ZapCap error: {e}")
            return None
    
    def _wait_for_completion(self, video_id: str, task_id: str, timeout: int = 600) -> Optional[str]:
        """Wait for ZapCap to finish processing.
        
        Args:
            video_id: ZapCap video ID.
            task_id: ZapCap task ID.
            timeout: Maximum wait time in seconds.
            
        Returns:
            URL of the captioned video, or None if failed/timeout.
        """
        headers = {"x-api-key": self.api_key}
        start_time = time.time()
        
        logger.info(f"   Waiting for ZapCap processing (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.base_url}/videos/{video_id}/task/{task_id}",
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                status = data.get("status", "").lower()
                
                if status == "completed":
                    download_url = data.get("downloadUrl")
                    if download_url:
                        logger.info(f"✅ ZapCap subtitles added: {download_url}")
                        return download_url
                    else:
                        logger.warning("⚠️ ZapCap completed but no download URL")
                        return None
                        
                elif status == "failed":
                    error_msg = data.get("error", "Unknown error")
                    logger.error(f"❌ ZapCap task failed: {error_msg}")
                    return None
                
                # Still processing
                time.sleep(10)
                
            except Exception as e:
                logger.warning(f"⚠️ Error checking ZapCap status: {e}")
                time.sleep(10)
        
        logger.error(f"❌ ZapCap timeout after {timeout}s")
        return None
    
    def _download_video(self, url: str) -> Optional[bytes]:
        """Download video from URL.
        
        Args:
            url: URL of the video to download.
            
        Returns:
            Video data as bytes, or None if failed.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=120)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"❌ Video download failed: {e}")
            return None


# =============================================================================
# SUNO MUSIC SERVICE (Music Generation via Kie.ai)
# =============================================================================
class SunoMusicService:
    """Service for generating music using Suno via Kie.ai API."""
    
    def __init__(self, api_key: str, openai_client):
        """Initialize Suno Music service.
        
        Args:
            api_key: Kie.ai API key.
            openai_client: OpenAI client for Whisper transcription.
        """
        self.api_key = api_key
        self.base_url = config.KIE_BASE_URL
        self.openai_client = openai_client
    
    def detect_lyrics_in_audio(self, audio_path: str) -> Tuple[bool, str]:
        """Detect if audio has lyrics using OpenAI Whisper.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Tuple of (has_lyrics: bool, lyrics_text: str)
        """
        try:
            logger.info("🎤 Detecting lyrics in audio using Whisper...")
            
            with open(audio_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            lyrics = transcript.text.strip()
            # If more than 20 characters, likely has lyrics (not just noise/artifacts)
            has_lyrics = len(lyrics) > 20
            
            if has_lyrics:
                logger.info(f"   Found lyrics ({len(lyrics)} chars): {lyrics[:100]}...")
            else:
                logger.info("   No lyrics detected (instrumental)")
            
            return has_lyrics, lyrics
            
        except Exception as e:
            logger.warning(f"⚠️ Could not detect lyrics: {e}")
            return False, ""
    
    def generate_instrumental_background(
        self, 
        audio_url: str,
        style: str = None,
        fallback_style: str = None
    ) -> Optional[str]:
        """Generate instrumental background music.
        
        First tries upload-cover with reference audio using creative parameters.
        If that fails (e.g., copyright detection), falls back to pure generation
        using the AI-generated style description.
        
        Args:
            audio_url: URL of the original audio to use as reference.
            style: Style description for the instrumental.
            fallback_style: AI-generated style description for pure generation fallback.
            
        Returns:
            URL of the generated instrumental, or None if all methods failed.
        """
        # Default style if not provided
        if not style:
            style = "upbeat corporate background music, modern, professional, energetic, no vocals"
        
        # Try upload-cover first with creative parameters
        result = self._try_upload_cover(audio_url, style)
        
        if result:
            return result
        
        # Fallback: Generate pure music without reference audio
        # This avoids copyright issues entirely
        fallback_description = fallback_style or style
        logger.info("🔄 Upload-cover failed, falling back to pure music generation (no reference audio)...")
        return self.generate_pure_music(fallback_description)
    
    def _try_upload_cover(self, audio_url: str, style: str) -> Optional[str]:
        """Try to generate music using upload-cover with reference audio.
        
        Uses creative parameters to minimize copyright detection.
        
        Args:
            audio_url: URL of the original audio.
            style: Style description.
            
        Returns:
            URL of generated music, or None if failed.
        """
        try:
            logger.info("🎵 Trying Suno upload-cover (creative params)...")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # CREATIVE PARAMETERS to minimize copyright detection
            request_body = {
                "uploadUrl": audio_url,
                "customMode": True,
                "instrumental": True,
                "style": style[:1000],
                "title": f"BGM_{int(time.time())}",
                "model": "V5",
                # Very creative settings to avoid "matches existing work" error
                "audioWeight": 0.25,       # Very low = minimal similarity to reference
                "styleWeight": 0.80,       # High = more influence from style text
                "weirdnessConstraint": 0.60,  # High = allows more creative deviation
                "callBackUrl": "https://httpbin.org/post"
            }
            
            logger.info(f"   Style: {style[:50]}...")
            logger.info(f"   Source audio: {audio_url[:50]}...")
            logger.info(f"   Creative params: audioWeight=0.25, styleWeight=0.80, weirdness=0.60")
            
            response = requests.post(
                f"{self.base_url}/api/v1/generate/upload-cover",
                headers=headers,
                json=request_body,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.warning(f"⚠️ Suno upload-cover API error: {response.status_code}")
                return None
            
            result = response.json()
            if result.get("code") != 200:
                logger.warning(f"⚠️ Suno upload-cover returned error: {result.get('msg')}")
                return None
            
            task_id = result.get("data", {}).get("taskId")
            if not task_id:
                logger.warning("⚠️ No task ID returned from Suno upload-cover")
                return None
            
            logger.info(f"   Suno upload-cover task started: {task_id}")
            
            return self._wait_for_music(task_id)
            
        except Exception as e:
            logger.warning(f"⚠️ Suno upload-cover error: {e}")
            return None
    
    def generate_pure_music(self, style_description: str) -> Optional[str]:
        """Generate music purely from a text description (no reference audio).
        
        This avoids copyright issues entirely by not using any reference audio.
        Uses the Kie.ai /api/v1/generate endpoint with customMode=True and instrumental=True.
        
        Args:
            style_description: Detailed description of the music style to generate.
            
        Returns:
            URL of the generated music, or None if failed.
        """
        try:
            logger.info("🎵 Generating original music with Suno (pure generation, no reference audio)...")
            logger.info(f"   Style: {style_description[:100]}...")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Use /api/v1/generate endpoint (correct endpoint for pure generation)
            # With customMode=True and instrumental=True, only style and title are required
            request_body = {
                "customMode": True,
                "instrumental": True,  # Force instrumental (no vocals)
                "style": style_description[:1000],  # V5 supports up to 1000 chars for style
                "title": f"BGM_{int(time.time())}",
                "model": "V5",
                "callBackUrl": "https://httpbin.org/post",
                # Optional: control creativity
                "styleWeight": 0.85,  # Strong adherence to style description
                "weirdnessConstraint": 0.40  # Some creative freedom
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/generate",
                headers=headers,
                json=request_body,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"❌ Suno pure generation API error: {response.status_code} - {response.text}")
                return None
            
            result = response.json()
            if result.get("code") != 200:
                logger.error(f"❌ Suno pure generation returned error: {result.get('msg')}")
                return None
            
            task_id = result.get("data", {}).get("taskId")
            if not task_id:
                logger.error("❌ No task ID returned from Suno pure generation")
                return None
            
            logger.info(f"   Suno pure generation task started: {task_id}")
            
            return self._wait_for_music(task_id)
            
        except Exception as e:
            logger.error(f"❌ Suno pure generation error: {e}")
            return None
    
    def generate_cover_music(
        self, 
        audio_url: str, 
        audio_path: str = None, 
        style: str = None
    ) -> Optional[str]:
        """Generate a cover of the audio with similar style using Suno.
        
        This is used when the original video has NO voice-over (music only),
        and we want to create a new version of that music.
        
        Uses creative parameters to avoid copyright issues while still
        capturing the dynamic style/mood of the reference audio.
        
        Args:
            audio_url: URL of the original audio.
            audio_path: Local path to audio (for lyrics detection).
            style: Optional style description.
            
        Returns:
            URL of the generated music, or None if failed.
        """
        try:
            logger.info("🎵 Generating new music with Suno (creative mode)...")
            
            # Detect if audio has lyrics
            has_lyrics = False
            lyrics = ""
            
            if audio_path and os.path.exists(audio_path):
                has_lyrics, lyrics = self.detect_lyrics_in_audio(audio_path)
            
            # Build request based on whether it has lyrics
            # CREATIVE PARAMETERS: Lower audioWeight + higher weirdnessConstraint
            # to create original music inspired by the reference without copying it
            if has_lyrics and lyrics:
                # Vocal cover configuration with CREATIVE settings
                logger.info("   Using VOCAL cover mode (with lyrics) - creative params")
                request_body = {
                    "uploadUrl": audio_url,
                    "customMode": True,
                    "instrumental": False,
                    "prompt": lyrics[:5000],  # Max 5000 chars for V5
                    "style": style or "Same style as original, modern production, fresh interpretation",
                    "title": f"Cover_{int(time.time())}",
                    "model": "V5",
                    # CREATIVE SETTINGS to avoid "matches existing work of art" error:
                    "audioWeight": 0.40,       # LOW = loosely inspired, not copying
                    "styleWeight": 0.65,       # HIGH = more influence from style description
                    "weirdnessConstraint": 0.45,  # HIGH = allows creative deviation
                    "callBackUrl": "https://httpbin.org/post"
                }
            else:
                # Instrumental cover configuration with CREATIVE settings
                logger.info("   Using INSTRUMENTAL cover mode - creative params")
                request_body = {
                    "uploadUrl": audio_url,
                    "customMode": True,
                    "instrumental": True,
                    "style": style or "Same instrumental style, modern production, unique arrangement",
                    "title": f"Cover_{int(time.time())}",
                    "model": "V5",
                    # CREATIVE SETTINGS to avoid "matches existing work of art" error:
                    "audioWeight": 0.35,       # LOW = loosely inspired, not copying
                    "styleWeight": 0.70,       # HIGH = more influence from style description
                    "weirdnessConstraint": 0.50,  # HIGH = allows creative deviation
                    "callBackUrl": "https://httpbin.org/post"
                }
            
            logger.info(f"   Creative params: audioWeight={request_body['audioWeight']}, "
                       f"styleWeight={request_body['styleWeight']}, weirdness={request_body['weirdnessConstraint']}")
            
            # Submit to Kie.ai
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/generate/upload-cover",
                headers=headers,
                json=request_body,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"❌ Suno API error: {response.status_code} - {response.text}")
                return None
            
            result = response.json()
            if result.get("code") != 200:
                logger.error(f"❌ Suno API returned error: {result.get('msg')}")
                return None
            
            task_id = result.get("data", {}).get("taskId")
            if not task_id:
                logger.error("❌ No task ID returned from Suno")
                return None
            
            logger.info(f"   Suno task started: {task_id}")
            
            # Wait for completion
            return self._wait_for_music(task_id)
            
        except Exception as e:
            logger.error(f"❌ Suno music generation error: {e}")
            return None
    
    def _wait_for_music(self, task_id: str, timeout: int = 600) -> Optional[str]:
        """Poll for music generation completion.
        
        Args:
            task_id: Suno task ID.
            timeout: Maximum wait time in seconds.
            
        Returns:
            URL of the generated music, or None if failed/timeout.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        start_time = time.time()
        
        logger.info(f"   Waiting for Suno music generation (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.base_url}/api/v1/generate/record-info",
                    headers=headers,
                    params={"taskId": task_id},
                    timeout=30
                )
                
                result = response.json()
                if result.get("code") != 200:
                    logger.warning(f"⚠️ Suno status check error: {result.get('msg')}")
                    time.sleep(15)
                    continue
                
                data = result.get("data", {})
                status = data.get("status", "")
                
                logger.debug(f"   Suno status: {status}")
                
                if status == "SUCCESS":
                    suno_data = data.get("response", {}).get("sunoData", [])
                    if suno_data:
                        audio_url = suno_data[0].get("audioUrl")
                        if audio_url:
                            logger.info(f"✅ Suno music generated: {audio_url}")
                            return audio_url
                            
                elif status in ["CREATE_TASK_FAILED", "GENERATE_AUDIO_FAILED", "CALLBACK_EXCEPTION", "SENSITIVE_WORD_ERROR"]:
                    error_msg = data.get("errorMessage", status)
                    logger.error(f"❌ Suno task failed: {error_msg}")
                    return None
                
                # Still processing (PENDING, TEXT_SUCCESS, FIRST_SUCCESS)
                time.sleep(15)
                
            except Exception as e:
                logger.warning(f"⚠️ Error checking Suno status: {e}")
                time.sleep(15)
        
        logger.error(f"❌ Suno timeout after {timeout}s")
        return None


# =============================================================================
# GCS VIDEO UPLOAD SERVICE
# =============================================================================
class GCSVideoUploadService:
    """Service for uploading final videos to Google Cloud Storage."""
    
    def __init__(self, credentials_file: str = None, bucket_name: str = None):
        """Initialize GCS Video Upload Service.
        
        Args:
            credentials_file: Path to service account JSON file.
            bucket_name: GCS bucket name for video uploads.
        """
        self.credentials_file = credentials_file or "service_account.json"
        self.bucket_name = bucket_name
        self.storage_client = None
        self.bucket = None
        self._initialized = False
    
    def _initialize(self) -> bool:
        """Lazy initialization of GCS client."""
        if self._initialized:
            return True
        
        try:
            # Get absolute path to credentials file
            if os.path.isabs(self.credentials_file):
                creds_path = self.credentials_file
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                creds_path = os.path.join(script_dir, self.credentials_file)
            
            if not os.path.exists(creds_path):
                # Try current working directory
                creds_path = os.path.join(os.getcwd(), self.credentials_file)
            
            if not os.path.exists(creds_path):
                logger.warning(f"⚠️ GCS credentials file not found: {self.credentials_file}")
                return False
            
            from google.oauth2 import service_account
            
            gcs_creds = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            self.storage_client = storage.Client(credentials=gcs_creds)
            self.bucket = self.storage_client.bucket(self.bucket_name)
            
            self._initialized = True
            logger.info(f"✅ GCS Video Upload Service initialized (bucket: {self.bucket_name})")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize GCS Video Upload: {e}")
            return False
    
    def upload_video_from_url(
        self, 
        source_url: str, 
        key_name: str,
        folder: str = "influencer_videos"
    ) -> Optional[str]:
        """Upload a video from URL to GCS.
        
        Args:
            source_url: URL of the video to upload.
            key_name: Name for the GCS object.
            folder: Folder path within the bucket.
            
        Returns:
            Public URL of the uploaded video, or None if failed.
        """
        if not self._initialize():
            return None
        
        try:
            logger.info(f"📤 Uploading video to GCS bucket '{self.bucket_name}'...")
            
            # Download video
            with requests.get(source_url, stream=True, timeout=180) as r:
                r.raise_for_status()
                video_data = io.BytesIO()
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        video_data.write(chunk)
                video_data.seek(0)
            
            # Build GCS path
            blob_path = f"{folder}/{key_name}" if folder else key_name
            blob = self.bucket.blob(blob_path)
            
            # Upload to GCS
            blob.upload_from_file(video_data, content_type="video/mp4")
            
            # Try to make public (may fail if bucket has uniform bucket-level access)
            try:
                blob.make_public()
            except Exception as acl_error:
                logger.info(f"ℹ️ ACL not supported, using direct URL (bucket is likely public)")
            
            # Construct public URL directly
            public_url = f"https://storage.googleapis.com/{self.bucket_name}/{blob_path}"
            logger.info(f"✅ Video uploaded to GCS: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"❌ Error uploading video to GCS: {e}")
            return None
    
    def upload_product_reference(
        self, 
        frame_path: str,
        folder: str = None
    ) -> Optional[str]:
        """Upload a product reference frame to GCS for use in image generation.
        
        Args:
            frame_path: Path to the local frame image file.
            folder: GCS folder for reference images (defaults to config setting).
            
        Returns:
            Public URL of the uploaded reference image, or None if failed.
        """
        if not self._initialize():
            return None
        
        folder = folder or config.PRODUCT_REFERENCE_FOLDER
        
        try:
            if not os.path.exists(frame_path):
                logger.error(f"❌ [PRODUCT] Reference frame not found: {frame_path}")
                return None
            
            logger.info(f"📤 [PRODUCT] Uploading reference frame to GCS...")
            
            # Generate unique filename
            timestamp = int(time.time())
            filename = os.path.basename(frame_path)
            name_part = os.path.splitext(filename)[0]
            key_name = f"product_ref_{name_part}_{timestamp}.jpg"
            
            # Read the image file
            with open(frame_path, 'rb') as f:
                image_data = io.BytesIO(f.read())
                image_data.seek(0)
            
            # Build GCS path
            blob_path = f"{folder}/{key_name}" if folder else key_name
            blob = self.bucket.blob(blob_path)
            
            # Upload to GCS
            blob.upload_from_file(image_data, content_type="image/jpeg")
            
            # Try to make public
            try:
                blob.make_public()
            except Exception as acl_error:
                logger.info(f"ℹ️ [PRODUCT] ACL not supported, using direct URL")
            
            # Construct public URL
            public_url = f"https://storage.googleapis.com/{self.bucket_name}/{blob_path}"
            logger.info(f"✅ [PRODUCT] Reference uploaded: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"❌ [PRODUCT] Error uploading reference: {e}")
            return None


# =============================================================================
# GCS ARTICLE SERVICE
# =============================================================================
class GCSArticleService:
    """Service for fetching article data from Google Cloud Storage.
    
    When the Article column contains a URL, this service looks up the corresponding
    JSON file in GCS and extracts the article content (Title, 1stP, Rest of Content).
    """
    
    def __init__(self, credentials_file: str = None, bucket_name: str = None, folder_name: str = None):
        """Initialize GCS Article Service.
        
        Args:
            credentials_file: Path to GCS service account JSON file.
            bucket_name: GCS bucket name.
            folder_name: Folder/prefix in the bucket containing article JSON files.
        """
        self.credentials_file = credentials_file or config.GCS_CREDENTIALS_FILE
        self.bucket_name = bucket_name or config.GCS_BUCKET_NAME
        self.folder_name = folder_name or config.GCS_FOLDER_NAME
        
        self.cache = {}  # In-memory cache for article data
        self.gcs_file_list_cache = None  # Cache for GCS file list
        self.gcs_file_list_cache_time = 0
        self.cache_ttl = 300  # 5 minutes TTL
        
        self.storage_client = None
        self.bucket = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of GCS client."""
        if self._initialized:
            return True
        
        try:
            # Get absolute path to GCS credentials file
            if os.path.isabs(self.credentials_file):
                gcs_credentials_path = self.credentials_file
            else:
                # Try current working directory
                gcs_credentials_path = os.path.join(os.getcwd(), self.credentials_file)
            
            if not os.path.exists(gcs_credentials_path):
                # Try script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                fallback_path = os.path.join(script_dir, self.credentials_file)
                if os.path.exists(fallback_path):
                    gcs_credentials_path = fallback_path
                else:
                    logger.warning(f"⚠️ GCS credentials file not found: {self.credentials_file}")
                    logger.warning("   Article URL lookup will be disabled")
                    return False
            
            logger.info(f"🔧 Loading GCS credentials from: {gcs_credentials_path}")
            
            gcs_creds = Credentials.from_service_account_file(
                gcs_credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            self.storage_client = storage.Client(credentials=gcs_creds)
            self.bucket = self.storage_client.bucket(self.bucket_name)
            
            self._initialized = True
            logger.info("✅ GCS Article Service initialized successfully")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize GCS: {e}")
            logger.warning("   Article URL lookup will be disabled")
            return False
    
    def is_url(self, value: str) -> bool:
        """Check if a value is a URL."""
        if not value or not isinstance(value, str):
            return False
        return value.strip().startswith("http")
    
    def url_to_filename(self, url: str) -> str:
        """Convert URL to filename pattern for GCS lookup.
        
        Args:
            url: The article URL.
            
        Returns:
            Sanitized filename pattern.
        """
        if not url:
            return ""
        
        # Apply domain substitutions
        substitutions = {
            "legacy-site-1.example.com": "current-site-a.example.com",
            "legacy-site-2.example.com": "current-site-a.example.com",
            "legacy-site-3.example.com": "current-site-b.example.com",
            "legacy-site-4.example.com": "current-site-b.example.com",
            "legacy-site-5.example.com": "current-site-b.example.com"
        }
        
        for old_domain, new_domain in substitutions.items():
            url = url.replace(old_domain, new_domain)
        
        # Convert URL to filename pattern
        sanitized = str(url).strip()
        
        # Replace problematic characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', sanitized)
        # Replace multiple spaces with single underscore
        sanitized = re.sub(r'\s+', '_', sanitized)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        return sanitized
    
    def get_article_data(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch article data from GCS by URL.
        
        Args:
            url: The article URL to look up.
            
        Returns:
            Dictionary with 'Title', '1stp', 'Rest of Content' or None if not found.
        """
        if not url or not self.is_url(url):
            return None
        
        # Initialize GCS client if needed
        if not self._initialize():
            return None
        
        try:
            # Check cache first
            cache_key = f"gcs_{url}"
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if cached is not None:
                    logger.debug(f"📋 Cache hit for: {url}")
                    return cached
                return None  # Negative cache hit
            
            url_filename = self.url_to_filename(url)
            logger.debug(f"🔍 Looking for GCS file matching: {url_filename[:50]}...")
            
            # Get cached file list or fetch new one
            current_time = time.time()
            if (self.gcs_file_list_cache is None or 
                current_time - self.gcs_file_list_cache_time > self.cache_ttl):
                
                logger.info("🔄 Refreshing GCS file list cache...")
                blobs = list(self.bucket.list_blobs(prefix=f"{self.folder_name}/"))
                self.gcs_file_list_cache = blobs
                self.gcs_file_list_cache_time = current_time
                logger.info(f"📋 Cached {len(blobs)} files from GCS")
            else:
                blobs = self.gcs_file_list_cache
            
            # Find matching file
            matched_file = None
            highest_version = 0
            
            for blob in blobs:
                filename = blob.name.split('/')[-1]
                
                if url_filename in filename:
                    # Extract version number if present
                    version_match = re.search(r'_v(\d+)\.json$', filename)
                    if version_match:
                        version = int(version_match.group(1))
                        if version > highest_version:
                            highest_version = version
                            matched_file = blob
                    elif matched_file is None:
                        matched_file = blob
            
            if matched_file:
                logger.info(f"✅ Found GCS file: {matched_file.name}")
                json_content = matched_file.download_as_text()
                file_data = json.loads(json_content)
                
                # Cache the result
                self.cache[cache_key] = file_data
                return file_data
            
            logger.warning(f"❌ No GCS file found for URL: {url[:50]}...")
            # Cache negative result
            self.cache[cache_key] = None
            return None
            
        except Exception as e:
            logger.error(f"❌ Error fetching article from GCS: {e}")
        return None


# =============================================================================
# MAIN VIDEO SCENE PROCESSOR
# =============================================================================
class VideoSceneProcessor:
    """Main processor for the video scene processing pipeline."""
    
    def __init__(self):
        """Initialize the video scene processor with all services."""
        # Validate required API keys
        self._validate_config()
        
        # Thread lock for Google Sheets updates (prevents race conditions in parallel mode)
        self._sheets_lock = threading.Lock()
        # Global rate limiter for Sheets writes — keeps us under the 60/min quota
        # so higher row concurrency does not trigger 429 backoff storms.
        self._sheets_rate_limiter = _SheetsRateLimiter(max_calls=50, period=60.0)
        
        # Initialize services
        self.sheets_service = GoogleSheetsService(config.SERVICE_ACCOUNT_FILE)
        self.openai_service = OpenAIService(config.OPENAI_API_KEY)
        
        # Initialize S3 first so we can pass it to other services
        self.s3_service = S3Service(
            access_key_id=config.AWS_ACCESS_KEY_ID,
            secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            bucket_name=config.AWS_BUCKET_NAME,
            region=config.AWS_REGION,
            folder_path=config.AWS_FOLDER_PATH
        )
        
        # Pass S3 service to KieAIService for CTA button uploads
        self.kie_service = KieAIService(config.KIE_API_KEY, s3_service=self.s3_service)
        self.rendi_service = RendiService(config.RENDI_API_KEY, s3_service=self.s3_service)
        
        # Gemini service for native video analysis (via Kie.ai - uses same API key)
        if config.ENABLE_GEMINI_VIDEO_ANALYSIS and config.KIE_API_KEY:
            self.gemini_service = GeminiService(config.KIE_API_KEY, s3_service=self.s3_service)
        else:
            self.gemini_service = None
            if config.ENABLE_GEMINI_VIDEO_ANALYSIS:
                logger.warning("⚠️ Gemini video analysis enabled but KIE_API_KEY not set")
        
        # Pass OpenAI client to ElevenLabs for speech detection
        self.elevenlabs_service = ElevenLabsService(
            config.ELEVENLABS_API_KEY,
            openai_client=self.openai_service.client
        )
        
        # ZapCap service for subtitles (optional - only if API key is set)
        if config.ZAPCAP_API_KEY:
            self.zapcap_service = ZapCapService(
                api_key=config.ZAPCAP_API_KEY,
                template_id=config.ZAPCAP_TEMPLATE_ID
            )
            logger.info("   ✅ ZapCap service initialized")
        else:
            self.zapcap_service = None
            logger.info("   ⚠️ ZapCap service not available (no API key)")
        
        # Suno Music service for music generation (uses Kie.ai API key)
        self.suno_service = SunoMusicService(
            api_key=config.KIE_API_KEY,
            openai_client=self.openai_service.client
        )
        
        # GCS Article service for fetching article data from URLs
        self.gcs_article_service = GCSArticleService(
            credentials_file=config.GCS_CREDENTIALS_FILE,
            bucket_name=config.GCS_BUCKET_NAME,
            folder_name=config.GCS_FOLDER_NAME
        )
        
        # GCS Video Upload service for final influencer videos
        self.gcs_video_service = GCSVideoUploadService(
            credentials_file="service_account.json",
            bucket_name=os.environ.get("GCS_VIDEO_BUCKET_NAME", "")
        )
        
        logger.info("✅ VideoSceneProcessor initialized successfully")
    
    def _validate_config(self) -> None:
        """Validate that all required configuration is present."""
        required_keys = {
            "OPENAI_API_KEY": config.OPENAI_API_KEY,
            "KIE_API_KEY": config.KIE_API_KEY,
            "RENDI_API_KEY": config.RENDI_API_KEY,
            "ELEVENLABS_API_KEY": config.ELEVENLABS_API_KEY
        }
        
        missing = [key for key, value in required_keys.items() if not value]
        
        if missing:
            raise ValueError(f"Missing required API keys: {', '.join(missing)}")
    
    def process_all_videos(self) -> Dict[str, Any]:
        """Process all videos from the Google Sheet.
        
        Returns:
            Dict with processing results.
        """
        logger.info("🚀 Starting video processing pipeline...")
        
        # Read data from Google Sheet
        headers, data_rows = self.sheets_service.get_worksheet_data(
            config.GOOGLE_SHEET_ID,
            config.GOOGLE_SHEET_TAB
        )
        
        # Get input column index
        try:
            input_col = self.sheets_service.get_column_index(
                headers, 
                config.INPUT_VIDEO_COLUMN
            )
        except ValueError:
            logger.error(f"❌ Input column '{config.INPUT_VIDEO_COLUMN}' not found")
            return {"error": "Input column not found", "processed": 0}
        
        # Get manual instructions column index (optional)
        try:
            manual_instructions_col = self.sheets_service.get_column_index(
                headers, 
                config.MANUAL_INSTRUCTIONS_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Manual instructions column not found, proceeding without")
            manual_instructions_col = None
        
        # Get CTA button column index (optional)
        try:
            cta_button_col = self.sheets_service.get_column_index(
                headers, 
                config.ADD_CTA_BUTTON_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ CTA button column not found, proceeding without")
            cta_button_col = None
        
        # Get CTA text column index (optional)
        try:
            cta_text_col = self.sheets_service.get_column_index(
                headers, 
                config.CTA_TEXT_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ CTA text column not found, proceeding without")
            cta_text_col = None
        
        # Get CTA Duration column index (optional)
        try:
            cta_duration_col = self.sheets_service.get_column_index(
                headers, 
                config.CTA_DURATION_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ CTA Duration column not found, defaulting to 'At the End'")
            cta_duration_col = None
        
        # Get Add subtitles column index (optional)
        try:
            add_subtitles_col = self.sheets_service.get_column_index(
                headers, 
                config.ADD_SUBTITLES_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Add subtitles column not found, proceeding without")
            add_subtitles_col = None
        
        # Get Opening Text columns (optional)
        try:
            add_opening_text_col = self.sheets_service.get_column_index(
                headers, 
                config.ADD_OPENING_TEXT_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Opening Text? column not found, proceeding without")
            add_opening_text_col = None
        
        try:
            opening_text_col = self.sheets_service.get_column_index(
                headers, 
                config.OPENING_TEXT_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Opening Text column not found, proceeding without")
            opening_text_col = None
        
        # Get Article column index (optional - for content adaptation)
        try:
            article_col = self.sheets_service.get_column_index(
                headers, 
                config.ARTICLE_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Article column not found, proceeding without")
            article_col = None
        
        # Get Vertical column index (optional - for content adaptation)
        try:
            vertical_col = self.sheets_service.get_column_index(
                headers, 
                config.VERTICAL_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Vertical column not found, proceeding without")
            vertical_col = None
        
        # Get Language column index (optional - for ZapCap subtitles)
        try:
            language_col = self.sheets_service.get_column_index(
                headers, 
                config.LANGUAGE_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Language column not found, proceeding without")
            language_col = None
        
        # Get Manual VO text column index (optional - override generated VO)
        try:
            manual_vo_text_col = self.sheets_service.get_column_index(
                headers, 
                config.MANUAL_VO_TEXT_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Manual VO text column not found, proceeding without")
            manual_vo_text_col = None
        
        # Get Manual music link column index (optional - override Suno music)
        try:
            manual_music_link_col = self.sheets_service.get_column_index(
                headers, 
                config.MANUAL_MUSIC_LINK_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Manual music link column not found, proceeding without")
            manual_music_link_col = None
        
        # Get Free text column index (optional - overrides Title, 1stP, Rest of Content)
        try:
            free_text_col = self.sheets_service.get_column_index(
                headers, 
                config.FREE_TEXT_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Free text column not found, proceeding without")
            free_text_col = None
        
        # Get Influencer Mode columns (Image 1-4, Time)
        image_cols = []
        for i in range(1, 5):
            col_name = getattr(config, f"IMAGE_{i}_COLUMN", f"Image {i}")
            try:
                img_col = self.sheets_service.get_column_index(headers, col_name)
                image_cols.append(img_col)
            except ValueError:
                image_cols.append(None)
        
        try:
            time_col = self.sheets_service.get_column_index(headers, config.TIME_COLUMN)
        except ValueError:
            logger.info(f"ℹ️ Time column not found, proceeding without")
            time_col = None
        
        # Get Voice id column index (optional - custom ElevenLabs voice)
        try:
            voice_id_col = self.sheets_service.get_column_index(headers, config.VOICE_ID_COLUMN)
        except ValueError:
            logger.info(f"ℹ️ Voice id column not found, using default voice")
            voice_id_col = None
        
        # Get Animation model column index (optional - "runway" or "kling")
        try:
            animation_model_col = self.sheets_service.get_column_index(headers, config.ANIMATION_MODEL_COLUMN)
        except ValueError:
            logger.info(f"ℹ️ Animation model column not found, using default (runway)")
            animation_model_col = None
        
        # Get Title column index (optional - populated from GCS when Article is URL)
        try:
            title_col = self.sheets_service.get_column_index(
                headers, 
                config.TITLE_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Title column not found, proceeding without")
            title_col = None
        
        # Get 1stP column index (optional - populated from GCS when Article is URL)
        try:
            first_para_col = self.sheets_service.get_column_index(
                headers, 
                config.FIRST_PARAGRAPH_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ 1stP column not found, proceeding without")
            first_para_col = None
        
        # Get Rest of Content column index (optional - populated from GCS when Article is URL)
        try:
            rest_content_col = self.sheets_service.get_column_index(
                headers, 
                config.REST_CONTENT_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Rest of Content column not found, proceeding without")
            rest_content_col = None
        
        # Get Article related to Video column index (optional - "Yes" or "No")
        # "Yes" = Article is similar to video, adapt video for new offer/language
        # "No" = Article is fundamentally different, keep style but create new content
        try:
            article_related_col = self.sheets_service.get_column_index(
                headers, 
                config.ARTICLE_RELATED_TO_VIDEO_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Article related to Video column not found, defaulting to 'Yes' behavior")
            article_related_col = None
        
        # Get Product image (optional) column: if has link use as product reference; if empty generate and write back
        try:
            product_image_col = self.sheets_service.get_column_index(
                headers,
                config.PRODUCT_IMAGE_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Product image (optional) column not found, proceeding without")
            product_image_col = None

        # Get Final Video column index (used to skip rows that are already done)
        try:
            final_video_col = self.sheets_service.get_column_index(
                headers,
                config.FINAL_VIDEO_COLUMN
            )
        except ValueError:
            logger.info(f"ℹ️ Final Video column not found, cannot skip completed rows")
            final_video_col = None

        results = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        # Helper function to process a single row (for parallel execution)
        def process_row(row_idx: int, row_data: List[str]) -> Optional[Dict[str, Any]]:
            """Process a single row - designed for parallel execution."""
            row_num = row_idx + 2  # 1-based, accounting for header
            
            # Make a copy of row_data to avoid mutation issues
            row_data = list(row_data)
            
            # Ensure row has enough columns
            while len(row_data) < len(headers):
                row_data.append('')
            
            # Get video URL
            video_url = row_data[input_col].strip() if input_col < len(row_data) else ""
            
            # Get Free text for potential influencer mode
            free_text = ""
            if free_text_col is not None and free_text_col < len(row_data):
                free_text = row_data[free_text_col].strip()

            # Skip rows that already have a Final Video — avoids re-processing and
            # re-spending on rows completed by a previous run.
            if final_video_col is not None and final_video_col < len(row_data):
                if row_data[final_video_col].strip():
                    logger.info(f"⏭️ Row {row_num}: already has Final Video — skipping")
                    return None

            # Check if this is an influencer mode row (no video URL but has Free text)
            if not video_url:
                if not free_text:
                    return None  # Skip empty rows (no video URL and no Free text)
                
                # INFLUENCER MODE - route to influencer processing
                logger.info(f"🎭 Row {row_num}: Influencer mode detected (no Input Videos, has Free text)")
                
                # Get additional columns for influencer mode
                manual_instructions = ""
                if manual_instructions_col is not None and manual_instructions_col < len(row_data):
                    manual_instructions = row_data[manual_instructions_col].strip()
                
                cta_button = False
                if cta_button_col is not None and cta_button_col < len(row_data):
                    cta_button = row_data[cta_button_col].strip().lower() == "yes"
                
                cta_text = ""
                if cta_text_col is not None and cta_text_col < len(row_data):
                    cta_text = row_data[cta_text_col].strip()
                
                # Get CTA Duration setting for influencer mode
                cta_duration = "at_the_end"  # Default
                if cta_duration_col is not None and cta_duration_col < len(row_data):
                    duration_value = row_data[cta_duration_col].strip().lower()
                    if "whole" in duration_value:
                        cta_duration = "whole_video"
                    else:
                        cta_duration = "at_the_end"
                
                add_subtitles = False
                if add_subtitles_col is not None and add_subtitles_col < len(row_data):
                    add_subtitles = row_data[add_subtitles_col].strip().lower() == "yes"
                
                language = ""
                if language_col is not None and language_col < len(row_data):
                    language = row_data[language_col].strip()
                
                manual_vo_text = ""
                if manual_vo_text_col is not None and manual_vo_text_col < len(row_data):
                    manual_vo_text = row_data[manual_vo_text_col].strip()
                
                manual_music_link = ""
                if manual_music_link_col is not None and manual_music_link_col < len(row_data):
                    manual_music_link = row_data[manual_music_link_col].strip()
                
                # Get Image 1-4 URLs
                image_urls = []
                for img_col in image_cols:
                    if img_col is not None and img_col < len(row_data):
                        img_url = row_data[img_col].strip()
                        if img_url:
                            image_urls.append(img_url)
                
                # Get scene count from Time column
                scene_count = config.DEFAULT_INFLUENCER_SCENES
                if time_col is not None and time_col < len(row_data):
                    time_value = row_data[time_col].strip()
                    if time_value.isdigit():
                        scene_count = min(int(time_value), 10)  # Max 10 scenes
                
                # Get custom voice ID if specified
                custom_voice_id = ""
                if voice_id_col is not None and voice_id_col < len(row_data):
                    custom_voice_id = row_data[voice_id_col].strip()
                
                # Process in influencer mode
                return self.process_influencer_row(
                    row_num=row_num,
                    row_data=row_data,
                    headers=headers,
                    free_text=free_text,
                    manual_instructions=manual_instructions,
                    language=language,
                    cta_button=cta_button,
                    cta_text=cta_text,
                    cta_duration=cta_duration,
                    add_subtitles=add_subtitles,
                    manual_vo_text=manual_vo_text,
                    manual_music_link=manual_music_link,
                    image_urls=image_urls,
                    scene_count=scene_count,
                    voice_id=custom_voice_id
                )
            
            # NORMAL VIDEO MODE - continue with existing logic
            # Get manual instructions if available
            manual_instructions = ""
            if manual_instructions_col is not None and manual_instructions_col < len(row_data):
                manual_instructions = row_data[manual_instructions_col].strip()
                if manual_instructions:
                    logger.info(f"📝 Row {row_num}: Manual instructions found: {manual_instructions[:50]}...")
            
            # Get CTA button setting
            cta_button = False
            if cta_button_col is not None and cta_button_col < len(row_data):
                cta_value = row_data[cta_button_col].strip().lower()
                cta_button = cta_value == "yes"
            
            # Get CTA text
            cta_text = ""
            if cta_text_col is not None and cta_text_col < len(row_data):
                cta_text = row_data[cta_text_col].strip()
            
            # Get CTA Duration setting ("Whole Video" or "At the End", default: "At the End")
            cta_duration = "at_the_end"  # Default
            if cta_duration_col is not None and cta_duration_col < len(row_data):
                duration_value = row_data[cta_duration_col].strip().lower()
                if "whole" in duration_value:
                    cta_duration = "whole_video"
                else:
                    cta_duration = "at_the_end"
            
            # Get Add subtitles setting
            add_subtitles = False
            if add_subtitles_col is not None and add_subtitles_col < len(row_data):
                subtitles_value = row_data[add_subtitles_col].strip().lower()
                add_subtitles = subtitles_value == "yes"
            
            # Get Opening Text settings
            add_opening_text = False
            if add_opening_text_col is not None and add_opening_text_col < len(row_data):
                add_opening_text = row_data[add_opening_text_col].strip().lower() == "yes"
            
            opening_text = ""
            if opening_text_col is not None and opening_text_col < len(row_data):
                opening_text = row_data[opening_text_col].strip()
            
            # Get Animation model (optional - "runway", "kling", or "kling-3.0", default is runway)
            animation_model = "runway"  # Default
            if animation_model_col is not None and animation_model_col < len(row_data):
                anim_value = row_data[animation_model_col].strip().lower()
                if anim_value in ["kling-3.0", "kling 3.0", "kling3.0"]:
                    animation_model = "kling-3.0"
                    logger.info(f"🎬 Row {row_num}: Using Kling 3.0 for video generation")
                elif anim_value in ["kling", "kling v2.5", "kling v2-5"]:
                    animation_model = "kling"
                    logger.info(f"🎬 Row {row_num}: Using Kling V2.5 for video generation")
                elif anim_value:
                    logger.info(f"🎬 Row {row_num}: Using Runway for video generation (default)")
            
            # Get Article content (optional - for content adaptation)
            # Step 1: Read existing data from Title, 1stP, Rest of Content columns
            existing_title = ""
            existing_first_para = ""
            existing_rest_content = ""
            
            if title_col is not None and title_col < len(row_data):
                existing_title = row_data[title_col].strip()
            if first_para_col is not None and first_para_col < len(row_data):
                existing_first_para = row_data[first_para_col].strip()
            if rest_content_col is not None and rest_content_col < len(row_data):
                existing_rest_content = row_data[rest_content_col].strip()
            
            # Step 2: If Article column contains a URL, fetch missing data from GCS
            article_text = ""
            article_value = ""
            if article_col is not None and article_col < len(row_data):
                article_value = row_data[article_col].strip()
                
                # Check if it's a URL - if so, try to fill missing columns from GCS
                if self.gcs_article_service.is_url(article_value):
                    logger.info(f"🔗 Row {row_num}: Article contains URL, fetching from GCS...")
                    gcs_data = self.gcs_article_service.get_article_data(article_value)
                    
                    if gcs_data:
                        # Extract article components from GCS
                        gcs_title = gcs_data.get('Title', '')
                        gcs_first_para = gcs_data.get('1stp', '')
                        gcs_rest_content = gcs_data.get('Rest of Content', '')
                        gcs_language = gcs_data.get('language', '')
                        
                        # Prepare updates - only update empty columns
                        updates_to_make = []
                        
                        # Fill Title if empty in sheet but available in GCS
                        if not existing_title and gcs_title and title_col is not None:
                            existing_title = gcs_title
                            updates_to_make.append({
                                'row': row_num,
                                'column': config.TITLE_COLUMN,
                                'value': gcs_title
                            })
                            logger.info(f"   📋 Row {row_num}: Title (from GCS): {gcs_title[:50]}...")
                        
                        # Fill 1stP if empty in sheet but available in GCS
                        if not existing_first_para and gcs_first_para and first_para_col is not None:
                            existing_first_para = gcs_first_para
                            updates_to_make.append({
                                'row': row_num,
                                'column': config.FIRST_PARAGRAPH_COLUMN,
                                'value': gcs_first_para
                            })
                            logger.info(f"   📋 Row {row_num}: 1stP (from GCS): {gcs_first_para[:50]}...")
                        
                        # Fill Rest of Content if empty in sheet but available in GCS
                        if not existing_rest_content and gcs_rest_content and rest_content_col is not None:
                            existing_rest_content = gcs_rest_content
                            updates_to_make.append({
                                'row': row_num,
                                'column': config.REST_CONTENT_COLUMN,
                                'value': gcs_rest_content
                            })
                            logger.info(f"   📋 Row {row_num}: Rest of Content (from GCS): {gcs_rest_content[:50]}...")
                        
                        # Update Language column from GCS if it's empty
                        if gcs_language and language_col is not None:
                            current_language = row_data[language_col].strip() if language_col < len(row_data) else ""
                            if not current_language:
                                updates_to_make.append({
                                    'row': row_num,
                                    'column': config.LANGUAGE_COLUMN,
                                    'value': gcs_language.lower()
                                })
                                logger.info(f"   🌍 Row {row_num}: Language (from GCS): {gcs_language}")
                        
                        # Batch update the sheet with new data
                        if updates_to_make:
                            with self._sheets_lock:
                                self.sheets_service.batch_update_cells(
                                    config.GOOGLE_SHEET_ID,
                                    config.GOOGLE_SHEET_TAB,
                                    updates_to_make,
                                    headers
                                )
                            logger.info(f"   ✅ Row {row_num}: Updated {len(updates_to_make)} columns from GCS")
                    else:
                        logger.warning(f"⚠️ Row {row_num}: Could not fetch article from URL: {article_value[:50]}...")
            
            # Step 3: Combine all available content for OpenAI processing
            # Priority: 1. Free text (if provided), 2. Title/1stP/Rest of Content, 3. Article value as-is
            
            # Check for Free text override first
            free_text = ""
            if free_text_col is not None and free_text_col < len(row_data):
                free_text = row_data[free_text_col].strip()
            
            if free_text:
                # Free text overrides Title, 1stP, Rest of Content
                article_text = free_text
                logger.info(f"📰 Row {row_num}: Using Free text content ({len(article_text)} chars)")
                
                # Detect language from Free text and update Language column if empty
                if language_col is not None:
                    current_language = row_data[language_col].strip() if language_col < len(row_data) else ""
                    if not current_language:
                        detected_lang = detect_language(free_text)
                        if detected_lang:
                            try:
                                self.sheets_service.update_cell(
                                    sheet_id=config.GOOGLE_SHEET_ID,
                                    worksheet_name=config.GOOGLE_SHEET_TAB,
                                    row=row_num,
                                    column_name=config.LANGUAGE_COLUMN,
                                    value=detected_lang.lower(),
                                    headers=headers
                                )
                                logger.info(f"   🌍 Row {row_num}: Language (from Free text): {detected_lang}")
                            except Exception as e:
                                logger.warning(f"⚠️ Row {row_num}: Could not update Language column: {e}")
            elif existing_title or existing_first_para or existing_rest_content:
                article_text = f"{existing_title}\n\n{existing_first_para}\n\n{existing_rest_content}".strip()
                logger.info(f"📰 Row {row_num}: Combined article content ({len(article_text)} chars)")
            elif article_value and not self.gcs_article_service.is_url(article_value):
                # Not a URL - use as-is (backward compatibility)
                article_text = article_value
            
            # Get Vertical/offer name (optional - for content adaptation)
            vertical = ""
            if vertical_col is not None and vertical_col < len(row_data):
                vertical = row_data[vertical_col].strip()
            
            # Get Language code (optional - for ZapCap subtitles)
            subtitle_language = ""
            if language_col is not None and language_col < len(row_data):
                subtitle_language = row_data[language_col].strip().lower()
            
            # Get Manual VO text (optional - override generated VO)
            manual_vo_text = ""
            if manual_vo_text_col is not None and manual_vo_text_col < len(row_data):
                manual_vo_text = row_data[manual_vo_text_col].strip()
            
            # Get Manual music link (optional - override Suno music)
            manual_music_link = ""
            if manual_music_link_col is not None and manual_music_link_col < len(row_data):
                manual_music_link = row_data[manual_music_link_col].strip()
            
            # Get custom voice ID (optional - custom ElevenLabs voice)
            custom_voice_id = ""
            if voice_id_col is not None and voice_id_col < len(row_data):
                custom_voice_id = row_data[voice_id_col].strip()
            
            # Get Article related to Video value (optional - "Yes" or "No")
            # "Yes" (default) = Article is similar to video content, adapt video for new offer/language
            # "No" = Article is fundamentally different from video, keep style/atmosphere but create new content
            article_related_to_video = True  # Default to "Yes" behavior
            if article_related_col is not None and article_related_col < len(row_data):
                article_related_value = row_data[article_related_col].strip().lower()
                if article_related_value == "no":
                    article_related_to_video = False
                    logger.info(f"🔄 Row {row_num}: Article NOT related to video - will create new content while keeping video style")
                elif article_related_value == "yes":
                    logger.info(f"✅ Row {row_num}: Article IS related to video - will adapt video for new offer/language")
            
            if cta_button and cta_text:
                duration_str = "whole video" if cta_duration == "whole_video" else "at the end"
                logger.info(f"🔘 Row {row_num}: CTA button enabled: '{cta_text}' ({duration_str})")
            if add_subtitles:
                logger.info(f"📝 Row {row_num}: Subtitles will be added via ZapCap")
            if article_text:
                logger.info(f"📰 Row {row_num}: Article content provided ({len(article_text)} chars)")
            if vertical:
                logger.info(f"📊 Row {row_num}: Vertical: '{vertical}'")
            if subtitle_language:
                logger.info(f"🌍 Row {row_num}: Subtitle language: '{subtitle_language}'")
            if manual_vo_text:
                logger.info(f"🎤 Row {row_num}: Manual VO text provided ({len(manual_vo_text)} chars)")
            if manual_music_link:
                logger.info(f"🎵 Row {row_num}: Manual music link: '{manual_music_link[:50]}...'")
            if custom_voice_id:
                logger.info(f"🎤 Row {row_num}: Custom voice ID: '{custom_voice_id}'")
            
            # Get Product image (optional): if has link use as product reference; if empty we generate and write back
            product_image_from_sheet = ""
            if product_image_col is not None and product_image_col < len(row_data):
                product_image_from_sheet = row_data[product_image_col].strip()
            if product_image_from_sheet:
                logger.info(f"📷 Row {row_num}: Using provided product image from sheet")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📹 Processing row {row_num}: {video_url[:50]}...")
            logger.info(f"{'='*60}")
            
            try:
                result = self.process_single_video(
                    video_url=video_url,
                    row_num=row_num,
                    headers=headers,
                    manual_instructions=manual_instructions,
                    cta_button=cta_button,
                    cta_text=cta_text,
                    cta_duration=cta_duration,
                    add_subtitles=add_subtitles,
                    article_text=article_text,
                    vertical=vertical,
                    subtitle_language=subtitle_language,
                    manual_vo_text=manual_vo_text,
                    manual_music_link=manual_music_link,
                    voice_id=custom_voice_id,
                    add_opening_text=add_opening_text,
                    opening_text=opening_text,
                    animation_model=animation_model,
                    article_related_to_video=article_related_to_video,
                    product_image_url=product_image_from_sheet
                )
                return {"row": row_num, "result": result, "success": result.get("success", False)}
                
            except Exception as e:
                logger.error(f"❌ Error processing row {row_num}: {e}")
                return {"row": row_num, "result": None, "success": False, "error": str(e)}
        
        # Process rows in PARALLEL. Sheets writes are globally rate-limited
        # (see _SheetsRateLimiter), so higher concurrency no longer triggers quota errors.
        row_workers = getattr(config, "ROW_PARALLELISM", 6)
        logger.info(f"🚀 Processing {len(data_rows)} rows in PARALLEL (max {row_workers} concurrent workers)...")

        with ThreadPoolExecutor(max_workers=row_workers) as executor:
            # Submit all row processing tasks
            future_to_row = {
                executor.submit(process_row, row_idx, row_data): row_idx
                for row_idx, row_data in enumerate(data_rows)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_row):
                row_idx = future_to_row[future]
                try:
                    row_result = future.result()
                    if row_result is None:
                        continue  # Skipped row (no video URL)
                    
                    results["processed"] += 1
                    if row_result.get("success"):
                        results["successful"] += 1
                    else:
                        results["failed"] += 1
                    results["details"].append(row_result.get("result") or row_result)
                    
                except Exception as e:
                    logger.error(f"❌ Unexpected error in row {row_idx + 2}: {e}")
                    results["processed"] += 1
                    results["failed"] += 1
                    results["details"].append({
                        "row": row_idx + 2,
                        "success": False,
                        "error": str(e)
                    })
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 Processing complete!")
        logger.info(f"   Processed: {results['processed']}")
        logger.info(f"   Successful: {results['successful']}")
        logger.info(f"   Failed: {results['failed']}")
        logger.info(f"{'='*60}")
        
        return results
    
    def process_single_video(
        self, 
        video_url: str, 
        row_num: int,
        headers: List[str],
        manual_instructions: str = "",
        cta_button: bool = False,
        cta_text: str = "",
        cta_duration: str = "at_the_end",
        add_subtitles: bool = False,
        article_text: str = "",
        vertical: str = "",
        subtitle_language: str = "",
        manual_vo_text: str = "",
        manual_music_link: str = "",
        voice_id: str = "",
        add_opening_text: bool = False,
        opening_text: str = "",
        animation_model: str = "runway",
        article_related_to_video: bool = True,
        product_image_url: str = ""
    ) -> Dict[str, Any]:
        """Process a single video through the entire pipeline (unified OpenAI flow).
        
        NEW FLOW:
        1. Download video
        2. Run PySceneDetect for initial scene timestamps
        3. Extract frames for ENTIRE video (1/sec)
        4. Send ALL frames + timestamps to OpenAI (single unified call)
        5. OpenAI returns corrected timestamps + scene prompts
        6. Process each scene with Nano Banana + Runway
        7. Concatenate and finalize
        8. Generate new music with Suno (if original has background music)
        9. Add subtitles with ZapCap (if requested)
        
        If article_text is provided, the pipeline adapts content to match the article.
        
        Args:
            video_url: URL of the video to process.
            row_num: Row number in the Google Sheet (1-based).
            headers: List of column headers.
            manual_instructions: Optional custom instructions for OpenAI analysis.
            cta_button: Whether to include a CTA button in image prompts.
            cta_text: Text for the CTA button.
            add_subtitles: Whether to add subtitles to the final video.
            article_text: Optional article content for content adaptation.
            vertical: Optional vertical/offer name for content adaptation.
            subtitle_language: Optional language code for ZapCap subtitles (e.g., 'de', 'en').
            manual_vo_text: Optional manual text for voice-over (overrides generated VO).
            manual_music_link: Optional manual music URL (overrides Suno generation).
            voice_id: Optional custom ElevenLabs voice ID (uses default if empty).
            
        Returns:
            Dict with processing results.
        """
        result = {
            "row": row_num,
            "video_url": video_url,
            "success": False,
            "scenes_processed": 0,
            "errors": [],
            "manual_instructions": manual_instructions,
            "cta_button": cta_button,
            "cta_text": cta_text,
            "add_subtitles": add_subtitles,
            "article_text": article_text,
            "vertical": vertical,
            "subtitle_language": subtitle_language,
            "manual_vo_text": manual_vo_text,
            "manual_music_link": manual_music_link
        }
        
        # =================================================================
        # ARTICLE ADAPTATION SETUP
        # =================================================================
        has_article_adaptation = bool(article_text.strip())
        article_language = "en"  # Default to English
        
        if has_article_adaptation:
            logger.info(f"📰 [Row {row_num}] Article adaptation mode enabled")
            # Priority: 1) subtitle_language from sheet, 2) detect from article text
            if subtitle_language:
                article_language = subtitle_language
                logger.info(f"🌍 Detected language: {article_language} (from sheet Language column)")
            else:
                article_language = detect_language(article_text)
                logger.info(f"🌍 Detected language: {article_language}")
            logger.info(f"   Detected article language: {article_language}")
            if vertical:
                logger.info(f"   Vertical/Offer: {vertical}")
        
        # Set Rendi API key for cloud fallback
        FFmpegProcessor.set_rendi_api_key(config.RENDI_API_KEY)
        
        # Check if FFmpeg is available
        ffmpeg_available = FFmpegProcessor.check_ffmpeg_installed()
        
        # Create temp directory for processing (using mkdtemp for manual control)
        temp_dir = tempfile.mkdtemp()
        
        try:
            video_path = None
            
            # =================================================================
            # STEP 1: Download video
            # =================================================================
            logger.info(f"📥 [Row {row_num}] Step 1: Downloading video...")
            video_path = os.path.join(temp_dir, "input_video.mp4")
            download_success = FFmpegProcessor.download_video(video_url, video_path)
            
            if not download_success:
                logger.error(f"❌ [Row {row_num}] Failed to download video")
                result["errors"].append("Failed to download video")
                return result
            
            # =================================================================
            # STEP 2: Get video duration and run PySceneDetect
            # =================================================================
            logger.info(f"🎬 [Row {row_num}] Step 2: Detecting scenes with PySceneDetect...")
            
            # Get video duration using OpenCV (no FFmpeg needed)
            video_duration = 0
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0 and frame_count > 0:
                        video_duration = frame_count / fps
                    cap.release()
            except Exception as e:
                logger.warning(f"⚠️ Could not get duration via OpenCV: {e}")
            
            if video_duration <= 0:
                video_duration = self.rendi_service.get_video_duration_cloud(video_url)
            
            logger.info(f"   Video duration: {video_duration:.2f}s")
            
            # Run PySceneDetect for initial scene timestamps
            if PYSCENEDETECT_AVAILABLE:
                pyscenedetect_timestamps = FFmpegProcessor.detect_scenes(
                    video_path,
                    threshold=config.PYSCENEDETECT_THRESHOLD,
                    min_scene_duration=config.PYSCENEDETECT_MIN_SCENE_DURATION,
                    use_adaptive=config.PYSCENEDETECT_USE_ADAPTIVE
                )
            else:
                # Fallback: divide into equal segments
                num_segments = min(5, int(video_duration / 3))
                segment_duration = video_duration / num_segments
                pyscenedetect_timestamps = [i * segment_duration for i in range(num_segments)]
            
            logger.info(f"   PySceneDetect found {len(pyscenedetect_timestamps)} initial scenes")
            
            # =================================================================
            # STEP 3: Extract frames for ENTIRE video (1 per second)
            # =================================================================
            logger.info(f"🎬 [Row {row_num}] Step 3: Extracting frames for entire video...")
            
            frames_dir = os.path.join(temp_dir, "all_frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Extract frames (1 per second)
            frames_with_timestamps = FFmpegProcessor.extract_frames_entire_video(
                video_path=video_path,
                video_duration=video_duration,
                output_dir=frames_dir,
                fps=5  # 1 frame per second
            )
            
            if not frames_with_timestamps:
                # Fallback to cloud extraction
                logger.info("🌐 Falling back to cloud frame extraction...")
                frames_with_timestamps = FFmpegProcessor.extract_frames_entire_video_cloud(
                    video_url=video_url,
                    video_duration=video_duration,
                    output_dir=frames_dir,
                    rendi_api_key=config.RENDI_API_KEY,
                    fps=5
                )
            
            if not frames_with_timestamps:
                result["errors"].append("Failed to extract frames")
                return result
            
            logger.info(f"   Extracted {len(frames_with_timestamps)} frames")
            
            # =================================================================
            # STEP 3.35: EARLY AUDIO EXTRACTION & TRANSCRIPTION (NEW)
            # =================================================================
            # Extract and transcribe audio BEFORE Gemini analysis so we can
            # understand the audio-visual relationship and what's being said
            # =================================================================
            original_transcript = ""
            audio_path = os.path.join(temp_dir, "original_audio.mp3")
            
            logger.info(f"🎤 [Row {row_num}] Step 3.35: Extracting and transcribing audio...")
            
            try:
                # Try local extraction first
                audio_extracted = FFmpegProcessor.extract_audio(video_path, audio_path)
                
                if not audio_extracted:
                    # Fallback to cloud extraction
                    logger.info("🌐 Extracting audio via cloud...")
                    original_audio_url = FFmpegProcessor.extract_audio_from_url(
                        video_url=video_url,
                        output_path=audio_path,
                        rendi_api_key=config.RENDI_API_KEY
                    )
                    
                    if original_audio_url:
                        try:
                            response = requests.get(original_audio_url, timeout=60)
                            response.raise_for_status()
                            with open(audio_path, 'wb') as f:
                                f.write(response.content)
                            audio_extracted = True
                        except Exception:
                            pass
                
                # Transcribe if we have audio
                if audio_extracted and os.path.exists(audio_path):
                    original_transcript = self.elevenlabs_service.get_transcript_from_audio(audio_path) or ""
                    if original_transcript:
                        logger.info(f"✅ [Row {row_num}] Transcribed: {original_transcript[:100]}...")
                    else:
                        logger.info(f"ℹ️ [Row {row_num}] No speech detected in video")
                else:
                    logger.warning(f"⚠️ [Row {row_num}] Could not extract audio for transcription")
                    
            except Exception as audio_err:
                logger.warning(f"⚠️ [Row {row_num}] Audio extraction error: {audio_err}")
            
            # =================================================================
            # STEP 3.4: GEMINI COMPREHENSIVE VIDEO ANALYSIS (Native Video)
            # =================================================================
            # Gemini analyzes the ENTIRE video including:
            # - What's shown visually in each scene
            # - What's being said (from transcript)
            # - How audio relates to visuals
            # - Product appearance and usage
            # - Style, tone, and mood
            # =================================================================
            gemini_analysis = None
            
            if self.gemini_service and self.gemini_service.initialized:
                logger.info(f"🔮 [Row {row_num}] Step 3.4: Running Gemini comprehensive video analysis...")
                
                try:
                    # Prepare article content from the article_text parameter
                    article_content_for_gemini = {
                        'title': vertical or "",  # Use vertical as context
                        'first_paragraph': article_text[:500] if article_text else "",
                        'free_text': article_text or ""
                    }
                    
                    # Run comprehensive video analysis WITH TRANSCRIPT
                    # Pass subtitle_language to ensure VO script is in correct language
                    # Pass article_related_to_video to control prompt generation strategy:
                    # - True (Yes): Article is similar to video - adapt video for new offer/language
                    # - False (No): Article is different - keep video style but create new content
                    gemini_analysis = self.gemini_service.analyze_video_comprehensive(
                        video_path=video_path,
                        article_content=article_content_for_gemini,
                        manual_instructions=manual_instructions,
                        target_language=subtitle_language or "en",  # Use Language column or default to English
                        original_transcript=original_transcript,  # Pass transcript to Gemini
                        article_related_to_video=article_related_to_video  # Controls prompt generation strategy
                    )
                    
                    if gemini_analysis and gemini_analysis.get("scenes"):
                        logger.info(f"✅ [Row {row_num}] Gemini analysis complete (NEW FORMAT):")
                        logger.info(f"   - Video type: {gemini_analysis.get('video_story', {}).get('type', 'unknown')}")
                        logger.info(f"   - Scenes: {len(gemini_analysis.get('scenes', []))}")
                        logger.info(f"   - Product detected: {gemini_analysis.get('product', {}).get('detected', False)}")
                        logger.info(f"   - Style: {gemini_analysis.get('style', {}).get('aesthetic', 'modern')}")
                        
                        # Log new VO script
                        new_vo = gemini_analysis.get("new_voiceover", {})
                        if new_vo.get("full_script"):
                            logger.info(f"   - New VO: {new_vo.get('full_script', '')[:60]}...")
                            logger.info(f"   - VO style: {new_vo.get('style', 'unknown')}")
                        
                        # Log first scene prompt
                        scenes = gemini_analysis.get("scenes", [])
                        if scenes:
                            first_prompt = scenes[0].get('prompts', {}).get('image_prompt', '')
                            logger.info(f"   - First scene prompt: {first_prompt[:80]}...")
                        
                        # Store style prefix for later use
                        style_prefix = gemini_analysis.get("style", {}).get("style_prefix", "")
                        if style_prefix:
                            logger.info(f"   - Style prefix: {style_prefix[:100]}...")
                        
                        # =============================================================
                        # Extract and save product frames based on Gemini recommendations
                        # =============================================================
                        product_frames_urls = []
                        product = gemini_analysis.get("product", {})
                        recommended_timestamps = product.get("best_frame_timestamps", [])
                        
                        if recommended_timestamps and product.get("detected"):
                            logger.info(f"📸 [Row {row_num}] Extracting {len(recommended_timestamps)} product frames...")
                            
                            for i, timestamp in enumerate(recommended_timestamps[:5]):  # Max 5 frames
                                try:
                                    # Parse timestamp (format: "0:02" or "1:30")
                                    parts = timestamp.replace("s", "").split(":")
                                    if len(parts) == 2:
                                        seconds = int(parts[0]) * 60 + float(parts[1])
                                    else:
                                        seconds = float(parts[0])
                                    
                                    # Find the closest frame
                                    # frames_with_timestamps is a list of tuples (timestamp, path)
                                    frame_idx = int(seconds * 5)  # 5 fps
                                    if frame_idx < len(frames_with_timestamps):
                                        # Access tuple: (timestamp, path)
                                        frame_timestamp, frame_path = frames_with_timestamps[frame_idx]
                                        
                                        # Read frame file and upload to S3
                                        if frame_path and os.path.exists(frame_path):
                                            with open(frame_path, 'rb') as f:
                                                frame_data = f.read()
                                            
                                            product_frame_key = f"product_references/row_{row_num}_product_{i+1}.jpg"
                                            frame_url = self.s3_service.upload_image_bytes(
                                                image_data=frame_data,
                                                key_name=product_frame_key,
                                                make_public=True
                                            )
                                            
                                            if frame_url:
                                                product_frames_urls.append(frame_url)
                                                logger.info(f"   ✅ Product frame {i+1} saved: {frame_url[:60]}...")
                                except Exception as frame_err:
                                    logger.warning(f"   ⚠️ Could not extract frame at {timestamp}: {frame_err}")
                            
                            if product_frames_urls:
                                gemini_analysis["product_frame_urls"] = product_frames_urls
                                logger.info(f"✅ [Row {row_num}] Saved {len(product_frames_urls)} product reference frames")
                    else:
                        logger.warning(f"⚠️ [Row {row_num}] Gemini analysis returned empty results")
                        
                except Exception as gemini_error:
                    logger.warning(f"⚠️ [Row {row_num}] Gemini analysis failed: {gemini_error}")
                    gemini_analysis = None
            else:
                logger.info(f"ℹ️ [Row {row_num}] Gemini not available, using GPT-4o for analysis")
            
            # =================================================================
            # STEP 3.5: Product Detection with Context Understanding (ENHANCED)
            # =================================================================
            # If "Product image(optional)" has a link: use it as product_reference_url and SKIP ONLY
            # generating the clean product image. Still run full product detection (Gemini/GPT) for
            # product_info (usage_contexts, narrative, scene_plan) so scene images are generated.
            # If column empty: detect product, generate clean product image, write URL to column.
            # =================================================================
            product_info = {"has_product": False}
            product_reference_url = None
            if product_image_url and product_image_url.strip():
                product_reference_url = product_image_url.strip()
                logger.info(f"📷 [Row {row_num}] Product image from sheet will be used as reference (skip clean-image generation only)")
            
            # Use Gemini product info if available, otherwise use GPT-4o
            gemini_product = gemini_analysis.get("product", {}) if gemini_analysis else {}
            if gemini_product.get("detected"):
                logger.info(f"🔍 [Row {row_num}] Step 3.5: Using Gemini product analysis (skipping GPT)...")
                
                # Convert NEW Gemini format to our internal format
                product_info = {
                    "has_product": True,
                    "product_detected": gemini_product.get("type", "product"),
                    "overall_confidence": 0.95,  # Gemini provides high confidence
                    "product_description": gemini_product.get("visual_description", ""),
                    "product_purpose": gemini_product.get("purpose", ""),
                    "product_usage_method": gemini_product.get("usage_method", ""),
                    "product_details": {
                        "application_rules": gemini_product.get("application_rules", ""),
                    },
                    "usage_contexts": [],  # Will be inferred from scenes
                    "best_frame_index": 0
                }
                
                logger.info(f"   Product type: {product_info.get('product_detected')}")
                logger.info(f"   Description: {product_info.get('product_description', '')[:100]}...")
                logger.info(f"   Purpose: {product_info.get('product_purpose', '')[:100]}...")
                logger.info(f"   Usage method: {product_info.get('product_usage_method', '')[:100]}...")
                logger.info(f"   Application rules: {product_info.get('product_details', {}).get('application_rules', '')[:100]}...")
                
                # Generate clean product image only if we don't already have one from sheet
                if not product_reference_url:
                    product_frame_urls = gemini_analysis.get("product_frame_urls", [])
                    if product_frame_urls:
                        logger.info(f"🧹 [Row {row_num}] Generating clean product image from {len(product_frame_urls)} reference frames...")
                        
                        product_desc = product_info.get("product_description", "")
                        clean_product_url = self.kie_service.generate_clean_product_image(
                            reference_image_urls=product_frame_urls[:3],
                            product_description=product_desc
                        )
                        
                        if clean_product_url:
                            product_reference_url = clean_product_url
                            logger.info(f"   ✅ Clean product image generated: {product_reference_url[:60]}...")
                            try:
                                clean_product_key = f"product_references/row_{row_num}_clean_product.png"
                                import requests as req
                                clean_response = req.get(clean_product_url, timeout=30)
                                if clean_response.status_code == 200:
                                    s3_clean_url = self.s3_service.upload_image_bytes(
                                        image_data=clean_response.content,
                                        key_name=clean_product_key,
                                        make_public=True
                                    )
                                    if s3_clean_url:
                                        product_reference_url = s3_clean_url
                                        logger.info(f"   ✅ Clean product image saved to S3: {s3_clean_url[:60]}...")
                            except Exception as upload_err:
                                logger.warning(f"   ⚠️ Could not save clean product to S3: {upload_err}")
                        else:
                            product_reference_url = product_frame_urls[0]
                            logger.warning(f"   ⚠️ Clean product generation failed, using raw frame: {product_reference_url[:60]}...")
                    
                    if product_reference_url:
                        try:
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.PRODUCT_REFERENCE_COLUMN,
                                value=product_reference_url,
                                headers=headers
                            )
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.PRODUCT_IMAGE_COLUMN,
                                value=product_reference_url,
                                headers=headers
                            )
                            logger.info(f"   📷 Row {row_num}: Product image URL written to sheet")
                        except Exception as e:
                            logger.warning(f"   ⚠️ Could not write product image to sheet: {e}")
                
            elif config.ENABLE_PRODUCT_DETECTION and frames_with_timestamps:
                logger.info(f"🔍 [Row {row_num}] Step 3.5: Detecting product and analyzing usage context...")
                
                try:
                    # Select frames spread across the entire video for context understanding
                    # Instead of just first N frames, sample frames from beginning, middle, and end
                    total_frames = len(frames_with_timestamps)
                    detection_frame_count = min(config.PRODUCT_DETECTION_FRAMES, total_frames)
                    
                    if total_frames >= 10:
                        # Evenly distribute frames across the entire video
                        # For 60 frames from 136 total: sample every ~2.3 frames
                        step = max(1, total_frames / detection_frame_count)
                        indices = []
                        for i in range(detection_frame_count):
                            idx = min(int(i * step), total_frames - 1)
                            if idx not in indices:
                                indices.append(idx)
                        
                        # Ensure we always include first and last frame
                        if 0 not in indices:
                            indices[0] = 0
                        if total_frames - 1 not in indices:
                            indices[-1] = total_frames - 1
                        
                        indices = sorted(set(indices))
                        detection_frames = [frames_with_timestamps[i][1] for i in indices]
                        logger.info(f"   Analyzing {len(detection_frames)} frames spread across entire video (every ~{step:.1f} frames)")
                    else:
                        # Not enough frames, use all available
                        detection_frames = [f[1] for f in frames_with_timestamps[:detection_frame_count]]
                        logger.info(f"   Using first {len(detection_frames)} frames")
                    
                    # Comprehensive video analysis: product + narrative + audio correlation
                    logger.info(f"   🎬 Running comprehensive video analysis with {len(detection_frames)} frames + audio transcript...")
                    product_info = self.openai_service.detect_product_in_frames(
                        frame_paths=detection_frames,
                        min_confidence=config.PRODUCT_MIN_CONFIDENCE,
                        audio_transcript=original_transcript,
                        video_duration=video_duration
                    )
                    
                    # Log comprehensive video analysis results
                    if product_info.get("has_product"):
                        # Log video narrative
                        video_narrative = product_info.get("video_narrative", {})
                        if video_narrative:
                            logger.info(f"   🎬 VIDEO NARRATIVE:")
                            logger.info(f"      Type: {video_narrative.get('video_type', 'unknown')}")
                            logger.info(f"      Hook: {video_narrative.get('opening_hook', '')[:80]}...")
                            logger.info(f"      Story: {video_narrative.get('main_story', '')[:80]}...")
                            logger.info(f"      Style: {video_narrative.get('style', 'unknown')}")
                        
                        # Log sequential breakdown
                        sequential = product_info.get("sequential_breakdown", [])
                        if sequential:
                            logger.info(f"   📊 SEQUENTIAL BREAKDOWN ({len(sequential)} segments):")
                            for seg in sequential[:3]:
                                logger.info(f"      {seg.get('segment', '?')}: {seg.get('what_happens', '')[:60]}...")
                        
                        # Log audio-visual sync
                        av_sync = product_info.get("audio_visual_sync", [])
                        if av_sync:
                            logger.info(f"   🔊 AUDIO-VISUAL SYNC ({len(av_sync)} segments):")
                            for sync in av_sync[:2]:
                                logger.info(f"      VO: \"{sync.get('vo_text', '')[:50]}...\"")
                                logger.info(f"      Visual: {sync.get('visual_description', '')[:50]}...")
                        
                        # Log usage contexts
                        usage_contexts = product_info.get("usage_contexts", [])
                        if usage_contexts:
                            context_types = [c.get("context_type") for c in usage_contexts]
                            logger.info(f"   📋 Usage contexts found: {', '.join(context_types)}")
                        
                        # Log detailed product info
                        product_desc = product_info.get("product_description", "")
                        if product_desc:
                            logger.info(f"   📝 Product description ({len(product_desc)} chars): {product_desc[:150]}...")
                        
                        product_purpose = product_info.get("product_purpose", "")
                        if product_purpose:
                            logger.info(f"   🎯 Product purpose: {product_purpose[:150]}...")
                        
                        product_usage = product_info.get("product_usage_method", "")
                        if product_usage:
                            logger.info(f"   🔧 Usage method: {product_usage[:150]}...")
                        
                        # Log detailed product details
                        product_details = product_info.get("product_details", {})
                        if product_details:
                            shape = product_details.get("shape", "")
                            dims = product_details.get("dimensions", "")
                            if shape or dims:
                                logger.info(f"   📐 Shape/Size: {shape} | {dims}")
                        
                        # Log recreation notes
                        recreation_notes = product_info.get("recreation_notes", "")
                        if recreation_notes:
                            logger.info(f"   💡 Recreation notes: {recreation_notes[:150]}...")
                    
                    # Generate clean product image only if we don't already have one from sheet
                    if product_info.get("has_product") and not product_reference_url:
                        # Get up to 3 best frames for reference
                        best_frame_index = product_info.get("best_frame_index", 0)
                        
                        # Collect reference frames (best frame + adjacent frames)
                        ref_frame_paths = []
                        if best_frame_index is not None and 0 <= best_frame_index < len(detection_frames):
                            ref_frame_paths.append(detection_frames[best_frame_index])
                            if best_frame_index > 0:
                                ref_frame_paths.append(detection_frames[best_frame_index - 1])
                            if best_frame_index < len(detection_frames) - 1:
                                ref_frame_paths.append(detection_frames[best_frame_index + 1])
                        
                        if ref_frame_paths:
                            ref_frame_urls = []
                            for i, frame_path in enumerate(ref_frame_paths[:3]):
                                try:
                                    with open(frame_path, 'rb') as f:
                                        frame_data = f.read()
                                    frame_key = f"product_references/row_{row_num}_gpt_ref_{i+1}.jpg"
                                    frame_url = self.s3_service.upload_image_bytes(
                                        image_data=frame_data,
                                        key_name=frame_key,
                                        make_public=True
                                    )
                                    if frame_url:
                                        ref_frame_urls.append(frame_url)
                                except Exception as frame_err:
                                    logger.warning(f"   ⚠️ Could not upload frame {i+1}: {frame_err}")
                            
                            if ref_frame_urls:
                                logger.info(f"🧹 [Row {row_num}] Generating clean product image from {len(ref_frame_urls)} GPT reference frames...")
                                product_desc = product_info.get("product_description", "")
                                clean_product_url = self.kie_service.generate_clean_product_image(
                                    reference_image_urls=ref_frame_urls,
                                    product_description=product_desc
                                )
                                
                                if clean_product_url:
                                    try:
                                        import requests as req
                                        clean_response = req.get(clean_product_url, timeout=30)
                                        if clean_response.status_code == 200:
                                            clean_key = f"product_references/row_{row_num}_clean_product.png"
                                            s3_clean_url = self.s3_service.upload_image_bytes(
                                                image_data=clean_response.content,
                                                key_name=clean_key,
                                                make_public=True
                                            )
                                            if s3_clean_url:
                                                product_reference_url = s3_clean_url
                                                logger.info(f"   ✅ Clean product image saved: {product_reference_url[:60]}...")
                                    except Exception as upload_err:
                                        product_reference_url = clean_product_url
                                        logger.warning(f"   ⚠️ Could not save to S3, using original: {upload_err}")
                                else:
                                    product_reference_url = ref_frame_urls[0]
                                    logger.warning(f"   ⚠️ Clean product generation failed, using raw frame")
                            else:
                                logger.warning(f"⚠️ [Row {row_num}] Failed to upload reference frames")
                        else:
                            logger.warning(f"⚠️ [Row {row_num}] No valid reference frames found")
                    
                    if product_info.get("has_product"):
                        # Store product info in result for tracking
                        result["product_detected"] = True
                        result["product_type"] = product_info.get("product_detected", "unknown")
                        result["product_confidence"] = product_info.get("overall_confidence", 0)
                        
                        # Write product detection results to Google Sheet (ENHANCED)
                        try:
                            # Basic product info
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.PRODUCT_DETECTED_COLUMN,
                                value=product_info.get("product_detected", "unknown"),
                                headers=headers
                            )
                            if product_reference_url:
                                self._update_sheet_cell(
                                    row_num=row_num,
                                    column=config.PRODUCT_REFERENCE_COLUMN,
                                    value=product_reference_url,
                                    headers=headers
                                )
                                # Also write to Product image (optional) so next run uses this link and skips generation
                                self._update_sheet_cell(
                                    row_num=row_num,
                                    column=config.PRODUCT_IMAGE_COLUMN,
                                    value=product_reference_url,
                                    headers=headers
                                )
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.PRODUCT_CONFIDENCE_COLUMN,
                                value=f"{product_info.get('overall_confidence', 0):.2f}",
                                headers=headers
                            )
                            
                            # NEW: Write product purpose (what the product does)
                            product_purpose = product_info.get("product_purpose", "")
                            if product_purpose:
                                self._update_sheet_cell(
                                    row_num=row_num,
                                    column=config.PRODUCT_PURPOSE_COLUMN,
                                    value=product_purpose[:500],  # Truncate if too long
                                    headers=headers
                                )
                            
                            # NEW: Write product usage method (how it's applied)
                            product_usage = product_info.get("product_usage_method", "")
                            if product_usage:
                                self._update_sheet_cell(
                                    row_num=row_num,
                                    column=config.PRODUCT_USAGE_COLUMN,
                                    value=product_usage[:500],
                                    headers=headers
                                )
                            
                            # NEW: Write usage contexts (how product appears in different scenes)
                            usage_contexts = product_info.get("usage_contexts", [])
                            if usage_contexts:
                                context_summary = ", ".join([
                                    f"{c.get('context_type')}: {c.get('description', '')[:50]}"
                                    for c in usage_contexts[:5]
                                ])
                                self._update_sheet_cell(
                                    row_num=row_num,
                                    column=config.PRODUCT_CONTEXTS_COLUMN,
                                    value=context_summary[:500],
                                    headers=headers
                                )
                            
                            logger.info(f"✅ [Row {row_num}] Product detection results (incl. context) written to sheet")
                        except Exception as sheet_error:
                            logger.warning(f"⚠️ [Row {row_num}] Failed to write product info to sheet: {sheet_error}")
                    else:
                        logger.info(f"ℹ️ [Row {row_num}] No product detected, continuing with standard flow")
                        
                except Exception as e:
                    logger.error(f"❌ [Row {row_num}] Product detection failed: {e}")
                    logger.info(f"   Continuing with standard flow...")
                    product_info = {"has_product": False, "error": str(e)}
            
            # =================================================================
            # STEP 3.55: Video Style Analysis
            # =================================================================
            # Comprehensive visual style analysis to match the original video:
            # - Color palette, lighting, composition
            # - Camera style, mood, atmosphere
            # - Creates style guide for prompt generation
            # NOTE: Uses Gemini analysis if available, otherwise falls back to GPT-4o
            # =================================================================
            video_style = {}
            
            if gemini_analysis and gemini_analysis.get("style"):
                # Use Gemini's NEW visual style analysis
                logger.info(f"🎨 [Row {row_num}] Step 3.55: Using Gemini video style (skipping GPT)...")
                
                gemini_style = gemini_analysis.get("style", {})
                video_style = {
                    "mood_atmosphere": {
                        "overall_mood": gemini_style.get("mood", "professional")
                    },
                    "overall_aesthetic": gemini_style.get("aesthetic", "modern"),
                    "lighting": gemini_style.get("lighting", "natural"),
                    "style_prompt_prefix": gemini_style.get("style_prefix", "")
                }
                
                # Store key style elements in result for reference
                result["video_style"] = {
                    "aesthetic": gemini_style.get("aesthetic", "modern"),
                    "lighting": gemini_style.get("lighting", "natural"),
                    "mood": gemini_style.get("mood", "professional"),
                    "style_prefix": gemini_style.get("style_prefix", "")
                }
                
                logger.info(f"✅ [Row {row_num}] Video style from Gemini:")
                logger.info(f"   - Aesthetic: {gemini_style.get('aesthetic', 'modern')}")
                logger.info(f"   - Lighting: {gemini_style.get('lighting', 'natural')}")
                logger.info(f"   - Mood: {gemini_style.get('mood', 'professional')}")
                logger.info(f"   - Style prefix: {gemini_style.get('style_prefix', 'N/A')[:60]}...")
                
            elif frames_with_timestamps:
                # Fallback to GPT-4o frame analysis
                logger.info(f"🎨 [Row {row_num}] Step 3.55: Analyzing video visual style with GPT-4o...")
                
                try:
                    all_frame_paths = [f[1] for f in frames_with_timestamps]
                    video_style = self.openai_service.analyze_video_style(
                        frame_paths=all_frame_paths,
                        video_duration=video_duration
                    )
                    
                    if video_style and not video_style.get("error"):
                        logger.info(f"✅ [Row {row_num}] Video style analyzed successfully")
                        
                        # Store key style elements in result for reference
                        result["video_style"] = {
                            "color_temperature": video_style.get("color_palette", {}).get("color_temperature", "neutral"),
                            "lighting": video_style.get("lighting", {}).get("type", "natural"),
                            "framing": video_style.get("composition", {}).get("primary_framing", "medium"),
                            "mood": video_style.get("mood_atmosphere", {}).get("overall_mood", "professional")
                        }
                    else:
                        logger.warning(f"⚠️ [Row {row_num}] Style analysis failed, using defaults")
                        
                except Exception as style_error:
                    logger.warning(f"⚠️ [Row {row_num}] Style analysis error: {style_error}")
                    video_style = {}
            
            # =================================================================
            # STEP 3.6: Video Structure Analysis
            # =================================================================
            # Analyze video narrative structure considering:
            # - Article content (Free text, Title, 1stP, Rest of Content)
            # - Manual instructions
            # - Product information (if detected)
            # NOTE: Uses Gemini analysis if available, otherwise falls back to GPT-4o
            # =================================================================
            video_structure = {"video_structure": "unknown", "scene_plan": []}
            
            if gemini_analysis and gemini_analysis.get("scenes"):
                # Use Gemini's NEW scene structure (skipping GPT)
                logger.info(f"📊 [Row {row_num}] Step 3.6: Using Gemini scenes (skipping GPT)...")
                
                gemini_scenes = gemini_analysis.get("scenes", [])
                video_story = gemini_analysis.get("video_story", {})
                
                # Convert Gemini scenes to our scene_plan format
                scene_plan = []
                for scene in gemini_scenes:
                    understanding = scene.get("understanding", {})
                    prompts = scene.get("prompts", {})
                    scene_plan.append({
                        "scene_number": scene.get("scene_number", 0),
                        "narrative_role": understanding.get("narrative_role", "content"),
                        "key_message": understanding.get("what_happens", "")[:100],
                        "visual_suggestion": prompts.get("image_prompt", ""),
                        "motion_prompt": prompts.get("motion_prompt", ""),
                        "product_visible": understanding.get("product_visible", False),
                        "product_action": understanding.get("product_action", ""),
                        "subject_appearance": understanding.get("subject_appearance", "")
                    })
                
                video_structure = {
                    "video_structure": video_story.get("type", "advertisement"),
                    "narrative_summary": video_story.get("one_sentence_summary", ""),
                    "scene_plan": scene_plan,
                    "has_subject_changes": video_story.get("subject_changes", {}).get("has_visible_change", False),
                    "start_state": video_story.get("subject_changes", {}).get("start_state", ""),
                    "end_state": video_story.get("subject_changes", {}).get("end_state", "")
                }
                
                logger.info(f"✅ [Row {row_num}] Video structure from Gemini:")
                logger.info(f"   - Type: {video_structure.get('video_structure')}")
                logger.info(f"   - Scenes: {len(scene_plan)}")
                logger.info(f"   - Subject changes: {video_structure.get('has_subject_changes')}")
                if scene_plan:
                    for sp in scene_plan[:3]:
                        logger.info(f"      Scene {sp.get('scene_number')}: {sp.get('narrative_role')} - {sp.get('key_message', '')[:40]}...")
                    if len(scene_plan) > 3:
                        logger.info(f"      ... and {len(scene_plan) - 3} more scenes")
                
            elif config.ENABLE_PRODUCT_DETECTION and frames_with_timestamps:
                # Fallback to GPT-4o analysis
                logger.info(f"📊 [Row {row_num}] Step 3.6: Analyzing video structure with GPT-4o...")
                
                try:
                    # Prepare article content dict from available parameters
                    article_content = {
                        'free_text': article_text or "",
                        'title': vertical or "",
                        'first_paragraph': article_text[:500] if article_text else "",
                        'rest_content': article_text[500:] if article_text and len(article_text) > 500 else ""
                    }
                    
                    # Get frame paths for analysis
                    all_frame_paths = [f[1] for f in frames_with_timestamps]
                    
                    # Analyze video structure
                    video_structure = self.openai_service.analyze_video_structure(
                        frame_paths=all_frame_paths,
                        article_content=article_content,
                        manual_instructions=manual_instructions,
                        product_info=product_info
                    )
                    
                    # Log structure results
                    if video_structure.get("video_structure") != "unknown":
                        logger.info(f"✅ [Row {row_num}] Video structure: {video_structure.get('video_structure')}")
                        logger.info(f"   Narrative: {video_structure.get('narrative_summary', '')[:80]}...")
                        scene_plan = video_structure.get("scene_plan", [])
                        if scene_plan:
                            logger.info(f"   Scene plan ({len(scene_plan)} scenes):")
                            for sp in scene_plan[:3]:
                                logger.info(f"      Scene {sp.get('scene_number')}: {sp.get('narrative_role')} - {sp.get('key_message', '')[:40]}...")
                            if len(scene_plan) > 3:
                                logger.info(f"      ... and {len(scene_plan) - 3} more scenes")
                    else:
                        logger.info(f"ℹ️ [Row {row_num}] Could not determine video structure")
                        
                except Exception as e:
                    logger.error(f"❌ [Row {row_num}] Video structure analysis failed: {e}")
                    logger.info(f"   Continuing with standard flow...")
            
            # =================================================================
            # STEP 4: Scene Analysis (Gemini or OpenAI fallback)
            # =================================================================
            
            # Check if Gemini already provided all prompts
            gemini_scenes = gemini_analysis.get("scenes", []) if gemini_analysis else []
            gemini_has_prompts = gemini_scenes and all(
                s.get("prompts", {}).get("image_prompt") for s in gemini_scenes
            )
            
            if gemini_has_prompts:
                # USE GEMINI PROMPTS DIRECTLY - Skip OpenAI!
                logger.info(f"🎯 [Row {row_num}] Step 4: Using Gemini prompts directly (skipping OpenAI)...")
                
                # Convert Gemini scenes to our format
                corrected_scenes = []
                scene_prompts = []
                
                for gs in gemini_scenes:
                    scene_num = gs.get("scene_number", 0)
                    understanding = gs.get("understanding", {})
                    prompts = gs.get("prompts", {})
                    
                    # Calculate approximate timestamps based on scene number
                    # Use video_structure's scene_plan if available for better timing
                    corrected_scenes.append({
                        "scene_num": scene_num,
                        "duration": gs.get("duration_seconds", 3)
                    })
                    
                    scene_prompts.append({
                        "scene_num": scene_num,
                        "image_prompt": prompts.get("image_prompt", ""),
                        "motion_prompt": prompts.get("motion_prompt", ""),
                        "visible_elements": prompts.get("visible_elements", []),
                        "narrative_role": understanding.get("narrative_role", "content"),
                        "story_beat": understanding.get("story_beat", ""),
                        "transition_logic": understanding.get("transition_logic", "")
                    })
                
                logger.info(f"   ✅ Using {len(gemini_scenes)} Gemini scene prompts")
                for sp in scene_prompts[:3]:
                    logger.info(f"      Scene {sp['scene_num']}: {sp.get('image_prompt', '')[:50]}...")
            else:
                # FALLBACK: Use OpenAI to generate prompts
                logger.info(f"🤖 [Row {row_num}] Step 4: Fallback to OpenAI analysis...")
                if cta_button and cta_text:
                    logger.info(f"   🔘 [Row {row_num}] Including CTA button in prompts: '{cta_text}'")
                
                openai_result = self.openai_service.analyze_full_video(
                    frame_paths_with_timestamps=frames_with_timestamps,
                    pyscenedetect_timestamps=pyscenedetect_timestamps,
                    video_duration=video_duration,
                    manual_instructions=manual_instructions,
                    cta_button=cta_button,
                    cta_text=cta_text,
                    row_num=row_num,
                    article_text=article_text,
                    vertical=vertical,
                    article_language=article_language,
                    article_related_to_video=article_related_to_video
                )
                
                corrected_scenes = openai_result.get("corrected_scenes", [])
                scene_prompts = openai_result.get("scene_prompts", [])
            
            if not corrected_scenes:
                logger.error(f"❌ [Row {row_num}] No corrected scenes available")
                result["errors"].append("No corrected scenes available")
                return result
            
            logger.info(f"   [Row {row_num}] Using {len(corrected_scenes)} scenes with prompts")
            
            # Write prompts to Google Sheet
            for prompt_data in scene_prompts[:config.MAX_SCENES]:
                scene_num = prompt_data.get("scene_num", 0)
                if scene_num > 0:
                    # Write image prompt
                    first_prompt = prompt_data.get("image_prompt", "")
                    if first_prompt:
                        self._update_sheet_cell(
                            row_num=row_num,
                            column=config.SCENE_FIRST_PROMPT_PREFIX.replace("{n}", str(scene_num)),
                            value=first_prompt[:4000],  # Truncate if needed
                            headers=headers
                        )
                    
                    # Write motion prompt
                    second_prompt = prompt_data.get("motion_prompt", "")
                    if second_prompt:
                        self._update_sheet_cell(
                            row_num=row_num,
                            column=config.SCENE_SECOND_PROMPT_PREFIX.replace("{n}", str(scene_num)),
                            value=second_prompt,
                            headers=headers
                        )
            
            # =================================================================
            # STEP 5: Process SCENES + AUDIO in PARALLEL
            # =================================================================
            # We run two parallel workflows:
            # A) Scene processing: Nano Banana → Runway for each scene
            # B) Audio processing: Extract → Detect speech → ElevenLabs + Suno
            # This saves significant time as audio takes ~2 minutes
            # =================================================================
            logger.info(f"🚀 [Row {row_num}] Step 5: Processing SCENES + AUDIO in PARALLEL...")
            
            # Build scene data from OpenAI results or Gemini scenes
            scene_data = []
            cumulative_time = 0.0  # Track cumulative time for Gemini scenes
            
            for i, scene in enumerate(corrected_scenes[:config.MAX_SCENES]):
                scene_num = scene.get("scene_num", i + 1)
                
                # Find matching prompts
                prompt_data = next(
                    (p for p in scene_prompts if p.get("scene_num") == scene_num),
                    {"image_prompt": "", "motion_prompt": ""}
                )
                
                # Determine duration: use from scene if available (Gemini), otherwise calculate from start/end
                if "duration" in scene:
                    # Gemini scenes have duration directly
                    scene_duration = scene.get("duration", 3.0)
                    start_time = cumulative_time
                    end_time = cumulative_time + scene_duration
                    cumulative_time = end_time  # Update for next scene
                else:
                    # OpenAI scenes have start/end
                    start_time = scene.get("start", 0)
                    end_time = scene.get("end", video_duration)
                    scene_duration = end_time - start_time
                
                scene_data.append({
                    "scene_num": scene_num,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": scene_duration,
                    "image_prompt": prompt_data.get("image_prompt", ""),
                    "motion_prompt": prompt_data.get("motion_prompt", ""),
                    "visible_elements": prompt_data.get("visible_elements", []),
                    "story_beat": prompt_data.get("story_beat", ""),
                    "transition_logic": prompt_data.get("transition_logic", "")
                })
                
                logger.info(f"   Scene {scene_num}: duration = {scene_duration:.2f}s (start: {start_time:.2f}s, end: {end_time:.2f}s)")
            
            # =================================================================
            # SCALE SCENE DURATIONS TO MATCH EXPECTED VO DURATION
            # =================================================================
            # This ensures the video is created at the correct length from the start,
            # so we don't need to apply slow motion or cut the video later
            if gemini_analysis:
                new_vo = gemini_analysis.get("new_voiceover", {})
                vo_script = new_vo.get("full_script", "")
                vo_word_count = new_vo.get("word_count", 0)
                
                if vo_script and not vo_word_count:
                    vo_word_count = len(vo_script.split())
                
                if vo_word_count > 0:
                    # Estimate VO duration: ~2.5 words per second for natural speech
                    # Add a small buffer (0.5s) to ensure video is slightly longer than VO
                    estimated_vo_duration = (vo_word_count / 2.5) + 0.5
                    
                    # Calculate current total scene duration
                    current_total_duration = sum(s["duration"] for s in scene_data)
                    
                    if current_total_duration > 0 and abs(estimated_vo_duration - current_total_duration) > 1.0:
                        # Scale scene durations proportionally to match VO
                        scale_factor = estimated_vo_duration / current_total_duration
                        
                        # Only scale if within reasonable bounds (0.7x to 1.5x)
                        if 0.7 <= scale_factor <= 1.5:
                            logger.info(f"📐 Scaling scene durations to match expected VO duration...")
                            logger.info(f"   VO word count: {vo_word_count} words")
                            logger.info(f"   Estimated VO duration: {estimated_vo_duration:.2f}s")
                            logger.info(f"   Current total scene duration: {current_total_duration:.2f}s")
                            logger.info(f"   Scale factor: {scale_factor:.2f}x")
                            
                            cumulative_time = 0.0
                            for scene in scene_data:
                                old_duration = scene["duration"]
                                new_duration = old_duration * scale_factor
                                # Ensure minimum duration of 2s (Kling/Runway minimum)
                                new_duration = max(2.0, new_duration)
                                scene["duration"] = new_duration
                                scene["start_time"] = cumulative_time
                                scene["end_time"] = cumulative_time + new_duration
                                cumulative_time += new_duration
                                logger.info(f"   Scene {scene['scene_num']}: {old_duration:.2f}s → {new_duration:.2f}s")
                            
                            new_total_duration = sum(s["duration"] for s in scene_data)
                            logger.info(f"   ✅ New total scene duration: {new_total_duration:.2f}s (target VO: {estimated_vo_duration:.2f}s)")
                        else:
                            logger.info(f"ℹ️ Scale factor {scale_factor:.2f}x out of range (0.7-1.5), keeping original durations")
            
            scene_results = {}
            audio_result = {"new_voice_url": None, "new_music_url": None, "final_audio_url": None, "has_speech": False}
            
            def process_scene_with_prompts(scene_info):
                """Process a single scene with pre-generated prompts.
                
                If a product was detected, enhances the prompt to maintain product accuracy
                AND uses the appropriate usage context for the scene.
                """
                scene_num = scene_info["scene_num"]
                image_prompt = scene_info["image_prompt"]
                motion_prompt = scene_info["motion_prompt"]
                duration = scene_info["duration"]
                scene_start_time = scene_info.get("start_time", 0)
                
                result_data = {
                    "scene_num": scene_num,
                    "duration": duration,
                    "image_url": None,
                    "video_url": None
                }
                scene_context = None
                
                try:
                    # Step 1: Generate image with Nano Banana
                    if image_prompt:
                        # Check if Gemini provided ready-made prompts for this scene
                        gemini_scene_prompts = None
                        if gemini_analysis:
                            gemini_scenes = gemini_analysis.get("scenes", [])
                            for gs in gemini_scenes:
                                if gs.get("scene_number") == scene_num:
                                    gemini_scene_prompts = gs.get("prompts", {})
                                    break
                        
                        # USE GEMINI PROMPTS DIRECTLY if available
                        if gemini_scene_prompts and gemini_scene_prompts.get("image_prompt"):
                            final_image_prompt = gemini_scene_prompts.get("image_prompt")
                            final_motion_prompt = gemini_scene_prompts.get("motion_prompt", motion_prompt)
                            # Override visible_elements from Gemini if available
                            gemini_visible = gemini_scene_prompts.get("visible_elements", [])
                            if gemini_visible:
                                scene_info["visible_elements"] = gemini_visible
                            
                            # Check if product is visible in THIS specific scene
                            gemini_scene_info = None
                            for gs in gemini_scenes:
                                if gs.get("scene_number") == scene_num:
                                    gemini_scene_info = gs.get("understanding", {})
                                    break
                            
                            product_visible_in_scene = gemini_scene_info.get("product_visible", False) if gemini_scene_info else False
                            
                            # Check if Manual Instructions say to remove text
                            manual_instructions_lower = manual_instructions.lower() if manual_instructions else ""
                            should_remove_text = any(phrase in manual_instructions_lower for phrase in [
                                "remove text", "remove any text", "no text", "without text", 
                                "remove all text", "delete text", "הסר טקסט", "ללא טקסט"
                            ])
                            
                            if should_remove_text:
                                logger.info(f"📝 [Scene {scene_num}] Skipping text overlay - Manual Instructions say to remove text")
                            else:
                                # Check if original video has NO VO - if so, add text to image prompt
                                # Default to False - if Gemini didn't detect VO, assume there's no VO
                                original_has_vo = gemini_analysis.get("audio", {}).get("original_has_vo", False)
                                
                                # Also check if original video had text overlays
                                original_has_text = gemini_scene_info.get("has_branding_overlay", False) if gemini_scene_info else False
                                
                                # Only add text if: original had text AND original has no VO (text shown instead of spoken)
                                if original_has_text and not original_has_vo:
                                    # Get text for this scene from Gemini analysis
                                    scene_text = gemini_scene_info.get("text_on_screen", "") if gemini_scene_info else ""
                                    
                                    # Clean scene_text - remove "none", "None", "NONE", "no text", etc.
                                    if scene_text:
                                        scene_text_lower = scene_text.lower().strip()
                                        # Remove common negative text indicators
                                        if scene_text_lower in ["none", "no text", "no", "n/a", "na", ""]:
                                            scene_text = ""
                                        # Remove "none" if it appears as a word
                                        scene_text = re.sub(r'\b(none|no text|no)\b', '', scene_text, flags=re.IGNORECASE).strip()
                                    
                                    # Add text to image prompt if we have valid text (not empty, not "none")
                                    if scene_text and scene_text.lower().strip() not in ["none", "no text", "no", "n/a", "na", ""]:
                                        # Add text overlay instruction to prompt
                                        final_image_prompt += f" | Text overlay on image: '{scene_text}' - The text should be prominently displayed, styled, and clearly readable as part of the image composition."
                                        logger.info(f"📝 [Scene {scene_num}] Added text to image prompt (original had text, no VO): '{scene_text[:50]}...'")
                                    elif scene_text:
                                        logger.warning(f"⚠️ [Scene {scene_num}] Skipping invalid text (contains 'none' or empty): '{scene_text}'")
                            
                            # Only include product reference if product is visible in this scene
                            if product_visible_in_scene and gemini_analysis.get("product", {}).get("detected"):
                                ref_url = product_reference_url
                                
                                # Build comprehensive product description for better accuracy
                                product_info_gemini = gemini_analysis.get("product", {})
                                visual_desc = product_info_gemini.get("visual_description", "")
                                application_rules = product_info_gemini.get("application_rules", "")
                                usage_method = product_info_gemini.get("usage_method", "")
                                product_image_details = product_info_gemini.get("product_image_details", "")
                                
                                # Combine all product details for comprehensive reference
                                # Include EXTREMELY DETAILED description for pixel-perfect accuracy
                                ref_desc_parts = []
                                if visual_desc:
                                    ref_desc_parts.append(f"VISUAL DESCRIPTION (CRITICAL - MATCH EXACTLY): {visual_desc}")
                                if product_image_details:
                                    ref_desc_parts.append(f"PRODUCT IMAGE DETAILS (FROM REFERENCE FRAME): {product_image_details}")
                                if application_rules:
                                    ref_desc_parts.append(f"APPLICATION RULES: {application_rules}")
                                if usage_method:
                                    ref_desc_parts.append(f"USAGE METHOD: {usage_method}")
                                
                                ref_desc = "\n\n".join(ref_desc_parts) if ref_desc_parts else visual_desc
                                
                                # Add emphasis on accuracy
                                if ref_desc:
                                    ref_desc = f"""PRODUCT REFERENCE - EXTREMELY DETAILED DESCRIPTION FOR PIXEL-PERFECT ACCURACY:

{ref_desc}

CRITICAL INSTRUCTIONS:
- The reference image shows the EXACT product appearance
- Match the product's EXACT shape, colors (with specific hex codes), size, materials, textures, and text/logos from the reference image
- Use the reference image to ensure pixel-perfect product accuracy
- The product in the generated image MUST match the reference image exactly in appearance
- Pay special attention to: exact colors (use hex codes if specified), exact shape and dimensions, exact materials and textures, exact text/logos if visible, exact lighting and shadows
- The product placement and usage should be logical based on the reference image and the scene context"""
                                
                                # Embed concise product visual summary directly into image prompt text
                                product_visual_summary = visual_desc[:300] if visual_desc else ""
                                if product_visual_summary and product_visual_summary.strip():
                                    final_image_prompt += f" | PRODUCT IN SCENE: {product_visual_summary.strip()}"
                                
                                logger.info(f"🎯 [Scene {scene_num}] Using Gemini-generated prompts (product VISIBLE in this scene)")
                                logger.info(f"   📸 Product reference: {ref_url[:60] if ref_url else 'None'}...")
                                logger.info(f"   📝 Product description: {ref_desc[:100] if ref_desc else 'None'}...")
                            else:
                                ref_url = None
                                ref_desc = None
                                logger.info(f"🎯 [Scene {scene_num}] Using Gemini-generated prompts (product NOT visible in this scene)")
                            
                            logger.info(f"   Image prompt: {final_image_prompt[:80]}...")
                            if final_motion_prompt:
                                logger.info(f"   Motion prompt: {final_motion_prompt[:80]}...")
                        else:
                            # FALLBACK: Use existing GPT enhancement logic
                            final_image_prompt = image_prompt
                            ref_url = None
                            ref_desc = None
                            
                            # Check if Manual Instructions say to remove text
                            manual_instructions_lower = manual_instructions.lower() if manual_instructions else ""
                            should_remove_text = any(phrase in manual_instructions_lower for phrase in [
                                "remove text", "remove any text", "no text", "without text", 
                                "remove all text", "delete text", "הסר טקסט", "ללא טקסט"
                            ])
                            
                            # Check if original video has NO VO - if so, add text to image prompt
                            if gemini_analysis and not should_remove_text:
                                # Default to False - if Gemini didn't detect VO, assume there's no VO
                                original_has_vo = gemini_analysis.get("audio", {}).get("original_has_vo", False)
                                
                                # Get scene info for branding check
                                gemini_scene_info = None
                                for gs in gemini_analysis.get("scenes", []):
                                    if gs.get("scene_number") == scene_num:
                                        gemini_scene_info = gs.get("understanding", {})
                                        break
                                
                                # Check if original had text overlays
                                original_has_text = gemini_scene_info.get("has_branding_overlay", False) if gemini_scene_info else False
                                
                                # Only add text if: original had text AND original has no VO
                                if original_has_text and not original_has_vo:
                                    scene_text = gemini_scene_info.get("text_on_screen", "") if gemini_scene_info else ""
                                    
                                    # Clean scene_text - remove "none", "None", "NONE", "no text", etc.
                                    if scene_text:
                                        scene_text_lower = scene_text.lower().strip()
                                        # Remove common negative text indicators
                                        if scene_text_lower in ["none", "no text", "no", "n/a", "na", ""]:
                                            scene_text = ""
                                        # Remove "none" if it appears as a word
                                        scene_text = re.sub(r'\b(none|no text|no)\b', '', scene_text, flags=re.IGNORECASE).strip()
                                    
                                    # Add text to image prompt if we have valid text (not empty, not "none")
                                    if scene_text and scene_text.lower().strip() not in ["none", "no text", "no", "n/a", "na", ""]:
                                        final_image_prompt += f" | Text overlay on image: '{scene_text}' - The text should be prominently displayed, styled, and clearly readable as part of the image composition."
                                        logger.info(f"📝 [Scene {scene_num}] Added text to image prompt (original had text, no VO, fallback): '{scene_text[:50]}...'")
                                    elif scene_text:
                                        logger.warning(f"⚠️ [Scene {scene_num}] Skipping invalid text (contains 'none' or empty): '{scene_text}'")
                            
                            if product_info.get("has_product"):
                                scene_context = None
                                # Check per-scene product_visible from multiple sources
                                product_visible_fallback = True  # default to true, but check sources
                                
                                # Source 1: Gemini understanding (most reliable)
                                if gemini_analysis:
                                    for gs in gemini_analysis.get("scenes", []):
                                        if gs.get("scene_number") == scene_num:
                                            product_visible_fallback = gs.get("understanding", {}).get("product_visible", True)
                                            break
                                
                                # Source 2: Story beat from GPT (hook/problem/agitation typically don't show product)
                                scene_story_beat = scene_info.get("story_beat", "")
                                if scene_story_beat in ("hook", "problem", "agitation", "transition"):
                                    product_visible_fallback = False
                                    logger.info(f"   [Scene {scene_num}] Product hidden - story beat '{scene_story_beat}' typically doesn't show product")
                                
                                if not product_visible_fallback:
                                    logger.info(f"🎯 [Scene {scene_num}] GPT fallback: product NOT visible in this scene (skipping product enhancement)")
                                    ref_url = None
                                    ref_desc = None
                                
                                if product_visible_fallback:
                                    scene_context = None
                                    scene_plan_entry = None
                                    narrative_role = None
                                    article_content_for_scene = None
                                
                                    # First, check video structure scene plan
                                    scene_plan = video_structure.get("scene_plan", [])
                                    for sp in scene_plan:
                                        if sp.get("scene_number") == scene_num:
                                            scene_plan_entry = sp
                                            scene_context = sp.get("product_appearance")
                                            narrative_role = sp.get("narrative_role")
                                            article_content_for_scene = sp.get("article_content_to_use")
                                            if scene_context:
                                                logger.info(f"   [Scene {scene_num}] Context from structure analysis: {scene_context} (role: {narrative_role})")
                                            break
                                
                                    # If no context from structure, try frame analysis
                                    if not scene_context:
                                        frame_analysis = product_info.get("frame_analysis", [])
                                        usage_contexts = product_info.get("usage_contexts", [])
                                        
                                        # Try to find context from frame analysis based on scene timing
                                        if frame_analysis:
                                            for frame_info in frame_analysis:
                                                frame_idx = frame_info.get("frame_index", 0)
                                                estimated_time = frame_idx / 5.0
                                                if scene_start_time <= estimated_time < scene_start_time + duration:
                                                    scene_context = frame_info.get("usage_context")
                                                    if scene_context:
                                                        logger.info(f"   [Scene {scene_num}] Context from frame analysis: {scene_context}")
                                                        break
                                        
                                        # If no context found from frames, use the first usage context
                                        if not scene_context and usage_contexts:
                                            scene_context = usage_contexts[0].get("context_type", "static_display")
                                
                                    # Log scene planning info
                                    if scene_plan_entry:
                                        logger.info(f"   [Scene {scene_num}] Narrative role: {narrative_role}")
                                        if article_content_for_scene:
                                            logger.info(f"   [Scene {scene_num}] Content to use: {article_content_for_scene[:50]}...")
                                
                                    logger.info(f"🎨 [Scene {scene_num}] Enhancing prompt with product details (context: {scene_context})...")
                                    try:
                                        enhanced_article_text = article_text
                                        if article_content_for_scene:
                                            enhanced_article_text = f"{article_content_for_scene}\n\n{article_text}"
                                        
                                        enhanced_product_info = product_info.copy() if product_info else {}
                                        if scene_plan_entry:
                                            enhanced_product_info["scene_plan"] = {
                                                "narrative_role": narrative_role,
                                                "key_message": scene_plan_entry.get("key_message", ""),
                                                "visual_suggestion": scene_plan_entry.get("visual_suggestion", "")
                                            }
                                        
                                        if gemini_analysis:
                                            video_story = gemini_analysis.get("video_story", {})
                                            scene_breakdown = gemini_analysis.get("scene_breakdown", [])
                                            
                                            scene_gemini_info = None
                                            for s in scene_breakdown:
                                                if s.get("scene_number") == scene_num:
                                                    scene_gemini_info = s
                                                    break
                                            
                                            blueprint = video_story.get("recreation_blueprint", {})
                                            subject_appearances = blueprint.get("subject_appearance_per_scene", {})
                                            scene_subject_appearance = subject_appearances.get(str(scene_num), "")
                                            
                                            story_arc = video_story.get("story_arc", {})
                                            subject_journey = video_story.get("subject_journey", {})
                                            
                                            enhanced_product_info["story_context"] = {
                                                "story_type": video_story.get("story_type", ""),
                                                "story_summary": video_story.get("one_sentence_summary", ""),
                                                "scene_subject_appearance": scene_subject_appearance,
                                                "has_visible_change": subject_journey.get("has_visible_change", False),
                                                "change_type": subject_journey.get("change_type", ""),
                                                "start_state": subject_journey.get("start_state", ""),
                                                "end_state": subject_journey.get("end_state", ""),
                                                "essential_story_beats": blueprint.get("essential_story_beats", []),
                                                "must_preserve": blueprint.get("must_preserve", [])
                                            }
                                            
                                            if scene_gemini_info:
                                                visual_content = scene_gemini_info.get("visual_content", {})
                                                subjects = visual_content.get("subjects", {})
                                                enhanced_product_info["story_context"]["scene_details"] = {
                                                    "physical_state": subjects.get("physical_state", ""),
                                                    "action": subjects.get("action", ""),
                                                    "what_changed": scene_gemini_info.get("scene_changes", {}).get("what_changed_from_previous", ""),
                                                    "purpose": scene_gemini_info.get("story_element", {}).get("purpose", ""),
                                                    "emotional_beat": scene_gemini_info.get("story_element", {}).get("emotional_beat", "")
                                                }
                                                
                                                if scene_subject_appearance:
                                                    logger.info(f"   [Scene {scene_num}] Subject appearance: {scene_subject_appearance[:60]}...")
                                                if subjects.get("physical_state"):
                                                    logger.info(f"   [Scene {scene_num}] Physical state: {subjects.get('physical_state', '')[:60]}...")
                                        
                                        # Embed concise product visual summary directly into the image prompt
                                        product_visual_summary = product_info.get("product_description", "")[:300]
                                        if product_visual_summary and product_visual_summary.strip():
                                            final_image_prompt += f" | PRODUCT IN SCENE: {product_visual_summary.strip()}"
                                        
                                        final_image_prompt = self.openai_service.enhance_prompt_with_product(
                                            original_prompt=final_image_prompt,
                                            product_description=product_info.get("product_description", ""),
                                            article_text=enhanced_article_text,
                                            product_info=enhanced_product_info,
                                            scene_context=scene_context,
                                            video_style=video_style
                                        )
                                        ref_url = product_reference_url
                                        ref_desc = product_info.get("product_description")
                                        logger.info(f"✅ [Scene {scene_num}] Prompt enhanced with product + style matching")
                                    except Exception as e:
                                        logger.warning(f"⚠️ [Scene {scene_num}] Failed to enhance prompt: {e}")
                        
                        # When using product reference image, ALWAYS include product text description to reduce hallucinations
                        if ref_url and (not ref_desc or not ref_desc.strip()) and product_info.get("has_product"):
                            ref_desc = (product_info.get("product_description") or "").strip()[:800]
                            if ref_desc:
                                logger.info(f"   [Scene {scene_num}] Using product description in prompt (text + image)")
                        
                        logger.info(f"🎨 [Scene {scene_num}] Generating image...")
                        image_url = self.kie_service.generate_image_nano_banana(
                            prompt=final_image_prompt,
                            reference_image_url=ref_url,
                            reference_description=ref_desc,
                            target_language=subtitle_language or "en",
                            article_text=article_text
                        )
                        if image_url:
                            result_data["image_url"] = image_url
                            
                            # Write to sheet
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.SCENE_NEW_IMAGE_PREFIX.replace("{n}", str(scene_num)),
                                value=image_url,
                                headers=headers
                            )
                            
                            # Step 2: Generate video with animation model (Runway or Kling)
                            if motion_prompt:
                                # Use ONLY the original Second prompt for the video API. Do NOT send the
                                # enhanced motion prompt (it adds long instructions about hands/body/product
                                # and causes the model to hallucinate elements like feet that are not in the image).
                                if product_info.get("has_product"):
                                    try:
                                        _ = self.openai_service.enhance_motion_prompt_with_product(
                                            original_motion_prompt=motion_prompt,
                                            product_info=product_info,
                                            scene_context=scene_context,
                                            video_style=video_style
                                        )
                                        logger.info(f"✅ [Scene {scene_num}] Motion prompt product context noted (API receives original Second prompt only)")
                                    except Exception as me:
                                        logger.warning(f"⚠️ [Scene {scene_num}] Motion enhancement failed: {me}")
                                
                                final_motion_prompt = motion_prompt
                                # Validate motion against visible_elements before sanitizing
                                scene_visible_elements = scene_info.get("visible_elements", [])
                                if scene_visible_elements:
                                    final_motion_prompt = self._validate_motion_against_visible_elements(
                                        final_motion_prompt, scene_visible_elements
                                    )
                                final_motion_prompt = self._sanitize_motion_prompt(final_motion_prompt)
                                logger.info(f"🎬 [Scene {scene_num}] Generating video with {animation_model.upper()}...")
                                if animation_model == "kling-3.0":
                                    video_url = self.kie_service.generate_video_kling_30(
                                        prompt=final_motion_prompt,
                                        image_url=image_url,
                                        duration=duration
                                    )
                                elif animation_model == "kling":
                                    video_url = self.kie_service.generate_video_kling(
                                        prompt=final_motion_prompt,
                                        image_url=image_url,
                                        duration=duration
                                    )
                                else:
                                    video_url = self.kie_service.generate_video_runway(
                                        prompt=final_motion_prompt,
                                        image_url=image_url,
                                        duration=duration
                                    )
                                if video_url:
                                    # Upload to S3 immediately to avoid temp URL expiration
                                    try:
                                        s3_video_url = self._upload_video_to_s3_from_url(
                                            video_url=video_url,
                                            row_num=row_num,
                                            scene_num=scene_num
                                        )
                                        if s3_video_url:
                                            video_url = s3_video_url
                                            logger.info(f"✅ [Scene {scene_num}] Video uploaded to S3 (permanent URL)")
                                    except Exception as upload_err:
                                        logger.warning(f"⚠️ [Scene {scene_num}] S3 upload failed, using temp URL: {upload_err}")
                                    
                                    result_data["video_url"] = video_url
                                    
                                    self._update_sheet_cell(
                                        row_num=row_num,
                                        column=config.SCENE_NEW_VIDEO_PREFIX.replace("{n}", str(scene_num)),
                                        value=video_url,
                                        headers=headers
                                    )
                    
                    return result_data
                    
                except Exception as e:
                    logger.error(f"❌ [Scene {scene_num}] Error: {e}")
                    return result_data
            
            def process_audio_pipeline():
                """Process audio: Extract → Detect speech → ElevenLabs + Suno.
                
                If article_text is provided, generates NEW VO from article using TTS.
                """
                nonlocal audio_result, voice_id
                
                # Ensure voice_id is valid at the start - use default if not
                if not is_valid_voice_id(voice_id):
                    voice_id = config.DEFAULT_VOICE_ID
                    logger.info(f"🎤 [AUDIO] Using default voice ID: {voice_id}")
                
                logger.info("🎤 [AUDIO] Starting audio pipeline in parallel...")
                audio_path = os.path.join(temp_dir, "original_audio.mp3")
                
                # Try local extraction first, then cloud
                audio_extracted = False
                if ffmpeg_available and video_path:
                    audio_extracted = FFmpegProcessor.extract_audio(video_path, audio_path)
                
                if not audio_extracted:
                    logger.info("🌐 [AUDIO] Extracting audio via Rendi.dev cloud...")
                    original_audio_url = FFmpegProcessor.extract_audio_from_url(
                        video_url=video_url,
                        output_path=audio_path,
                        rendi_api_key=config.RENDI_API_KEY
                    )
                    
                    if original_audio_url:
                        try:
                            response = requests.get(original_audio_url, timeout=60)
                            response.raise_for_status()
                            with open(audio_path, 'wb') as f:
                                f.write(response.content)
                            audio_extracted = True
                            logger.info("✅ [AUDIO] Audio downloaded from cloud extraction")
                        except Exception as e:
                            logger.error(f"❌ [AUDIO] Failed to download audio: {e}")
                
                if not audio_extracted or not os.path.exists(audio_path):
                    logger.warning("⚠️ [AUDIO] Could not extract audio")
                    return
                
                # Upload original audio to S3 for Suno (needed in all paths)
                timestamp = int(time.time())
                temp_audio_key = f"temp_audio_for_suno_row_{row_num}_{timestamp}.mp3"
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
                audio_url_for_suno = self.s3_service.upload_audio_bytes(
                    audio_data=audio_data,
                    key_name=temp_audio_key
                )
                
                # =================================================================
                # DETECT VO PRESENCE AND GENDER FROM ORIGINAL VIDEO
                # =================================================================
                # Check if original video has voice-over narration
                detected_gender, original_transcript = self.elevenlabs_service.detect_vo_gender(audio_path)
                original_has_vo = detected_gender is not None
                
                if original_has_vo:
                    logger.info(f"🎤 [AUDIO] Original video HAS VO (gender: {detected_gender})")
                    # Write detected gender (m/f) to Gender column
                    logger.info(f"📝 [Row {row_num}] Writing gender '{detected_gender}' to column '{config.GENDER_COLUMN}'")
                    update_success = self._update_sheet_cell(
                        row_num=row_num,
                        column=config.GENDER_COLUMN,
                        value=detected_gender,
                        headers=headers
                    )
                    if update_success:
                        logger.info(f"✅ [Row {row_num}] Gender '{detected_gender}' written successfully")
                        
                        # ================================================================
                        # RE-READ VOICE ID FROM SHEET (formula depends on Gender)
                        # ================================================================
                        time.sleep(1.5)  # Wait for sheet formula to recalculate
                        try:
                            updated_row_data = self.sheets_service.get_row(
                                sheet_id=config.GOOGLE_SHEET_ID,
                                worksheet_name=config.GOOGLE_SHEET_TAB,
                                row_num=row_num
                            )
                            if updated_row_data:
                                voice_id_col = self.sheets_service.get_column_index(headers, config.VOICE_ID_COLUMN)
                                if voice_id_col is not None and voice_id_col < len(updated_row_data):
                                    new_voice_id = updated_row_data[voice_id_col].strip()
                                    if is_valid_voice_id(new_voice_id):
                                        voice_id = new_voice_id
                                        logger.info(f"🎤 [Row {row_num}] Updated Voice ID from sheet: {voice_id}")
                                    else:
                                        # If sheet voice_id is invalid, check if original is valid
                                        if not is_valid_voice_id(voice_id):
                                            voice_id = config.DEFAULT_VOICE_ID
                                            logger.info(f"🎤 [Row {row_num}] Voice ID from sheet is invalid ('{new_voice_id}'), using default: {voice_id}")
                                        else:
                                            logger.info(f"🎤 [Row {row_num}] Voice ID from sheet is invalid ('{new_voice_id}'), keeping original: {voice_id}")
                        except Exception as e:
                            logger.warning(f"⚠️ [Row {row_num}] Failed to re-read Voice ID: {e}")
                    else:
                        logger.warning(f"⚠️ [Row {row_num}] Failed to write gender to sheet")
                else:
                    logger.info("🔇 [AUDIO] Original video has NO VO - will generate music only")
                
                # =================================================================
                # MANUAL VO TEXT PATH - Use provided text instead of generating
                # =================================================================
                if manual_vo_text:
                    logger.info(f"🎤 [AUDIO] Using MANUAL VO text ({len(manual_vo_text)} chars)...")
                    
                    # Use manual text directly for TTS WITH TIMESTAMPS
                    vo_language = subtitle_language or article_language or "en"
                    
                    tts_result = self.elevenlabs_service.text_to_speech_with_timestamps(
                        text=manual_vo_text,
                        voice_id=voice_id,  # Validated inside function
                        language=vo_language
                    )
                    
                    if tts_result:
                        tts_audio_data, word_segments = tts_result
                        logger.info(f"📝 [AUDIO] Got {len(word_segments)} word segments from Manual TTS")
                        
                        # Store word segments for ZapCap
                        audio_result["tts_word_segments"] = word_segments
                        audio_result["tts_generated"] = True
                        
                        ts = int(time.time())
                        voice_key = f"manual_vo_row_{row_num}_{ts}.mp3"
                        voice_url = self.s3_service.upload_audio_bytes(
                            audio_data=tts_audio_data,
                            key_name=voice_key
                        )
                        
                        if voice_url:
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.NEW_VOICE_COLUMN,
                                value=voice_url,
                                headers=headers
                            )
                            audio_result["new_voice_url"] = voice_url
                            audio_result["has_speech"] = True
                            logger.info(f"✅ [AUDIO] Manual VO TTS generated: {voice_url}")
                    else:
                        logger.error("❌ [AUDIO] Failed to generate TTS from manual VO text")
                    
                    # Handle music: use manual link or generate
                    if manual_music_link:
                        logger.info(f"🎵 [AUDIO] Using MANUAL music link: {manual_music_link[:50]}...")
                        audio_result["new_music_url"] = manual_music_link
                        self._update_sheet_cell(
                            row_num=row_num,
                            column=config.NEW_MUSIC_COLUMN,
                            value=manual_music_link,
                            headers=headers
                        )
                    elif audio_url_for_suno:
                        # Generate music with Suno
                        music_description = self.openai_service.generate_music_description(
                            scene_prompts=scene_prompts
                        )
                        logger.info(f"🎵 [AUDIO] Generating music: {music_description[:80]}...")
                        
                        music_url = self.suno_service.generate_instrumental_background(
                            audio_url=audio_url_for_suno,
                            style=music_description,
                            fallback_style=music_description
                        )
                        
                        if music_url:
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.NEW_MUSIC_COLUMN,
                                value=music_url,
                                headers=headers
                            )
                            audio_result["new_music_url"] = music_url
                    
                    # Set final audio
                    if audio_result.get("new_voice_url"):
                        audio_result["final_audio_url"] = audio_result["new_voice_url"]
                        logger.info("✅ [AUDIO] Manual VO processing complete")
                    
                    return  # Exit - manual VO complete
                
                # =================================================================
                # ARTICLE ADAPTATION PATH - Generate NEW VO from article via TTS
                # (Only if original video has VO)
                # =================================================================
                if has_article_adaptation:
                    logger.info("📰 [AUDIO] Article adaptation mode...")
                    
                    # Check if original video has VO - if not, skip VO generation
                    if not original_has_vo:
                        logger.info("🔇 [AUDIO] Original video has NO VO - skipping VO generation, creating music only")
                        
                        # Generate music only (no VO)
                        if manual_music_link:
                            logger.info(f"🎵 [AUDIO] Using MANUAL music link: {manual_music_link[:50]}...")
                            audio_result["new_music_url"] = manual_music_link
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.NEW_MUSIC_COLUMN,
                                value=manual_music_link,
                                headers=headers
                            )
                        elif audio_url_for_suno:
                            music_description = self.openai_service.generate_music_description(
                                scene_prompts=scene_prompts
                            )
                            logger.info(f"🎵 [AUDIO] Generating music (no VO): {music_description[:80]}...")
                            
                            music_url = self.suno_service.generate_instrumental_background(
                                audio_url=audio_url_for_suno,
                                style=music_description,
                                fallback_style=music_description
                            )
                            
                            if music_url:
                                self._update_sheet_cell(
                                    row_num=row_num,
                                    column=config.NEW_MUSIC_COLUMN,
                                    value=music_url,
                                    headers=headers
                                )
                                audio_result["new_music_url"] = music_url
                                audio_result["final_audio_url"] = music_url
                                logger.info(f"✅ [AUDIO] Music only (no VO) generated: {music_url}")
                        
                        logger.info("✅ [AUDIO] No-VO mode complete (music only)")
                        return  # Exit - no VO mode complete
                    
                    # Original has VO - proceed with VO generation
                    logger.info("🎤 [AUDIO] Original video HAS VO - generating new VO from article...")
                    
                    # Check if Gemini provided a ready VO script
                    vo_script = None
                    if gemini_analysis and gemini_analysis.get("new_voiceover", {}).get("full_script"):
                        vo_script = gemini_analysis["new_voiceover"]["full_script"]
                        vo_style = gemini_analysis["new_voiceover"].get("style", "")
                        vo_word_count = gemini_analysis["new_voiceover"].get("word_count", len(vo_script.split()))
                        logger.info(f"🎯 [AUDIO] Using Gemini-generated VO script ({vo_word_count} words, style: {vo_style})")
                        logger.info(f"   Script preview: {vo_script[:100]}...")
                    else:
                        # FALLBACK: Use GPT to generate VO script
                        logger.info("📝 [AUDIO] Generating VO script with GPT (Gemini didn't provide one)...")
                        vo_script = self.openai_service.generate_vo_script_from_article(
                            article_text=article_text,
                            vertical=vertical,
                            target_duration=video_duration,
                            target_language=article_language,
                            original_vo_transcript=original_transcript,
                            scene_prompts=scene_prompts,  # Pass scene prompts so VO matches visuals
                            gemini_vo_recommendations=gemini_analysis  # Pass Gemini analysis for VO style matching
                        )
                    
                    if vo_script:
                        logger.info(f"✅ [AUDIO] Generated VO script ({len(vo_script.split())} words)")
                        
                        # Generate TTS audio WITH TIMESTAMPS for ZapCap subtitles
                        tts_result = self.elevenlabs_service.text_to_speech_with_timestamps(
                            text=vo_script,
                            voice_id=voice_id,  # Validated inside function
                            language=article_language
                        )
                        
                        if tts_result:
                            tts_audio_data, word_segments = tts_result
                            logger.info(f"📝 [AUDIO] Got {len(word_segments)} word segments from TTS")
                            
                            # Store word segments for ZapCap (will be used in Step 11)
                            audio_result["tts_word_segments"] = word_segments
                            audio_result["tts_generated"] = True
                            
                            ts = int(time.time())
                            voice_key = f"tts_voice_row_{row_num}_{ts}.mp3"
                            voice_url = self.s3_service.upload_audio_bytes(
                                audio_data=tts_audio_data,
                                key_name=voice_key
                            )
                            
                            if voice_url:
                                self._update_sheet_cell(
                                    row_num=row_num,
                                    column=config.NEW_VOICE_COLUMN,
                                    value=voice_url,
                                    headers=headers
                                )
                                audio_result["new_voice_url"] = voice_url
                                logger.info(f"✅ [AUDIO] TTS voice generated: {voice_url}")
                        else:
                            logger.error("❌ [AUDIO] Failed to generate TTS audio")
                    else:
                        logger.error("❌ [AUDIO] Failed to generate VO script from article")
                    
                    # Handle music: use manual link or generate
                    if manual_music_link:
                        logger.info(f"🎵 [AUDIO] Using MANUAL music link: {manual_music_link[:50]}...")
                        audio_result["new_music_url"] = manual_music_link
                        self._update_sheet_cell(
                            row_num=row_num,
                            column=config.NEW_MUSIC_COLUMN,
                            value=manual_music_link,
                            headers=headers
                        )
                    elif audio_url_for_suno:
                        music_description = self.openai_service.generate_music_description(
                            scene_prompts=scene_prompts
                        )
                        logger.info(f"🎵 [AUDIO] Generating music for article: {music_description[:80]}...")
                        
                        music_url = self.suno_service.generate_instrumental_background(
                            audio_url=audio_url_for_suno,
                            style=music_description,
                            fallback_style=music_description
                        )
                        
                        if music_url:
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.NEW_MUSIC_COLUMN,
                                value=music_url,
                                headers=headers
                            )
                            audio_result["new_music_url"] = music_url
                            logger.info(f"✅ [AUDIO] New music generated: {music_url}")
                    
                    # Set final audio
                    if audio_result.get("new_voice_url"):
                        audio_result["final_audio_url"] = audio_result["new_voice_url"]
                        audio_result["has_speech"] = True
                        logger.info("✅ [AUDIO] Article adaptation complete")
                    
                    return  # Exit - article adaptation complete
                
                # =================================================================
                # NORMAL PATH - Use existing speech detection and voice changer
                # =================================================================
                # Use the VO detection result from earlier (original_has_vo)
                audio_result["has_speech"] = original_has_vo
                
                if original_has_vo:
                    # =========================================================
                    # PATH A: Speech detected - Stem Separation → Voice Changer → New Music
                    # =========================================================
                    logger.info("🎤 [AUDIO] Speech detected - separating stems first...")
                    
                    # Step 1: Separate stems to get clean vocals
                    clean_vocals_path = self.elevenlabs_service.separate_stems(
                        audio_path=audio_path,
                        output_dir=temp_dir
                    )
                    
                    # Use clean vocals if available, otherwise fallback to original audio
                    voice_source_path = clean_vocals_path if clean_vocals_path else audio_path
                    if clean_vocals_path:
                        logger.info("✅ [AUDIO] Using clean vocals for voice changer")
                    else:
                        logger.warning("⚠️ [AUDIO] Stem separation failed, using original audio")
                    
                    def run_elevenlabs():
                        """Apply voice changer on clean vocals."""
                        new_voice_data = self.elevenlabs_service.voice_changer(voice_source_path)
                        if new_voice_data:
                            ts = int(time.time())
                            voice_key = f"voice_row_{row_num}_{ts}.mp3"
                            voice_url = self.s3_service.upload_audio_bytes(
                                audio_data=new_voice_data,
                                key_name=voice_key
                            )
                            if voice_url:
                                self._update_sheet_cell(
                                    row_num=row_num,
                                    column=config.NEW_VOICE_COLUMN,
                                    value=voice_url,
                                    headers=headers
                                )
                                logger.info(f"✅ [AUDIO] Voice changed: {voice_url}")
                                return voice_url
                        return None
                    
                    def run_suno_new_music():
                        """Generate NEW background music with Suno (or use manual link)."""
                        # Check for manual music link first
                        if manual_music_link:
                            logger.info(f"🎵 [AUDIO] Using MANUAL music link: {manual_music_link[:50]}...")
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.NEW_MUSIC_COLUMN,
                                value=manual_music_link,
                                headers=headers
                            )
                            return manual_music_link
                        
                        if not audio_url_for_suno:
                            return None
                        
                        # Generate dynamic music description based on video content
                        music_description = self.openai_service.generate_music_description(
                            scene_prompts=scene_prompts
                        )
                        logger.info(f"🎵 [AUDIO] Using AI-generated music description: {music_description[:80]}...")
                        
                        music_url = self.suno_service.generate_instrumental_background(
                            audio_url=audio_url_for_suno,
                            style=music_description,
                            fallback_style=music_description  # Use same for pure generation fallback
                        )
                        if music_url:
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.NEW_MUSIC_COLUMN,
                                value=music_url,
                                headers=headers
                            )
                            logger.info(f"✅ [AUDIO] New music generated: {music_url}")
                            return music_url
                        return None
                    
                    # Run Voice Changer and Suno in parallel
                    with ThreadPoolExecutor(max_workers=10) as audio_executor:
                        voice_future = audio_executor.submit(run_elevenlabs)
                        music_future = audio_executor.submit(run_suno_new_music)
                        
                        audio_result["new_voice_url"] = voice_future.result()
                        audio_result["new_music_url"] = music_future.result()
                    
                    # Final audio will combine: New Voice + New Suno Music
                    # (original audio is discarded)
                    if audio_result["new_voice_url"]:
                        audio_result["final_audio_url"] = audio_result["new_voice_url"]
                        logger.info("✅ [AUDIO] Voice ready, new Suno music will be added after video combination")
                    
                else:
                    # =========================================================
                    # PATH B: No speech - use manual music or generate cover music
                    # =========================================================
                    logger.info("🎵 [AUDIO] No speech detected - setting up music...")
                    
                    # Check for manual music link first
                    if manual_music_link:
                        logger.info(f"🎵 [AUDIO] Using MANUAL music link: {manual_music_link[:50]}...")
                        self._update_sheet_cell(
                            row_num=row_num,
                            column=config.NEW_MUSIC_COLUMN,
                            value=manual_music_link,
                            headers=headers
                        )
                        self._update_sheet_cell(
                            row_num=row_num,
                            column=config.NEW_VOICE_COLUMN,
                            value=f"[NO VOICE - Manual Music] {manual_music_link}",
                            headers=headers
                        )
                        audio_result["new_music_url"] = manual_music_link
                        audio_result["final_audio_url"] = manual_music_link
                    elif audio_url_for_suno:
                        music_url = self.suno_service.generate_cover_music(
                            audio_url=audio_url_for_suno,
                            audio_path=audio_path
                        )
                        
                        if music_url:
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.NEW_MUSIC_COLUMN,
                                value=music_url,
                                headers=headers
                            )
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.NEW_VOICE_COLUMN,
                                value=f"[NO VOICE - Music Only] {music_url}",
                                headers=headers
                            )
                            audio_result["new_music_url"] = music_url
                            audio_result["final_audio_url"] = music_url
                            logger.info(f"✅ [AUDIO] Cover music generated: {music_url}")
                        else:
                            # Fallback: use original audio
                            logger.warning("⚠️ [AUDIO] Could not generate cover music, using original")
                            original_audio_key = f"original_audio_row_{row_num}_{timestamp}.mp3"
                            original_url = self.s3_service.upload_audio_bytes(
                                audio_data=audio_data,
                                key_name=original_audio_key
                            )
                            audio_result["final_audio_url"] = original_url
            
            # =================================================================
            # Run SCENE PROCESSING + AUDIO PROCESSING in PARALLEL
            # =================================================================
            with ThreadPoolExecutor(max_workers=12) as executor:
                # Submit audio pipeline as one task
                audio_future = executor.submit(process_audio_pipeline)
                
                # Submit all scene processing tasks
                scene_futures = {
                    executor.submit(process_scene_with_prompts, scene): scene["scene_num"]
                    for scene in scene_data
                }
                
                # Wait for scene processing to complete
                for future in as_completed(scene_futures):
                    scene_num = scene_futures[future]
                    try:
                        scene_result = future.result()
                        scene_results[scene_num] = scene_result
                        if scene_result.get("video_url"):
                            result["scenes_processed"] += 1
                            logger.info(f"✅ Scene {scene_num} completed")
                    except Exception as e:
                        logger.error(f"❌ Scene {scene_num} failed: {e}")
                
                # Wait for audio pipeline to complete (it may already be done)
                try:
                    audio_future.result()
                    logger.info("✅ Audio pipeline completed")
                except Exception as e:
                    logger.error(f"❌ Audio pipeline failed: {e}")
            
            # Collect videos for concatenation - ONLY videos, NOT images
            # Use a dict to prevent duplicates by scene_num
            scene_videos_dict = {}
            missing_videos = []
            
            for scene in scene_data:
                scene_num = scene["scene_num"]
                # Skip if we already have this scene (prevent duplicates)
                if scene_num in scene_videos_dict:
                    logger.warning(f"   ⚠️ Scene {scene_num}: Already in list, skipping duplicate")
                    continue
                    
                if scene_num in scene_results:
                    scene_result = scene_results[scene_num]
                    # Only include if it has video_url (not just image_url)
                    if scene_result.get("video_url"):
                        scene_videos_dict[scene_num] = {
                            "video_url": scene_result["video_url"],
                            "duration": scene["duration"]
                        }
                        logger.info(f"   ✅ Scene {scene_num}: video ready (duration: {scene['duration']:.2f}s)")
                    elif scene_result.get("image_url"):
                        # Skip scenes that only have images (no video/animation)
                        logger.warning(f"   ⚠️ Scene {scene_num}: Only image available, no video/animation - skipping from concatenation")
                        missing_videos.append(scene_num)
                    else:
                        missing_videos.append(scene_num)
                        logger.warning(f"   ⚠️ Scene {scene_num}: video missing or failed")
                else:
                    missing_videos.append(scene_num)
                    logger.warning(f"   ⚠️ Scene {scene_num}: video missing or failed")
            
            # Convert dict to list, sorted by scene_num to maintain order
            scene_videos_with_durations = [scene_videos_dict[num] for num in sorted(scene_videos_dict.keys())]
            
            if missing_videos:
                logger.warning(f"⚠️ {len(missing_videos)} scenes missing videos: {missing_videos}")
            
            logger.info(f"✅ Parallel processing complete: {len(scene_videos_with_durations)}/{len(scene_data)} videos generated")
            
            # Extract audio results
            new_voice_url = audio_result.get("new_voice_url")
            new_music_url = audio_result.get("new_music_url")
            final_audio_url = audio_result.get("final_audio_url")
            result["new_music_url"] = new_music_url
            
            # Step 6: Trim and concatenate all scene videos with Rendi
            logger.info(f"🎬 [Row {row_num}] Step 6: Trimming videos to original scene durations and concatenating...")
            combined_video_url = None
            if scene_videos_with_durations:
                # First, trim each Runway video to match original scene duration
                logger.info(f"✂️ Trimming {len(scene_videos_with_durations)} videos to their original scene durations...")
                for i, item in enumerate(scene_videos_with_durations):
                    logger.info(f"   Scene {i+1}: target duration = {item['duration']:.2f}s")
                
                trimmed_videos = self.rendi_service.trim_videos_batch(scene_videos_with_durations)
                
                logger.info(f"✅ Trimmed {len(trimmed_videos)} videos, now uploading to S3 if needed...")
                
                # Upload any Rendi storage URLs to S3 to prevent download failures
                for i, video_item in enumerate(trimmed_videos):
                    video_url = video_item.get("video_url") if isinstance(video_item, dict) else video_item
                    if video_url and "storage.rendi.dev" in video_url:
                        # This is a Rendi storage URL - upload to S3 to prevent download failures
                        logger.info(f"   📤 Uploading Rendi storage video {i+1} to S3...")
                        s3_key = f"rendi_videos/row_{row_num}_scene_{i+1}_{int(time.time())}.mp4"
                        uploaded_url = self.s3_service.upload_video_from_url(video_url, s3_key)
                        if uploaded_url:
                            logger.info(f"   ✅ Uploaded to S3: {uploaded_url[:60]}...")
                            if isinstance(video_item, dict):
                                video_item["video_url"] = uploaded_url
                            else:
                                trimmed_videos[i] = uploaded_url
                        else:
                            logger.warning(f"   ⚠️ Failed to upload Rendi video to S3, using original URL")
                
                logger.info(f"✅ Videos ready for concatenation...")
                
                # =============================================================
                # OPENING TEXT OVERLAY (on first scene only)
                # =============================================================
                if add_opening_text and trimmed_videos:
                    logger.info(f"🎬 [Row {row_num}] Adding opening text to first scene...")
                    try:
                        # If no opening text provided, generate one based on the article/content
                        actual_opening_text = opening_text
                        if not actual_opening_text:
                            # Generate opening text based on VIDEO content (first scene description)
                            # Get video description from scene_data
                            video_description = None
                            if scene_data:
                                # Combine first few scene prompts to understand video content
                                scene_descriptions = []
                                for scene in scene_data[:3]:  # Use first 3 scenes for context
                                    if scene.get("image_prompt"):
                                        scene_descriptions.append(scene["image_prompt"])
                                if scene_descriptions:
                                    video_description = " | ".join(scene_descriptions)
                                    logger.info(f"🎬 Video description for opening text: {video_description[:100]}...")
                            
                            # Use OpenAI to generate short, compelling opening text
                            opening_lang = subtitle_language or article_language or "en"
                            generated_text = self.openai_service.generate_opening_text(
                                article_text=article_text[:1000] if article_text else "",
                                language=opening_lang,
                                video_description=video_description
                            )
                            if generated_text:
                                actual_opening_text = generated_text
                                logger.info(f"✅ Generated opening text: '{actual_opening_text}'")
                        
                        if actual_opening_text:
                            # Step 1: Generate opening text image with Nano Banana
                            opening_image_url = self.kie_service.generate_opening_text(actual_opening_text)
                            
                            if opening_image_url:
                                # Step 2: Download, process (remove bg), and re-upload
                                opening_processed_url = self._process_cta_button(
                                    cta_image_url=opening_image_url,
                                    temp_dir=temp_dir,
                                    row_num=row_num
                                )
                                
                                if opening_processed_url:
                                    # Step 3: Overlay on the first scene video
                                    first_scene_video = trimmed_videos[0]
                                    first_video_url = first_scene_video.get("video_url") if isinstance(first_scene_video, dict) else first_scene_video
                                    
                                    video_with_opening = self.rendi_service.overlay_cta_on_video(
                                        video_url=first_video_url,
                                        cta_image_url=opening_processed_url,
                                        position="center"
                                    )
                                    
                                    if video_with_opening:
                                        # Update the first scene with the opening text version
                                        if isinstance(trimmed_videos[0], dict):
                                            trimmed_videos[0]["video_url"] = video_with_opening
                                        else:
                                            trimmed_videos[0] = video_with_opening
                                        logger.info(f"✅ Opening text added to first scene: '{actual_opening_text}'")
                                    else:
                                        logger.warning("⚠️ Failed to overlay opening text on video")
                                else:
                                    logger.warning("⚠️ Failed to process opening text image")
                            else:
                                logger.warning("⚠️ Failed to generate opening text image")
                        else:
                            logger.warning("⚠️ No opening text available (empty and couldn't generate)")
                    except Exception as e:
                        logger.error(f"❌ Opening text overlay error: {e}, continuing without opening text")
                
                # =============================================================
                # CTA BUTTON OVERLAY (on last scene only - for "at_the_end" mode)
                # For "whole_video" mode, CTA is applied after concatenation
                # =============================================================
                if cta_button and cta_text and trimmed_videos and cta_duration == "at_the_end":
                    logger.info(f"🔘 [Row {row_num}] Adding CTA button to last scene: '{cta_text}'")
                    try:
                        # Step 1: Generate CTA button image with Nano Banana
                        cta_image_url = self.kie_service.generate_cta_button(cta_text)
                        
                        if cta_image_url:
                            # Step 2: Download, process (remove bg + add glow), and re-upload
                            cta_processed_url = self._process_cta_button(
                                cta_image_url=cta_image_url,
                                temp_dir=temp_dir,
                                row_num=row_num
                            )
                            
                            if cta_processed_url:
                                # Step 3: Overlay on the last scene video
                                last_scene_idx = len(trimmed_videos) - 1
                                last_scene_video = trimmed_videos[last_scene_idx]
                                last_video_url = last_scene_video.get("video_url") if isinstance(last_scene_video, dict) else last_scene_video
                                
                                video_with_cta = self.rendi_service.overlay_cta_on_video(
                                    video_url=last_video_url,
                                    cta_image_url=cta_processed_url,
                                    position="center"
                                )
                                
                                if video_with_cta:
                                    # Update the last scene with the CTA version
                                    if isinstance(trimmed_videos[last_scene_idx], dict):
                                        trimmed_videos[last_scene_idx]["video_url"] = video_with_cta
                                    else:
                                        trimmed_videos[last_scene_idx] = video_with_cta
                                    logger.info(f"✅ CTA button added to last scene")
                                else:
                                    logger.warning("⚠️ Failed to overlay CTA on video, continuing without CTA")
                            else:
                                logger.warning("⚠️ Failed to process CTA button image, continuing without CTA")
                        else:
                            logger.warning("⚠️ Failed to generate CTA button, continuing without CTA")
                    except Exception as e:
                        logger.error(f"❌ CTA overlay error: {e}, continuing without CTA")
                
                # =================================================================
                # STEP 6b: Adjust video duration to match VO or original video length
                # =================================================================
                # If we have VO: ensure video ends when VO ends
                # If no VO: ensure video is approximately original video length
                if trimmed_videos:
                    try:
                        # Calculate current total video duration
                        total_video_duration = sum(
                            v.get("duration", 0) if isinstance(v, dict) else 0 
                            for v in trimmed_videos
                        )
                        
                        target_duration = None
                        
                        if new_voice_url:
                            # CASE 1: We have VO - video should end when VO ends
                            logger.info("🔄 Adjusting video duration to match VO...")
                            vo_duration = self.rendi_service.get_audio_duration_cloud(new_voice_url)
                            
                            if vo_duration > 0:
                                logger.info(f"   VO duration: {vo_duration:.2f}s")
                                logger.info(f"   Current video duration: {total_video_duration:.2f}s")
                                
                                if abs(vo_duration - total_video_duration) > 0.5:  # More than 0.5s difference
                                    target_duration = vo_duration
                                    logger.info(f"   Target duration: {target_duration:.2f}s (matching VO)")
                        else:
                            # CASE 2: No VO - video should be approximately original video length
                            logger.info("🔄 Adjusting video duration to match original video length...")
                            logger.info(f"   Original video duration: {video_duration:.2f}s")
                            logger.info(f"   Current video duration: {total_video_duration:.2f}s")
                            
                            # Allow ±10% tolerance
                            tolerance = video_duration * 0.1
                            if abs(total_video_duration - video_duration) > tolerance:
                                target_duration = video_duration
                                logger.info(f"   Target duration: {target_duration:.2f}s (matching original)")
                        
                        # Adjust video if needed
                        if target_duration:
                            if target_duration > total_video_duration:
                                # Video is shorter than VO - use slow motion on individual scenes if within bounds
                                duration_ratio = target_duration / total_video_duration
                                max_slowdown = 2.0  # Maximum 100% slower (2x duration)
                                
                                if duration_ratio <= max_slowdown:
                                    # Apply slow motion to each scene proportionally
                                    speed_factor = total_video_duration / target_duration
                                    logger.info(f"⏸️ Video ({total_video_duration:.2f}s) is shorter than VO ({target_duration:.2f}s)")
                                    logger.info(f"   Will apply slow motion ({(1-speed_factor)*100:.0f}% slower) after concatenation...")
                                    # Note: Slow motion will be applied to the combined video after concatenation
                                    # This is more efficient than slowing each scene individually
                                else:
                                    logger.info(f"ℹ️ Video ({total_video_duration:.2f}s) is too short for slow motion ({duration_ratio:.2f}x needed > {max_slowdown:.1f}x max)")
                                    logger.info(f"   VO may extend slightly past video end.")
                            elif target_duration < total_video_duration:
                                # Only trim if video is significantly longer than VO (more than 10% longer)
                                # This ensures VO never cuts off
                                excess_ratio = (total_video_duration - target_duration) / target_duration
                                
                                if excess_ratio > 0.1:  # More than 10% longer
                                    trim_amount = total_video_duration - target_duration
                                    logger.info(f"✂️ Video is {excess_ratio*100:.1f}% longer than VO, trimming by {trim_amount:.2f}s...")
                                    
                                    last_scene_idx = len(trimmed_videos) - 1
                                    last_scene = trimmed_videos[last_scene_idx]
                                    last_video_url = last_scene.get("video_url") if isinstance(last_scene, dict) else last_scene
                                    last_scene_duration = last_scene.get("duration", 5.0) if isinstance(last_scene, dict) else 5.0
                                    
                                    new_last_scene_duration = max(1.0, last_scene_duration - trim_amount)  # Min 1 second
                                    
                                    trimmed_last_scene = self.rendi_service.trim_video(
                                        video_url=last_video_url,
                                        duration=new_last_scene_duration
                                    )
                                    
                                    if trimmed_last_scene:
                                        if isinstance(trimmed_videos[last_scene_idx], dict):
                                            trimmed_videos[last_scene_idx]["video_url"] = trimmed_last_scene
                                            trimmed_videos[last_scene_idx]["duration"] = new_last_scene_duration
                                        else:
                                            trimmed_videos[last_scene_idx] = {
                                                "video_url": trimmed_last_scene,
                                                "duration": new_last_scene_duration
                                            }
                                        logger.info(f"✅ Video trimmed to {target_duration:.2f}s")
                                    else:
                                        logger.warning("⚠️ Failed to trim video, continuing with current duration")
                                else:
                                    # Video is only slightly longer - keep it to ensure VO doesn't cut off
                                    logger.info(f"✅ Video is only {excess_ratio*100:.1f}% longer than VO, keeping extra time to ensure VO doesn't cut off")
                            else:
                                logger.info("✅ Video duration already matches target")
                        else:
                            logger.info("✅ Video duration is appropriate")
                    except Exception as e:
                        logger.error(f"❌ Error adjusting video duration: {e}, continuing with current duration")
                
                # Remove any duplicates before concatenation to prevent same video appearing twice
                seen_video_urls = set()
                unique_trimmed_videos = []
                for video_item in trimmed_videos:
                    video_url = video_item.get("video_url") if isinstance(video_item, dict) else video_item
                    if video_url and video_url not in seen_video_urls:
                        seen_video_urls.add(video_url)
                        unique_trimmed_videos.append(video_item)
                    elif video_url in seen_video_urls:
                        logger.warning(f"⚠️ Duplicate video URL detected before concatenation: {video_url[:60]}... - removing duplicate")
                
                if len(unique_trimmed_videos) < len(trimmed_videos):
                    logger.warning(f"⚠️ Removed {len(trimmed_videos) - len(unique_trimmed_videos)} duplicate videos before concatenation")
                    trimmed_videos = unique_trimmed_videos
                
                # Concatenate the trimmed videos with simple concat (more reliable, no repetition)
                # Using simple concat instead of transitions to avoid weird cuts and repetition
                combined_video_url = self.rendi_service.concatenate_videos(
                    trimmed_videos, 
                    use_transitions=False  # Use simple concat for clean cuts without repetition
                )
                if combined_video_url:
                    self._update_sheet_cell(
                        row_num=row_num,
                        column=config.RENDI_SCENE_COLUMN,
                        value=combined_video_url,
                        headers=headers
                    )
                    
                    # Final check: Log if video is shorter than VO (but do NOT loop to avoid jumps)
                    if new_voice_url:
                        try:
                            vo_duration = self.rendi_service.get_audio_duration_cloud(new_voice_url)
                            if vo_duration > 0:
                                combined_duration = self.rendi_service.get_video_duration_cloud(combined_video_url)
                                if combined_duration > 0 and combined_duration < vo_duration:
                                    # Use slow motion to extend video to match VO (up to 2x duration)
                                    duration_ratio = vo_duration / combined_duration
                                    max_slowdown = 2.0  # Maximum 100% slower (2x duration)
                                    
                                    if duration_ratio <= max_slowdown:
                                        speed_factor = combined_duration / vo_duration
                                        logger.info(f"⏸️ Combined video ({combined_duration:.2f}s) is shorter than VO ({vo_duration:.2f}s)")
                                        logger.info(f"   Applying slow motion ({(1-speed_factor)*100:.0f}% slower) to match VO...")
                                        
                                        slowmo_combined = self.rendi_service.slow_motion_video(
                                            video_url=combined_video_url,
                                            speed_factor=speed_factor,
                                            target_duration=vo_duration + 0.3  # Small buffer
                                        )
                                        
                                        if slowmo_combined:
                                            # Upload to S3 if it's a Rendi storage URL
                                            if "storage.rendi.dev" in slowmo_combined:
                                                logger.info(f"   📤 Uploading slow-mo video to S3...")
                                                s3_key = f"rendi_videos/row_{row_num}_combined_slowmo_{int(time.time())}.mp4"
                                                uploaded_url = self.s3_service.upload_video_from_url(slowmo_combined, s3_key)
                                                if uploaded_url:
                                                    slowmo_combined = uploaded_url
                                                    logger.info(f"   ✅ Uploaded to S3: {uploaded_url[:60]}...")
                                            
                                            combined_video_url = slowmo_combined
                                            logger.info(f"✅ Combined video extended with slow motion to match VO")
                                        else:
                                            logger.warning("⚠️ Slow motion failed, VO may extend past video end")
                                    else:
                                        # Too much slowdown needed, keep as-is
                                        logger.info(f"ℹ️ Combined video ({combined_duration:.2f}s) is too short for slow motion ({duration_ratio:.2f}x needed > {max_slowdown:.1f}x max)")
                                        logger.info(f"   Keeping video as-is. VO may extend past video end.")
                                else:
                                    logger.info(f"✅ Video duration ({combined_duration:.2f}s) matches or exceeds VO ({vo_duration:.2f}s)")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not verify video duration vs VO: {e}")
            
            # =================================================================
            # STEP 7: Combine video with audio (TWO-STEP process)
            # =================================================================
            # If we have both voice AND music:
            #   Step 7a: Add voice to video
            #   Step 7b: Add background music to video (overlaid on voice)
            # If we have only voice OR only music:
            #   Single step: Add the available audio
            # =================================================================
            logger.info(f"🎬 [Row {row_num}] Step 7: Combining video with audio...")
            final_video_with_voice = None
            final_video_with_music = None
            
            if combined_video_url and final_audio_url:
                # Step 7a: Add voice/primary audio to video WITH RETRY LOGIC
                logger.info("🎬 Step 7a: Adding primary audio (voice) to video...")
                MAX_AUDIO_RETRIES = 3
                
                for audio_attempt in range(MAX_AUDIO_RETRIES):
                    final_video_with_voice = self.rendi_service.add_audio_to_video(
                        video_url=combined_video_url,
                        audio_url=final_audio_url
                    )
                    if final_video_with_voice:
                        # Validate that the video actually has audio
                        has_audio = self.rendi_service.validate_video_has_audio(final_video_with_voice)
                        if has_audio:
                            logger.info(f"✅ Audio combination successful (attempt {audio_attempt + 1})")
                            break
                        else:
                            logger.warning(f"⚠️ Video has no audio track (attempt {audio_attempt + 1}/{MAX_AUDIO_RETRIES})")
                            final_video_with_voice = None
                    else:
                        logger.warning(f"⚠️ Audio combination failed (attempt {audio_attempt + 1}/{MAX_AUDIO_RETRIES})")
                    
                    if audio_attempt < MAX_AUDIO_RETRIES - 1:
                        logger.info(f"   Retrying in 5 seconds...")
                        time.sleep(5)
                
                # Fallback: If audio combination failed, try using original audio from the video
                if not final_video_with_voice and combined_video_url:
                    logger.warning("⚠️ All audio combination attempts failed, trying fallback...")
                    # Try to extract and re-add original audio as fallback
                    try:
                        original_audio_fallback = FFmpegProcessor.extract_audio_from_url(
                            video_url=video_url,
                            output_path=os.path.join(temp_dir, "fallback_audio.mp3"),
                            rendi_api_key=config.RENDI_API_KEY
                        )
                        if original_audio_fallback:
                            logger.info("🔄 Attempting fallback with original video audio...")
                            final_video_with_voice = self.rendi_service.add_audio_to_video(
                                video_url=combined_video_url,
                                audio_url=original_audio_fallback
                            )
                            if final_video_with_voice:
                                logger.info("✅ Fallback audio combination successful")
                            else:
                                logger.error("❌ Fallback audio combination also failed")
                    except Exception as fallback_error:
                        logger.error(f"❌ Fallback audio extraction failed: {fallback_error}")
                
                if final_video_with_voice:
                    self._update_sheet_cell(
                        row_num=row_num,
                        column=config.RENDI_SCENE_VOICE_COLUMN,
                        value=final_video_with_voice,
                        headers=headers
                    )
                    
                    # Step 7b: If we have background music, add it as overlay (with retry)
                    if new_music_url:
                        # We have music - add it as background (with or without voice)
                        if new_voice_url:
                            logger.info("🎵 Step 7b: Adding background music overlay to video with voice...")
                            base_video = final_video_with_voice
                            music_volume = 0.25  # Music at 25% to not overpower voice
                        else:
                            logger.info("🎵 Step 7b: Adding background music to video (no voice)...")
                            base_video = combined_video_url
                            music_volume = 0.5  # Music at 50% if no voice
                        
                        final_video_with_music = None
                        for music_attempt in range(MAX_AUDIO_RETRIES):
                            if base_video:
                                final_video_with_music = self.rendi_service.add_background_music_to_video(
                                    video_url=base_video,
                                    music_url=new_music_url,
                                    music_volume=music_volume
                                )
                                if final_video_with_music:
                                    logger.info(f"✅ Background music added successfully (attempt {music_attempt + 1}): {final_video_with_music[:80]}...")
                                    # Update the reference to use the version with music
                                    if new_voice_url:
                                        final_video_with_voice = final_video_with_music
                                    break
                                else:
                                    logger.warning(f"⚠️ Music overlay failed (attempt {music_attempt + 1}/{MAX_AUDIO_RETRIES})")
                                    if music_attempt < MAX_AUDIO_RETRIES - 1:
                                        logger.info(f"   Retrying in 3 seconds...")
                                        time.sleep(3)
                            else:
                                logger.error("❌ No base video available for music overlay")
                                break
                        
                        if final_video_with_music:
                            logger.info(f"✅ Step 7b complete: Video with music ready")
                            # Update final_video_with_voice to include music
                            final_video_with_voice = final_video_with_music
                        else:
                            logger.error("❌ CRITICAL: Could not add background music after all retries!")
                            result["errors"].append("Failed to add background music to video")
                else:
                    # Mark as error - don't upload silent video
                    logger.error("❌ CRITICAL: Could not add audio to video after all attempts!")
                    result["errors"].append("Failed to add audio to video - video would be silent")
            
            # Handle case where we have music but no voice
            elif combined_video_url and new_music_url and not final_audio_url:
                logger.info("🎵 Step 7: Adding background music to video (no voice)...")
                final_video_with_music = None
                for music_attempt in range(MAX_AUDIO_RETRIES):
                    final_video_with_music = self.rendi_service.add_background_music_to_video(
                        video_url=combined_video_url,
                        music_url=new_music_url,
                        music_volume=0.5  # Music at 50% if no voice
                    )
                    if final_video_with_music:
                        logger.info(f"✅ Background music added successfully (attempt {music_attempt + 1})")
                        final_video_with_voice = final_video_with_music  # Use this as final video
                        break
                    else:
                        logger.warning(f"⚠️ Music overlay failed (attempt {music_attempt + 1}/{MAX_AUDIO_RETRIES})")
                        if music_attempt < MAX_AUDIO_RETRIES - 1:
                            time.sleep(3)
                
                if not final_video_with_music:
                    logger.error("❌ CRITICAL: Could not add background music after all retries!")
                    result["errors"].append("Failed to add background music to video")
            
            # Step 11: Add subtitles with ZapCap (if requested AND video has speech)
            subtitled_video_url = None
            source_for_subtitles = final_video_with_voice or combined_video_url
            has_speech = audio_result.get("has_speech", False)
            
            if add_subtitles and source_for_subtitles:
                # Only send to ZapCap if video has speech (VO)
                if not has_speech:
                    logger.info("📝 Step 11: Skipping ZapCap - no speech/VO in video (music only)")
                    # No subtitles needed, will upload directly to S3
                elif self.zapcap_service:
                    logger.info("📝 Step 11: Adding subtitles with ZapCap...")
                    
                    # Use subtitle_language if provided, else fall back to article_language, else "en"
                    zapcap_language = subtitle_language or article_language or "en"
                    
                    # Check if we have TTS word segments (from article adaptation TTS generation)
                    tts_word_segments = audio_result.get("tts_word_segments", [])
                    tts_generated = audio_result.get("tts_generated", False)
                    
                    if tts_generated and tts_word_segments:
                        # TTS path: Use timestamped transcript from ElevenLabs
                        logger.info(f"   Using TTS transcript for subtitles ({len(tts_word_segments)} words, language: {zapcap_language})")
                        subtitled_video_url = self.zapcap_service.add_subtitles(
                            video_url=source_for_subtitles,
                            language=zapcap_language,
                            transcript=tts_word_segments
                        )
                    else:
                        # Voice change path: ZapCap will auto-transcribe using the Language column
                        logger.info(f"   Using auto-transcription for subtitles (language: {zapcap_language})")
                        subtitled_video_url = self.zapcap_service.add_subtitles(
                            video_url=source_for_subtitles,
                            language=zapcap_language
                        )
                    
                    if subtitled_video_url:
                        self._update_sheet_cell(
                            row_num=row_num,
                            column=config.SUBTITLED_VIDEO_COLUMN,
                            value=subtitled_video_url,
                            headers=headers
                        )
                        result["subtitled_video_url"] = subtitled_video_url
                        logger.info(f"✅ Subtitles added: {subtitled_video_url}")
                    else:
                        logger.warning("⚠️ Could not add subtitles with ZapCap")
                else:
                    logger.warning("⚠️ ZapCap service not available (no API key)")
            
            # Step 11.5: Add CTA button overlay for "whole_video" mode
            # This is done after all processing so CTA appears throughout the entire video
            if cta_button and cta_text and cta_duration == "whole_video":
                source_for_cta = subtitled_video_url or final_video_with_voice or combined_video_url
                if source_for_cta:
                    logger.info(f"🔘 [Row {row_num}] Adding CTA button for WHOLE VIDEO: '{cta_text}'...")
                    try:
                        # Use the existing temp_dir from the function
                        cta_temp_dir = temp_dir  # Reuse existing temp directory
                        # Generate CTA button image
                        cta_image_url = self.kie_service.generate_cta_button(cta_text)
                        
                        if cta_image_url:
                            # Process CTA button (remove green background)
                            cta_processed_url = self._process_cta_button(
                                cta_image_url=cta_image_url,
                                temp_dir=cta_temp_dir,
                                row_num=row_num
                            )
                            
                            if cta_processed_url:
                                # Overlay CTA for entire video (start_time=0, end_time=None means whole video)
                                video_with_cta = self.rendi_service.overlay_cta_on_video_timed(
                                    video_url=source_for_cta,
                                    cta_image_url=cta_processed_url,
                                    position="center",
                                    start_time=0.0,
                                    end_time=None  # None means until end of video
                                )
                                
                                if video_with_cta:
                                    # Update the source video for S3 upload
                                    if subtitled_video_url:
                                        subtitled_video_url = video_with_cta
                                    elif final_video_with_voice:
                                        final_video_with_voice = video_with_cta
                                    else:
                                        combined_video_url = video_with_cta
                                    logger.info(f"✅ [Row {row_num}] CTA button added for whole video")
                                else:
                                    logger.warning(f"⚠️ [Row {row_num}] Failed to overlay CTA button")
                            else:
                                logger.warning(f"⚠️ [Row {row_num}] Failed to process CTA button image")
                        else:
                            logger.warning(f"⚠️ [Row {row_num}] Failed to generate CTA button image")
                    except Exception as e:
                        logger.warning(f"⚠️ [Row {row_num}] CTA button overlay failed: {e}")
            
            # Step 12: Upload final video to S3
            logger.info("📤 Step 12: Uploading final video to S3...")
            
            # Choose the best available source video (prioritize subtitled if available)
            source_video = subtitled_video_url or final_video_with_voice or combined_video_url
            
            if source_video:
                timestamp = int(time.time())
                final_key = f"final_video_row_{row_num}_{timestamp}.mp4"
                final_s3_url = self.s3_service.upload_video_from_url(
                    source_url=source_video,
                    key_name=final_key
                )
                if final_s3_url:
                    self._update_sheet_cell(
                        row_num=row_num,
                        column=config.FINAL_VIDEO_COLUMN,
                        value=final_s3_url,
                        headers=headers
                    )
                    result["final_video_url"] = final_s3_url
                    result["success"] = True
        
        except Exception as e:
            logger.error(f"❌ [Row {row_num}] UNHANDLED EXCEPTION in process_single_video: {e}")
            logger.error(f"   [Row {row_num}] Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"   [Row {row_num}] Traceback:\n{traceback.format_exc()}")
            result["errors"].append(f"Unhandled exception: {str(e)}")
        
        finally:
            # Clean up temp directory
            try:
                import shutil
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors
            
        return result
    
    def _process_single_scene(
        self,
        scene: Dict[str, Any],
        row_num: int,
        headers: List[str],
        manual_instructions: str = "",
        animation_model: str = "runway",
        target_language: str = "en"
    ) -> Dict[str, Any]:
        """Process a single scene: OpenAI analysis → Nano Banana → Runway/Kling.
        
        This method is designed to run in parallel with other scenes.
        
        Args:
            scene: Scene data with scene_num, start_time, end_time, duration, frame_paths.
            row_num: Row number in the Google Sheet (1-based).
            headers: List of column headers.
            manual_instructions: Optional custom instructions for OpenAI analysis.
            animation_model: Video generation model - "runway" (default) or "kling".
            target_language: Target language for text on images (e.g., 'en', 'he', 'da').
            
        Returns:
            Dict with image_url, video_url, and any errors.
        """
        scene_num = scene["scene_num"]
        scene_duration = scene.get("duration", 5.0)  # Default 5 seconds if not specified
        
        result = {
            "scene_num": scene_num,
            "duration": scene_duration,
            "image_url": None,
            "video_url": None,
            "first_prompt": None,
            "second_prompt": None,
            "error": None
        }
        
        try:
            logger.info(f"🔍 Scene {scene_num}: Analyzing {len(scene['frame_paths'])} frames with OpenAI...")
            
            # Step 1: Analyze frames with OpenAI (with manual instructions if provided)
            analysis = self.openai_service.analyze_scene_frames(
                scene["frame_paths"],
                manual_instructions=manual_instructions
            )
            
            first_prompt = analysis.get("first_prompt", "")
            second_prompt = analysis.get("second_prompt", "")
            result["first_prompt"] = first_prompt
            result["second_prompt"] = second_prompt
            
            # Update Google Sheet with prompts (thread-safe with gspread)
            self._update_sheet_cell(
                row_num=row_num,
                column=config.SCENE_FIRST_PROMPT_PREFIX.format(n=scene_num),
                value=first_prompt,
                headers=headers
            )
            self._update_sheet_cell(
                row_num=row_num,
                column=config.SCENE_SECOND_PROMPT_PREFIX.format(n=scene_num),
                value=second_prompt,
                headers=headers
            )
            
            # Step 2: Generate image with Nano Banana
            if first_prompt:
                logger.info(f"🍌 Scene {scene_num}: Generating image with Nano Banana...")
                image_url = self.kie_service.generate_image_nano_banana(
                    prompt=first_prompt,
                    target_language=target_language
                )
                
                if image_url:
                    result["image_url"] = image_url
                    self._update_sheet_cell(
                        row_num=row_num,
                        column=config.SCENE_NEW_IMAGE_PREFIX.format(n=scene_num),
                        value=image_url,
                        headers=headers
                    )
                    
                    # Step 3: Generate video with Runway or Kling (using scene duration)
                    if second_prompt:
                        second_prompt = self._sanitize_motion_prompt(second_prompt)
                        logger.info(f"🎬 Scene {scene_num}: Generating video with {animation_model.upper()} (target: {scene_duration:.1f}s)...")
                        if animation_model == "kling-3.0":
                            video_url = self.kie_service.generate_video_kling_30(
                                prompt=second_prompt,
                                image_url=image_url,
                                duration=scene_duration
                            )
                        elif animation_model == "kling":
                            video_url = self.kie_service.generate_video_kling(
                                prompt=second_prompt,
                                image_url=image_url,
                                duration=scene_duration
                            )
                        else:
                            video_url = self.kie_service.generate_video_runway(
                                prompt=second_prompt,
                                image_url=image_url,
                                duration=scene_duration
                            )
                        
                        if video_url:
                            # Upload to S3 immediately to avoid temp URL expiration
                            try:
                                s3_video_url = self._upload_video_to_s3_from_url(
                                    video_url=video_url,
                                    row_num=row_num,
                                    scene_num=scene_num
                                )
                                if s3_video_url:
                                    video_url = s3_video_url
                                    logger.info(f"✅ Scene {scene_num}: Video uploaded to S3 (permanent URL)")
                            except Exception as upload_err:
                                logger.warning(f"⚠️ Scene {scene_num}: S3 upload failed, using temp URL: {upload_err}")
                            
                            result["video_url"] = video_url
                            self._update_sheet_cell(
                                row_num=row_num,
                                column=config.SCENE_NEW_VIDEO_PREFIX.format(n=scene_num),
                                value=video_url,
                                headers=headers
                            )
                            logger.info(f"✅ Scene {scene_num}: Video generated successfully!")
                        else:
                            logger.warning(f"⚠️ Scene {scene_num}: {animation_model.upper()} video generation failed")
                else:
                    logger.warning(f"⚠️ Scene {scene_num}: Nano Banana image generation failed")
            else:
                logger.warning(f"⚠️ Scene {scene_num}: No prompt generated by OpenAI")
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Scene {scene_num}: Error during processing: {e}")
        
        return result
    
    def _sanitize_motion_prompt(self, motion_prompt: str, max_length: int = 220) -> str:
        """Ensure motion prompt is camera-first and bounded to reduce animation hallucinations.
        
        - Trims to max_length chars.
        - If prompt does not start with camera movement, prepends a safe default.
        """
        if not motion_prompt or not motion_prompt.strip():
            return "Slow zoom in. Subtle natural motion."
        s = motion_prompt.strip()
        camera_starts = (
            "slow", "zoom", "pan", "dolly", "static", "subtle", "gentle",
            "camera", "tracking", "crane", "tilt", "push", "pull"
        )
        lower = s.lower()
        if not any(lower.startswith(w) for w in camera_starts) and not lower.startswith("same"):
            s = "Slow zoom in. " + s
        if len(s) > max_length:
            s = s[: max_length - 3].rstrip() + "..."
        return s
    
    def _validate_motion_against_visible_elements(
        self,
        motion_prompt: str,
        visible_elements: list
    ) -> str:
        """Filter motion_prompt to only reference items in visible_elements.
        
        Extracts body-part/object keywords from motion_prompt and removes
        clauses that reference elements not in visible_elements.
        If the prompt becomes empty after filtering, returns camera-only motion.
        """
        if not motion_prompt or not motion_prompt.strip():
            return motion_prompt
        if not visible_elements:
            return motion_prompt
        
        visible_lower = set(v.lower() for v in visible_elements)
        visible_text = " ".join(visible_lower)
        
        body_object_keywords = {
            "hand": ["hand", "hands", "finger", "fingers", "grip", "grasp", "hold"],
            "arm": ["arm", "arms", "elbow", "reach", "reaching"],
            "leg": ["leg", "legs", "knee", "step", "stepping", "walk", "walking"],
            "foot": ["foot", "feet", "toe", "toes"],
            "face": ["face", "smile", "smiling", "blink", "blinking", "expression", "frown", "grin", "lip", "lips", "mouth", "eyes", "eye"],
            "head": ["head", "nod", "nodding", "turn", "turning"],
            "body": ["body", "torso", "chest", "shoulder", "shoulders", "lean", "leaning", "sway", "swaying", "breathing", "breath"],
            "bag": ["bag", "purse", "backpack", "tote", "satchel"],
            "bottle": ["bottle", "jar", "container", "tube"],
            "cup": ["cup", "mug", "glass", "drink"],
            "phone": ["phone", "smartphone", "device", "screen"],
            "box": ["box", "package", "parcel"],
            "product": ["product", "item", "object", "patch", "cream", "massager"],
        }
        
        def element_is_visible(keyword_group_name: str) -> bool:
            if keyword_group_name in visible_lower:
                return True
            for v in visible_lower:
                if keyword_group_name in v or v in keyword_group_name:
                    return True
            group_keywords = body_object_keywords.get(keyword_group_name, [keyword_group_name])
            return any(kw in visible_text for kw in group_keywords)
        
        clauses = re.split(r'[,;.]+', motion_prompt)
        if not clauses:
            return motion_prompt
        
        camera_keywords = {"slow", "zoom", "pan", "dolly", "static", "subtle", "gentle",
                          "camera", "tracking", "crane", "tilt", "push", "pull", "truck",
                          "orbit", "steadicam", "handheld", "aerial", "forward", "backward"}
        
        filtered_clauses = []
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            clause_lower = clause.lower()
            
            is_camera = any(kw in clause_lower for kw in camera_keywords)
            if is_camera:
                filtered_clauses.append(clause)
                continue
            
            should_remove = False
            for group_name, keywords in body_object_keywords.items():
                for kw in keywords:
                    if kw in clause_lower:
                        if not element_is_visible(group_name):
                            should_remove = True
                            logger.debug(f"   Motion validation: removing '{clause}' - '{group_name}' not in visible_elements")
                            break
                if should_remove:
                    break
            
            if not should_remove:
                filtered_clauses.append(clause)
        
        if not filtered_clauses:
            return "Slow zoom in."
        
        result = ", ".join(filtered_clauses)
        if result != motion_prompt:
            logger.info(f"   Motion validation: filtered '{motion_prompt[:60]}...' -> '{result[:60]}...'")
        return result
    
    def _process_cta_button(
        self,
        cta_image_url: str,
        temp_dir: str,
        row_num: int
    ) -> Optional[str]:
        """Process CTA button image: check if already processed, or download and process.
        
        If the image was generated by PIL (already has transparent background), 
        return it directly. Otherwise, download, remove green background, and upload.
        
        Args:
            cta_image_url: URL or path of the generated CTA button image.
            temp_dir: Temporary directory for processing.
            row_num: Row number for logging.
            
        Returns:
            URL of the processed CTA button image, or None if failed.
        """
        try:
            # Check if it's already uploaded to S3 (PIL-generated buttons are uploaded directly)
            # PIL-generated buttons have "cta_button_" in their S3 key
            if "cta_button_" in cta_image_url and "s3.amazonaws.com" in cta_image_url:
                logger.info(f"✅ CTA button already processed (PIL-generated): {cta_image_url}")
                return cta_image_url
            
            # Check if it's a local file path (fallback case)
            if os.path.isfile(cta_image_url):
                logger.info("🎨 Processing local CTA button image...")
                with open(cta_image_url, 'rb') as f:
                    image_data = f.read()
                
                timestamp = int(time.time())
                cta_key = f"cta_button_row_{row_num}_{timestamp}.png"
                
                cta_url = self.s3_service.upload_image_bytes(
                    image_data=image_data,
                    key_name=cta_key
                )
                
                if cta_url:
                    logger.info(f"✅ CTA button uploaded: {cta_url}")
                    return cta_url
                else:
                    logger.error("❌ Failed to upload CTA button to S3")
                    return None
            
            # Old flow: download from URL and process (for Nano Banana generated buttons)
            logger.info("🎨 Processing CTA button image from URL...")
            
            # Download the CTA image
            response = requests.get(cta_image_url, timeout=60)
            response.raise_for_status()
            
            original_path = os.path.join(temp_dir, "cta_original.png")
            with open(original_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"✅ Downloaded CTA image: {original_path}")
            
            # Remove green background (samples actual green from image)
            final_path = os.path.join(temp_dir, "cta_no_bg.png")
            if not remove_green_background(original_path, final_path):
                logger.warning("⚠️ Failed to remove background, using original")
                final_path = original_path
            
            # Upload to S3
            with open(final_path, 'rb') as f:
                image_data = f.read()
            
            timestamp = int(time.time())
            cta_key = f"cta_button_row_{row_num}_{timestamp}.png"
            
            # Upload to S3 (using the S3 service)
            cta_url = self.s3_service.upload_image_bytes(
                image_data=image_data,
                key_name=cta_key
            )
            
            if cta_url:
                logger.info(f"✅ CTA button processed and uploaded: {cta_url}")
                return cta_url
            else:
                logger.error("❌ Failed to upload CTA button to S3")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error processing CTA button: {e}")
            return None
    
    def _upload_video_to_s3_from_url(
        self,
        video_url: str,
        row_num: int,
        scene_num: int
    ) -> Optional[str]:
        """Download video from temp URL and upload to S3 for permanent storage.
        
        This prevents issues with temp URLs expiring before Rendi can use them.
        
        Args:
            video_url: Temporary video URL from Kling/Runway.
            row_num: Row number for naming.
            scene_num: Scene number for naming.
            
        Returns:
            S3 URL if successful, None otherwise.
        """
        try:
            import time
            
            # Use the existing S3 service method for uploading from URL
            s3_key = f"Comp/Final_Video/scene_video_row_{row_num}_scene_{scene_num}_{int(time.time())}.mp4"
            
            s3_url = self.s3_service.upload_video_from_url(
                source_url=video_url,
                key_name=s3_key
            )
            
            return s3_url
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to upload video to S3: {e}")
            return None
    
    def _update_sheet_cell(
        self, 
        row_num: int, 
        column: str, 
        value: str,
        headers: List[str]
    ) -> bool:
        """Update a cell in the Google Sheet (thread-safe with lock).
        
        The underlying GoogleSheetsService.update_cell has built-in retry logic
        with exponential backoff for Google Sheets API errors.
        
        Includes throttling to avoid Google Sheets rate limits (60 req/min).
        
        Args:
            row_num: Row number (1-based).
            column: Column name.
            value: Value to set.
            headers: List of column headers.
            
        Returns:
            True if update was successful, False otherwise.
        """
        # Pace writes globally to stay under the Sheets 60/min quota.
        # acquire() blocks OUTSIDE the lock, so a queued writer never freezes
        # the other worker threads (and 429 backoffs become rare).
        self._sheets_rate_limiter.acquire()
        with self._sheets_lock:  # Brief lock only around the API call (gspread thread-safety)
            try:
                self.sheets_service.update_cell(
                    sheet_id=config.GOOGLE_SHEET_ID,
                    worksheet_name=config.GOOGLE_SHEET_TAB,
                    row=row_num,
                    column_name=column,
                    value=value,
                    headers=headers
                )
                return True

            except ValueError as e:
                logger.warning(f"⚠️ Column '{column}' not found, skipping update")
                return False
                
            except Exception as e:
                # Service already retried, this is the final failure
                logger.error(f"❌ Failed to update cell ({row_num}, {column}) after retries: {e}")
                return False

    def process_influencer_row(
        self,
        row_num: int,
        row_data: List[str],
        headers: List[str],
        free_text: str,
        manual_instructions: str = "",
        language: str = "",
        cta_button: bool = False,
        cta_text: str = "",
        cta_duration: str = "at_the_end",
        add_subtitles: bool = False,
        manual_vo_text: str = "",
        manual_music_link: str = "",
        image_urls: List[str] = None,
        scene_count: int = None,
        voice_id: str = ""
    ) -> Dict[str, Any]:
        """Process a row in influencer mode (no input video, generate from Free text).
        
        Creates an influencer-style recommendation video based on Free text content.
        
        Args:
            row_num: Row number in the sheet.
            row_data: Full row data from the sheet.
            headers: List of column headers.
            free_text: Content describing the product/experience.
            manual_instructions: Optional custom instructions.
            language: ISO language code (detected from text if empty).
            cta_button: Whether to add CTA button.
            cta_text: Text for CTA button.
            add_subtitles: Whether to add subtitles.
            manual_vo_text: Optional manual VO text override.
            manual_music_link: Optional manual music link.
            image_urls: List of reference image URLs (Image 1-4).
            scene_count: Number of scenes to generate (default 6).
            voice_id: Custom ElevenLabs voice ID (uses default if empty).
            
        Returns:
            Dict with processing results.
        """
        result = {
            "row": row_num,
            "success": False,
            "mode": "influencer",
            "scenes_processed": 0,
            "errors": [],
            "final_video_url": None
        }
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎭 [Row {row_num}] INFLUENCER MODE - Generating recommendation video")
            logger.info(f"{'='*60}")
            
            # Set defaults
            scene_count = scene_count or config.DEFAULT_INFLUENCER_SCENES
            image_urls = image_urls or []
            
            # Step 1: Detect and set language
            if not language:
                language = detect_language(free_text)
                logger.info(f"🌍 [Row {row_num}] Detected language: {language}")
                self._update_sheet_cell(row_num, config.LANGUAGE_COLUMN, language, headers)
            
            logger.info(f"📝 [Row {row_num}] Free text: {len(free_text)} chars")
            logger.info(f"🎬 [Row {row_num}] Scene count: {scene_count}")
            logger.info(f"🖼️ [Row {row_num}] Reference images: {len(image_urls)}")
            logger.info(f"🔘 [Row {row_num}] CTA button: {cta_button}, CTA text: '{cta_text}'")
            logger.info(f"📝 [Row {row_num}] Add subtitles: {add_subtitles}")
            
            # Step 2: Download and analyze reference images
            reference_images = []
            for i, img_url in enumerate(image_urls):
                if img_url:
                    try:
                        logger.info(f"📥 [Row {row_num}] Downloading reference image {i+1}...")
                        response = requests.get(img_url, timeout=30)
                        response.raise_for_status()
                        
                        # Convert to base64 for OpenAI analysis
                        img_base64 = base64.b64encode(response.content).decode('utf-8')
                        
                        # Get image analysis from OpenAI
                        analysis_result = self.openai_service._generate_image_prompt(
                            [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}]
                        )
                        
                        reference_images.append({
                            "index": i + 1,
                            "url": img_url,
                            "base64": img_base64,
                            "analysis": analysis_result.get("analysis", "")[:500] if analysis_result else ""
                        })
                        logger.info(f"✅ [Row {row_num}] Reference image {i+1} analyzed")
                    except Exception as e:
                        logger.warning(f"⚠️ [Row {row_num}] Could not process reference image {i+1}: {e}")
                        reference_images.append({"index": i + 1, "url": img_url, "analysis": ""})
            
            # Step 3: Generate influencer prompts with OpenAI
            logger.info(f"🎭 [Row {row_num}] Generating influencer prompts...")
            prompts_result = self.openai_service.generate_influencer_prompts(
                free_text=free_text,
                reference_images=reference_images,
                scene_count=scene_count,
                manual_instructions=manual_instructions,
                cta_text=cta_text,
                language=language
            )
            
            scene_prompts = prompts_result.get("scene_prompts", [])
            influencer_description = prompts_result.get("influencer_description", "")
            
            if not scene_prompts:
                raise Exception("Failed to generate influencer prompts")
            
            logger.info(f"✅ [Row {row_num}] Generated {len(scene_prompts)} scene prompts")
            
            # Step 4: Write prompts to sheet and generate images/videos
            scene_videos = []
            scene_durations = []
            
            # Process scenes in parallel
            with ThreadPoolExecutor(max_workers=min(scene_count, 7)) as executor:
                futures = {}
                
                for prompt_data in scene_prompts:
                    scene_num = prompt_data.get("scene_number", 1)
                    first_prompt = prompt_data.get("first_prompt", "")
                    second_prompt = prompt_data.get("second_prompt", "")
                    ref_image_index = prompt_data.get("reference_image_index")
                    
                    # Write prompts to sheet
                    self._update_sheet_cell(
                        row_num, 
                        config.SCENE_FIRST_PROMPT_PREFIX.format(n=scene_num), 
                        first_prompt, 
                        headers
                    )
                    self._update_sheet_cell(
                        row_num, 
                        config.SCENE_SECOND_PROMPT_PREFIX.format(n=scene_num), 
                        second_prompt, 
                        headers
                    )
                    
                    # Get reference image for this scene (cycling through available images)
                    ref_url = None
                    ref_desc = None
                    if reference_images:
                        # Cycle through images: scene 1 -> img 0, scene 2 -> img 1, etc.
                        img_index = (scene_num - 1) % len(reference_images)
                        ref_img = reference_images[img_index]
                        ref_url = ref_img.get("url")
                        ref_desc = ref_img.get("analysis", "")[:300]
                    
                    scene_visible_elements = prompt_data.get("visible_elements", [])
                    
                    future = executor.submit(
                        self._process_influencer_scene,
                        scene_num=scene_num,
                        first_prompt=first_prompt,
                        second_prompt=second_prompt,
                        reference_image_url=ref_url,
                        reference_description=ref_desc,
                        row_num=row_num,
                        headers=headers,
                        visible_elements=scene_visible_elements
                    )
                    futures[future] = scene_num
                
                # Collect results
                for future in as_completed(futures):
                    scene_num = futures[future]
                    try:
                        scene_result = future.result()
                        if scene_result.get("video_url"):
                            scene_videos.append({
                                "scene_num": scene_num,
                                "video_url": scene_result["video_url"],
                                "duration": config.INFLUENCER_SCENE_DURATION
                            })
                            scene_durations.append(config.INFLUENCER_SCENE_DURATION)
                            result["scenes_processed"] += 1
                            logger.info(f"✅ [Row {row_num}] Scene {scene_num} completed")
                        else:
                            result["errors"].append(f"Scene {scene_num}: No video generated")
                    except Exception as e:
                        result["errors"].append(f"Scene {scene_num}: {str(e)}")
                        logger.error(f"❌ [Row {row_num}] Scene {scene_num} failed: {e}")
            
            # Sort scenes by number
            scene_videos.sort(key=lambda x: x["scene_num"])
            video_urls = [s["video_url"] for s in scene_videos]
            
            if not video_urls:
                raise Exception("No scene videos were generated")
            
            # Step 5: Concatenate videos
            logger.info(f"🎬 [Row {row_num}] Concatenating {len(video_urls)} scene videos...")
            combined_video = self.rendi_service.concatenate_videos(video_urls)
            
            if not combined_video:
                raise Exception("Failed to concatenate videos")
            
            self._update_sheet_cell(row_num, config.RENDI_SCENE_COLUMN, combined_video, headers)
            logger.info(f"✅ [Row {row_num}] Videos concatenated: {combined_video}")
            
            # Calculate total video duration
            total_video_duration = len(video_urls) * config.INFLUENCER_SCENE_DURATION
            
            # Step 6: Generate VO
            logger.info(f"🎤 [Row {row_num}] Generating voice-over...")
            
            if manual_vo_text:
                vo_script = manual_vo_text
                logger.info(f"📝 [Row {row_num}] Using manual VO text")
            else:
                vo_script = self.openai_service.generate_influencer_vo_script(
                    free_text=free_text,
                    scene_count=scene_count,
                    target_duration=total_video_duration,
                    manual_instructions=manual_instructions,
                    language=language
                )
            
            # Generate TTS with timestamps for precise subtitle synchronization
            # Use custom voice_id if valid, otherwise use default female voice
            tts_voice_id = get_validated_voice_id(voice_id, config.DEFAULT_FEMALE_VOICE_ID)
            logger.info(f"🎤 [Row {row_num}] Using voice ID: {tts_voice_id}")
            
            tts_result = self.elevenlabs_service.text_to_speech_with_timestamps(
                text=vo_script,
                voice_id=tts_voice_id,
                language=language
            )
            
            # Store word segments for ZapCap
            word_segments = []
            
            if tts_result:
                voice_audio, word_segments = tts_result
                logger.info(f"📝 [Row {row_num}] Got {len(word_segments)} word segments from TTS")
                
                # Upload voice to S3
                timestamp = int(time.time())
                voice_key = f"influencer_voice_row_{row_num}_{timestamp}.mp3"
                voice_url = self.s3_service.upload_audio_bytes(voice_audio, voice_key)
                
                if voice_url:
                    self._update_sheet_cell(row_num, config.NEW_VOICE_COLUMN, voice_url, headers)
                    logger.info(f"✅ [Row {row_num}] Voice generated: {voice_url}")
                else:
                    raise Exception("Failed to upload voice to S3")
            else:
                raise Exception("Failed to generate voice")
            
            # Step 7: Generate or use manual music
            if manual_music_link:
                music_url = manual_music_link
                logger.info(f"🎵 [Row {row_num}] Using manual music link")
            else:
                logger.info(f"🎵 [Row {row_num}] Generating background music...")
                music_description = self.openai_service.generate_music_description_from_text(free_text[:1000])
                # Use pure music generation (no reference audio) to avoid copyright issues
                music_url = self.suno_service.generate_pure_music(
                    style_description=music_description
                )
            
            if music_url:
                self._update_sheet_cell(row_num, config.NEW_MUSIC_COLUMN, music_url, headers)
                logger.info(f"✅ [Row {row_num}] Music ready: {music_url}")
            
            # Step 8: Combine video with voice
            logger.info(f"🎬 [Row {row_num}] Adding voice to video...")
            video_with_voice = self.rendi_service.add_audio_to_video(
                video_url=combined_video,
                audio_url=voice_url
            )
            
            if video_with_voice:
                self._update_sheet_cell(row_num, config.RENDI_SCENE_VOICE_COLUMN, video_with_voice, headers)
                logger.info(f"✅ [Row {row_num}] Voice added: {video_with_voice}")
            else:
                raise Exception("Failed to add voice to video")
            
            # Step 9: Add background music
            final_video = video_with_voice  # Default to voice-only
            if music_url:
                logger.info(f"🎵 [Row {row_num}] Adding background music (volume: 0.2)...")
                video_with_music = self.rendi_service.add_background_music_to_video(
                    video_url=video_with_voice,
                    music_url=music_url,
                    music_volume=0.2  # Lower volume for background music
                )
                if video_with_music:
                    final_video = video_with_music
                    logger.info(f"✅ [Row {row_num}] Background music added: {final_video}")
                else:
                    logger.warning(f"⚠️ [Row {row_num}] Failed to add music, using voice-only version")
            
            # Step 9.5: Add CTA button overlay if requested
            if cta_button and cta_text:
                # Determine CTA timing based on cta_duration setting
                if cta_duration == "whole_video":
                    logger.info(f"🔘 [Row {row_num}] Adding CTA button overlay for WHOLE VIDEO: '{cta_text}'...")
                else:
                    logger.info(f"🔘 [Row {row_num}] Adding CTA button overlay to last scene: '{cta_text}'...")
                
                try:
                    import tempfile
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Step 1: Generate CTA button image
                        cta_image_url = self.kie_service.generate_cta_button(cta_text)
                        
                        if cta_image_url:
                            logger.info(f"✅ [Row {row_num}] CTA button image generated: {cta_image_url[:50]}...")
                            
                            # Step 2: Process CTA button (remove green background)
                            cta_processed_url = self._process_cta_button(
                                cta_image_url=cta_image_url,
                                temp_dir=temp_dir,
                                row_num=row_num
                            )
                            
                            if cta_processed_url:
                                # Step 3: Overlay CTA button on video
                                if cta_duration == "whole_video":
                                    # CTA appears for the entire video
                                    cta_start_time = 0.0
                                    cta_end_time = total_video_duration
                                    logger.info(f"   CTA will appear from 0.0s to {total_video_duration:.1f}s (whole video)")
                                else:
                                    # CTA appears only in the last scene (last 5 seconds)
                                    cta_start_time = max(0, total_video_duration - 5.0)
                                    cta_end_time = total_video_duration
                                    logger.info(f"   CTA will appear from {cta_start_time:.1f}s to {total_video_duration:.1f}s (last scene)")
                                
                                video_with_cta = self.rendi_service.overlay_cta_on_video_timed(
                                    video_url=final_video,
                                    cta_image_url=cta_processed_url,
                                    position="center",
                                    start_time=cta_start_time,
                                    end_time=cta_end_time
                                )
                                
                                if video_with_cta:
                                    final_video = video_with_cta
                                    if cta_duration == "whole_video":
                                        logger.info(f"✅ [Row {row_num}] CTA button added for whole video")
                                    else:
                                        logger.info(f"✅ [Row {row_num}] CTA button added to last scene (starts at {cta_start_time:.1f}s)")
                                else:
                                    logger.warning(f"⚠️ [Row {row_num}] Failed to overlay CTA button")
                            else:
                                logger.warning(f"⚠️ [Row {row_num}] Failed to process CTA button image")
                        else:
                            logger.warning(f"⚠️ [Row {row_num}] Failed to generate CTA button image")
                except Exception as e:
                    logger.warning(f"⚠️ [Row {row_num}] CTA button overlay failed: {e}")
            
            # Step 10: Add subtitles if requested
            subtitled_video = None
            if add_subtitles and self.zapcap_service:
                logger.info(f"📝 [Row {row_num}] Adding subtitles...")
                # Pass word segments for precise timing (Bring Your Own Transcript)
                subtitled_video = self.zapcap_service.add_subtitles(
                    video_url=final_video or video_with_voice,
                    language=language,
                    transcript=word_segments if word_segments else None
                )
                if subtitled_video:
                    self._update_sheet_cell(row_num, config.SUBTITLED_VIDEO_COLUMN, subtitled_video, headers)
                    logger.info(f"✅ [Row {row_num}] Subtitles added with precise timing")
            
            # Step 11: Upload final video to GCS
            logger.info(f"📤 [Row {row_num}] Uploading final video to GCS...")
            source_video = subtitled_video or final_video or video_with_voice
            
            timestamp = int(time.time())
            final_gcs_url = self.gcs_video_service.upload_video_from_url(
                source_url=source_video,
                key_name=f"influencer_final_row_{row_num}_{timestamp}.mp4",
                folder="influencer_videos"
            )
            
            if final_gcs_url:
                self._update_sheet_cell(row_num, config.FINAL_VIDEO_COLUMN, final_gcs_url, headers)
                result["final_video_url"] = final_gcs_url
                result["success"] = True
                logger.info(f"✅ [Row {row_num}] Final video uploaded to GCS: {final_gcs_url}")
            else:
                raise Exception("Failed to upload final video to GCS")
            
            logger.info(f"\n🎉 [Row {row_num}] Influencer video completed successfully!")
            
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"❌ [Row {row_num}] Influencer mode failed: {e}")
        
        return result

    def _process_influencer_scene(
        self,
        scene_num: int,
        first_prompt: str,
        second_prompt: str,
        reference_image_url: Optional[str],
        reference_description: Optional[str],
        row_num: int,
        headers: List[str],
        animation_model: str = "runway",
        target_language: str = "en",
        visible_elements: list = None
    ) -> Dict[str, Any]:
        """Process a single influencer scene (image + video generation).
        
        Args:
            scene_num: Scene number.
            first_prompt: Image generation prompt.
            second_prompt: Motion/video prompt.
            reference_image_url: Optional reference image URL.
            reference_description: Optional reference image description.
            row_num: Row number for sheet updates.
            headers: Column headers.
            target_language: Target language for text on images.
            visible_elements: List of visible elements in the scene for motion validation.
            
        Returns:
            Dict with image_url and video_url.
        """
        result = {"scene_num": scene_num, "image_url": None, "video_url": None}
        
        # When using product reference image, always pass a text description to reduce hallucinations
        if reference_image_url and (not reference_description or not reference_description.strip()):
            reference_description = "Match the product appearance from the reference image (shape, colors, materials)."
        
        try:
            # Generate image with Nano Banana
            logger.info(f"🎨 [Scene {scene_num}] Generating image...")
            image_url = self.kie_service.generate_image_nano_banana(
                prompt=first_prompt,
                reference_image_url=reference_image_url,
                reference_description=reference_description,
                target_language=target_language
            )
            
            if image_url:
                result["image_url"] = image_url
                self._update_sheet_cell(
                    row_num,
                    config.SCENE_NEW_IMAGE_PREFIX.format(n=scene_num),
                    image_url,
                    headers
                )
                
                # Validate and sanitize motion prompt
                if visible_elements:
                    second_prompt = self._validate_motion_against_visible_elements(second_prompt, visible_elements)
                second_prompt = self._sanitize_motion_prompt(second_prompt)
                logger.info(f"🎬 [Scene {scene_num}] Generating video with {animation_model.upper()}...")
                if animation_model == "kling-3.0":
                    video_url = self.kie_service.generate_video_kling_30(
                        prompt=second_prompt,
                        image_url=image_url,
                        duration=config.INFLUENCER_SCENE_DURATION
                    )
                elif animation_model == "kling":
                    video_url = self.kie_service.generate_video_kling(
                        prompt=second_prompt,
                        image_url=image_url,
                        duration=config.INFLUENCER_SCENE_DURATION
                    )
                else:
                    video_url = self.kie_service.generate_video_runway(
                        prompt=second_prompt,
                        image_url=image_url,
                        duration=config.INFLUENCER_SCENE_DURATION
                    )
                
                if video_url:
                    # Upload to S3 immediately to avoid temp URL expiration
                    try:
                        s3_video_url = self._upload_video_to_s3_from_url(
                            video_url=video_url,
                            row_num=row_num,
                            scene_num=scene_num
                        )
                        if s3_video_url:
                            video_url = s3_video_url
                            logger.info(f"✅ [Scene {scene_num}] Video uploaded to S3 (permanent URL)")
                    except Exception as upload_err:
                        logger.warning(f"⚠️ [Scene {scene_num}] S3 upload failed, using temp URL: {upload_err}")
                    
                    result["video_url"] = video_url
                    self._update_sheet_cell(
                        row_num,
                        config.SCENE_NEW_VIDEO_PREFIX.format(n=scene_num),
                        video_url,
                        headers
                    )
                    logger.info(f"✅ [Scene {scene_num}] Video generated")
                else:
                    logger.warning(f"⚠️ [Scene {scene_num}] Video generation failed")
            else:
                logger.warning(f"⚠️ [Scene {scene_num}] Image generation failed")
                
        except Exception as e:
            logger.error(f"❌ [Scene {scene_num}] Error: {e}")
        
        return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point for the video scene processor."""
    logger.info("="*60)
    logger.info("🎬 VIDEO SCENE PROCESSOR - TVD X1 PIPELINE")
    logger.info("="*60)
    
    try:
        processor = VideoSceneProcessor()
        results = processor.process_all_videos()
        
        logger.info("\n📊 Final Results:")
        logger.info(json.dumps(results, indent=2, default=str))
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()

