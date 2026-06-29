#!/usr/bin/env python3
"""
Test script for final stages of video processing pipeline.

This script reads existing scene videos from Google Sheet and runs only:
1. RENDI Scene - Trim and concatenate scene videos
2. New Voice - ElevenLabs voice changer
3. RENDI Scene & Voice - Combine video + voice
4. Final Video - Upload to S3

Usage:
    python scripts/run_final_stages.py [--row ROW_NUMBER]
    
    --row: Specific row to process (default: 2, first data row)
"""

import os
import sys
import json
import time
import logging
import argparse
import tempfile
import requests
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Ensure the repo root is importable when run from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import from main script
from video_scene_processor import (
    config,
    logger,
    GoogleSheetsService,
    RendiService,
    ElevenLabsService,
    S3Service,
    FFmpegProcessor
)
from openai import OpenAI

# =============================================================================
# TEST CONFIGURATION
# =============================================================================
class TestConfig:
    """Test-specific configuration."""
    
    # Which row to process (1-based, row 2 = first data row)
    DEFAULT_ROW: int = 2
    
    # Scene video columns to read
    SCENE_VIDEO_COLUMNS: List[str] = [
        "Scene 1 - new video",
        "Scene 2 - new video",
        "Scene 3 - new video",
        "Scene 4 - new video",
        "Scene 5 - new video",
        "Scene 6 - new video",
        "Scene 7 - new video",
        "Scene 8 - new video",
    ]


# =============================================================================
# FINAL STAGES PROCESSOR
# =============================================================================
class FinalStagesProcessor:
    """Processor for final stages only (RENDI Scene → Voice → Combined → S3)."""
    
    def __init__(self):
        """Initialize services."""
        logger.info("="*60)
        logger.info("🧪 TEST: Final Stages Processor")
        logger.info("="*60)
        
        # Initialize services
        self.sheets_service = GoogleSheetsService(config.SERVICE_ACCOUNT_FILE)
        self.rendi_service = RendiService(config.RENDI_API_KEY)
        self.s3_service = S3Service(
            access_key_id=config.AWS_ACCESS_KEY_ID,
            secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            bucket_name=config.AWS_BUCKET_NAME,
            region=config.AWS_REGION,
            folder_path=config.AWS_FOLDER_PATH
        )
        
        # Initialize OpenAI client for ElevenLabs speech detection
        openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.elevenlabs_service = ElevenLabsService(
            api_key=config.ELEVENLABS_API_KEY,
            openai_client=openai_client
        )
        
        # Set Rendi API key for FFmpeg cloud operations
        FFmpegProcessor.set_rendi_api_key(config.RENDI_API_KEY)
        
        logger.info("✅ All services initialized")
    
    def get_sheet_data(self) -> Tuple[List[str], List[List[str]]]:
        """Get data from Google Sheet.
        
        Returns:
            Tuple of (headers, data_rows).
        """
        return self.sheets_service.get_worksheet_data(
            sheet_id=config.GOOGLE_SHEET_ID,
            worksheet_name=config.GOOGLE_SHEET_TAB
        )
    
    def get_column_value(
        self, 
        row_data: List[str], 
        headers: List[str], 
        column_name: str
    ) -> str:
        """Get value from a specific column.
        
        Args:
            row_data: Row data list.
            headers: Column headers.
            column_name: Name of the column.
            
        Returns:
            Cell value or empty string.
        """
        try:
            col_idx = headers.index(column_name)
            if col_idx < len(row_data):
                return row_data[col_idx].strip()
            return ""
        except (ValueError, IndexError):
            return ""
    
    def update_cell(
        self, 
        row_num: int, 
        column_name: str, 
        value: str, 
        headers: List[str]
    ) -> None:
        """Update a cell in the Google Sheet.
        
        Args:
            row_num: Row number (1-based).
            column_name: Column name.
            value: Value to set.
            headers: Column headers.
        """
        try:
            self.sheets_service.update_cell(
                sheet_id=config.GOOGLE_SHEET_ID,
                worksheet_name=config.GOOGLE_SHEET_TAB,
                row=row_num,
                column_name=column_name,
                value=value,
                headers=headers
            )
        except Exception as e:
            logger.error(f"❌ Error updating cell ({row_num}, {column_name}): {e}")
    
    def process_row(self, row_num: int) -> Dict[str, Any]:
        """Process a specific row - final stages only.
        
        Args:
            row_num: Row number to process (1-based, row 2 = first data row).
            
        Returns:
            Dict with results.
        """
        result = {
            "row": row_num,
            "success": False,
            "rendi_scene_url": None,
            "new_voice_url": None,
            "rendi_scene_voice_url": None,
            "final_video_url": None,
            "errors": []
        }
        
        # Get sheet data
        headers, data_rows = self.get_sheet_data()
        
        # Get the row data (row_num is 1-based, headers are row 1)
        row_index = row_num - 2  # Convert to 0-based index for data_rows
        if row_index < 0 or row_index >= len(data_rows):
            result["errors"].append(f"Row {row_num} not found in sheet")
            return result
        
        row_data = data_rows[row_index]
        
        # Get original video URL
        original_video_url = self.get_column_value(row_data, headers, config.INPUT_VIDEO_COLUMN)
        if not original_video_url:
            result["errors"].append("No input video URL found")
            return result
        
        logger.info(f"📹 Original video: {original_video_url[:50]}...")
        
        # Collect existing scene videos with durations
        logger.info("📥 Reading existing scene videos from Google Sheet...")
        scene_videos_with_durations = []
        
        # First, get video duration to estimate scene durations
        logger.info("🎬 Getting original video duration...")
        video_duration = self.rendi_service.get_video_duration_cloud(original_video_url)
        logger.info(f"✅ Video duration: {video_duration:.2f}s")
        
        # Read scene videos from sheet
        scene_count = 0
        for col_name in TestConfig.SCENE_VIDEO_COLUMNS:
            video_url = self.get_column_value(row_data, headers, col_name)
            if video_url and video_url.startswith("http"):
                scene_count += 1
                scene_videos_with_durations.append({
                    "video_url": video_url,
                    "duration": None  # Will calculate below
                })
                logger.info(f"   ✅ {col_name}: Found video URL")
            else:
                logger.info(f"   ⏭️ {col_name}: No video (skipping)")
        
        if not scene_videos_with_durations:
            result["errors"].append("No scene videos found in sheet")
            return result
        
        # Calculate approximate durations (divide video evenly among scenes)
        scene_duration = video_duration / len(scene_videos_with_durations)
        for item in scene_videos_with_durations:
            item["duration"] = scene_duration
        
        logger.info(f"📊 Found {len(scene_videos_with_durations)} scene videos")
        logger.info(f"   Estimated duration per scene: {scene_duration:.2f}s")
        
        # Create temp directory for audio processing
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # =================================================================
            # STEP 1: RENDI Scene - Trim and Concatenate
            # =================================================================
            logger.info("")
            logger.info("=" * 50)
            logger.info("🎬 STEP 1: RENDI Scene - Trim & Concatenate")
            logger.info("=" * 50)
            
            # Trim each video to its duration
            logger.info(f"✂️ Trimming {len(scene_videos_with_durations)} videos...")
            trimmed_videos = self.rendi_service.trim_videos_batch(scene_videos_with_durations)
            
            if trimmed_videos:
                logger.info(f"✅ Trimmed {len(trimmed_videos)} videos")
                
                # Concatenate (using simple method - more reliable, no freezing)
                logger.info("🔗 Concatenating videos...")
                combined_video_url = self.rendi_service.concatenate_videos(
                    trimmed_videos, 
                    use_transitions=False
                )
                
                if combined_video_url:
                    result["rendi_scene_url"] = combined_video_url
                    self.update_cell(row_num, config.RENDI_SCENE_COLUMN, combined_video_url, headers)
                    logger.info(f"✅ RENDI Scene: {combined_video_url[:60]}...")
                else:
                    result["errors"].append("Failed to concatenate videos")
            else:
                result["errors"].append("Failed to trim videos")
            
            # =================================================================
            # STEP 2: New Voice - Extract audio and apply ElevenLabs
            # =================================================================
            logger.info("")
            logger.info("=" * 50)
            logger.info("🎤 STEP 2: New Voice - ElevenLabs Voice Changer")
            logger.info("=" * 50)
            
            audio_path = os.path.join(temp_dir, "original_audio.mp3")
            new_voice_url = None
            original_audio_url = None
            use_original_audio = False
            
            # Extract audio from original video
            logger.info("🌐 Extracting audio via Rendi.dev cloud...")
            original_audio_url = FFmpegProcessor.extract_audio_from_url(
                video_url=original_video_url,
                output_path=audio_path,
                rendi_api_key=config.RENDI_API_KEY
            )
            
            if original_audio_url:
                # Download the audio file locally for analysis
                try:
                    response = requests.get(original_audio_url, timeout=60)
                    response.raise_for_status()
                    with open(audio_path, 'wb') as f:
                        f.write(response.content)
                    logger.info("✅ Audio downloaded from cloud extraction")
                    
                    # Check if audio contains speech
                    has_speech = self.elevenlabs_service.detect_speech_in_audio(audio_path)
                    
                    if has_speech:
                        logger.info("🎤 Speech detected - applying ElevenLabs voice changer...")
                        new_voice_data = self.elevenlabs_service.voice_changer(audio_path)
                        
                        if new_voice_data:
                            # Upload changed voice to S3
                            voice_filename = f"new_voice_test_row_{row_num}_{int(time.time())}.mp3"
                            new_voice_url = self.s3_service.upload_audio_bytes(
                                audio_data=new_voice_data,
                                key_name=voice_filename
                            )
                            
                            if new_voice_url:
                                result["new_voice_url"] = new_voice_url
                                self.update_cell(row_num, config.NEW_VOICE_COLUMN, new_voice_url, headers)
                                logger.info(f"✅ New Voice: {new_voice_url[:60]}...")
                        else:
                            logger.warning("⚠️ Voice changer returned no data, using original audio")
                            use_original_audio = True
                    else:
                        logger.info("🎵 No speech detected - using original audio without voice change")
                        use_original_audio = True
                        
                except Exception as e:
                    logger.error(f"❌ Failed to process audio: {e}")
                    result["errors"].append(f"Audio processing error: {e}")
            
            # =================================================================
            # STEP 3: RENDI Scene & Voice - Combine video + audio
            # =================================================================
            logger.info("")
            logger.info("=" * 50)
            logger.info("🎬🎤 STEP 3: RENDI Scene & Voice - Combine")
            logger.info("=" * 50)
            
            if result["rendi_scene_url"]:
                # Determine which audio to use
                audio_url_to_use = None
                if new_voice_url:
                    audio_url_to_use = new_voice_url
                    logger.info("🎤 Using changed voice")
                elif use_original_audio and original_audio_url:
                    audio_url_to_use = original_audio_url
                    logger.info("🎵 Using original audio")
                
                if audio_url_to_use:
                    logger.info("🔗 Combining video with audio...")
                    final_video_with_voice = self.rendi_service.add_audio_to_video(
                        video_url=result["rendi_scene_url"],
                        audio_url=audio_url_to_use
                    )
                    
                    if final_video_with_voice:
                        result["rendi_scene_voice_url"] = final_video_with_voice
                        self.update_cell(row_num, config.RENDI_SCENE_VOICE_COLUMN, final_video_with_voice, headers)
                        logger.info(f"✅ RENDI Scene & Voice: {final_video_with_voice[:60]}...")
                    else:
                        result["errors"].append("Failed to combine video and audio")
                else:
                    logger.warning("⚠️ No audio available for combination")
            
            # =================================================================
            # STEP 4: Final Video - Upload to S3
            # =================================================================
            logger.info("")
            logger.info("=" * 50)
            logger.info("☁️ STEP 4: Final Video - Upload to S3")
            logger.info("=" * 50)
            
            # Use the best available video (with voice preferred)
            final_video_source = result["rendi_scene_voice_url"] or result["rendi_scene_url"]
            
            if final_video_source:
                logger.info("📤 Uploading final video to S3...")
                try:
                    # Upload to S3 directly from URL
                    final_filename = f"final_video_test_row_{row_num}_{int(time.time())}.mp4"
                    final_s3_url = self.s3_service.upload_video_from_url(
                        source_url=final_video_source,
                        key_name=final_filename
                    )
                    
                    if final_s3_url:
                        result["final_video_url"] = final_s3_url
                        result["success"] = True
                        self.update_cell(row_num, config.FINAL_VIDEO_COLUMN, final_s3_url, headers)
                        logger.info(f"✅ Final Video: {final_s3_url}")
                    else:
                        result["errors"].append("Failed to upload to S3")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to upload final video: {e}")
                    result["errors"].append(f"S3 upload error: {e}")
            else:
                result["errors"].append("No video available for final upload")
        
        return result


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test final stages of video processing pipeline"
    )
    parser.add_argument(
        "--row",
        type=int,
        default=TestConfig.DEFAULT_ROW,
        help="Row number to process (default: 2)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("🧪 FINAL STAGES TEST")
    logger.info("="*60)
    logger.info(f"   Processing row: {args.row}")
    logger.info("   Steps: RENDI Scene → New Voice → Combined → S3")
    logger.info("="*60)
    
    try:
        processor = FinalStagesProcessor()
        result = processor.process_row(args.row)
        
        logger.info("")
        logger.info("="*60)
        logger.info("📊 TEST RESULTS")
        logger.info("="*60)
        logger.info(json.dumps(result, indent=2, default=str))
        
        if result["success"]:
            logger.info("✅ Test completed successfully!")
        else:
            logger.warning("⚠️ Test completed with errors")
            
        return result
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()

