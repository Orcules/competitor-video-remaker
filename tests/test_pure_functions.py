"""Unit tests for pure, module-level helper functions in video_scene_processor.

Importing video_scene_processor has no side effects beyond load_dotenv(),
logging setup, and building a Config dataclass from environment defaults,
so it is safe to import directly here.
"""

import video_scene_processor as vsp


# ---------------------------------------------------------------------------
# is_valid_voice_id
# ---------------------------------------------------------------------------

class TestIsValidVoiceId:
    def test_accepts_realistic_voice_id(self):
        assert vsp.is_valid_voice_id("JBFqnCBsd6RMkjVDRZzb") is True

    def test_rejects_empty_string(self):
        assert vsp.is_valid_voice_id("") is False

    def test_rejects_none(self):
        assert vsp.is_valid_voice_id(None) is False

    def test_rejects_spreadsheet_error_values(self):
        for bad in ["#N/A", "#n/a", "n/a", "NA", "#REF!", "#ERROR!",
                    "#VALUE!", "null", "None", "undefined", "-"]:
            assert vsp.is_valid_voice_id(bad) is False, bad

    def test_rejects_too_short_ids(self):
        assert vsp.is_valid_voice_id("abc") is False

    def test_strips_whitespace_before_checking(self):
        assert vsp.is_valid_voice_id("  #N/A  ") is False


# ---------------------------------------------------------------------------
# get_validated_voice_id
# ---------------------------------------------------------------------------

class TestGetValidatedVoiceId:
    def test_returns_voice_id_when_valid(self):
        assert vsp.get_validated_voice_id("JBFqnCBsd6RMkjVDRZzb", "fallback-id") == "JBFqnCBsd6RMkjVDRZzb"

    def test_falls_back_to_default_when_invalid(self):
        assert vsp.get_validated_voice_id("#N/A", "fallback-id") == "fallback-id"

    def test_falls_back_to_config_default_when_no_default_given(self):
        assert vsp.get_validated_voice_id("") == vsp.config.DEFAULT_VOICE_ID


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------
# These cases are chosen so they pass both with the langdetect library
# installed and with the built-in character-set heuristic fallback.

class TestDetectLanguage:
    def test_empty_text_defaults_to_english(self):
        assert vsp.detect_language("") == "en"

    def test_none_defaults_to_english(self):
        assert vsp.detect_language(None) == "en"

    def test_short_text_defaults_to_english(self):
        assert vsp.detect_language("hi") == "en"

    def test_detects_english(self):
        text = (
            "This is a fairly long piece of English text that describes the "
            "weather today and mentions that the sun is shining brightly over "
            "the city while people walk through the park."
        )
        assert vsp.detect_language(text) == "en"

    def test_detects_hebrew(self):
        text = "שלום לכולם, זהו טקסט בעברית שנועד לבדוק את זיהוי השפה של המערכת"
        assert vsp.detect_language(text) == "he"


# ---------------------------------------------------------------------------
# Config environment handling
# ---------------------------------------------------------------------------

class TestConfigEnvDefaults:
    def test_sheet_tab_default(self):
        # GOOGLE_SHEET_TAB is not set in the test environment, so the
        # env-driven default captured at import time must be "Input".
        assert vsp.Config().GOOGLE_SHEET_TAB == "Input"
