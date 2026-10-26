import unittest
from unittest.mock import patch

from audio_runtime import AudioRuntime, AudioUnavailable


class AudioRuntimeTests(unittest.TestCase):
    def test_rejects_invalid_sample_rate(self):
        with self.assertRaises(ValueError):
            AudioRuntime({"audio_sample_rate": 4000})

    def test_rejects_invalid_record_duration(self):
        with self.assertRaises(ValueError):
            AudioRuntime({"audio_record_seconds": 60})

    def test_rejects_invalid_vad_threshold(self):
        with self.assertRaises(ValueError):
            AudioRuntime({"audio_vad_rms_threshold": 40000})

    def test_rejects_invalid_silence_duration(self):
        with self.assertRaises(ValueError):
            AudioRuntime({"audio_vad_silence_seconds": 0.1})

    def test_missing_vosk_model_is_actionable_error(self):
        runtime = AudioRuntime({"vosk_model_path": "/path/that/does/not/exist"})
        with self.assertRaisesRegex(AudioUnavailable, "Vosk model not configured"):
            runtime.listen_once()

    def test_empty_speech_does_not_require_audio_device(self):
        runtime = AudioRuntime({})
        self.assertFalse(runtime.speak("  "))

    @patch("audio_runtime.shutil.which", return_value=None)
    def test_missing_aplay_is_actionable_error(self, _which):
        runtime = AudioRuntime({})
        with self.assertRaisesRegex(AudioUnavailable, "aplay is missing"):
            runtime.speak("hello")

    def test_default_devices_are_configurable(self):
        runtime = AudioRuntime({
            "audio_capture_device": "plughw:1,0",
            "audio_playback_device": "plughw:2,0",
        })
        self.assertEqual(runtime.capture_device, "plughw:1,0")
        self.assertEqual(runtime.playback_device, "plughw:2,0")

    def test_default_wake_phrase(self):
        self.assertEqual(AudioRuntime({}).wake_phrase, "achillus")

    def test_wake_phrase_is_required(self):
        with self.assertRaisesRegex(ValueError, "wake_phrase"):
            AudioRuntime({"audio_wake_phrase": " "})

    def test_wake_phrase_accepts_plain_request(self):
        runtime = AudioRuntime({"audio_wake_phrase": "achillus"})
        self.assertEqual(runtime._strip_wake_phrase("Achillus, hello there"), "hello there")

    def test_wake_phrase_alone_returns_empty_request(self):
        runtime = AudioRuntime({})
        self.assertEqual(runtime._strip_wake_phrase("Achillus!"), "")

    def test_ignores_utterance_without_wake_phrase(self):
        runtime = AudioRuntime({})
        self.assertIsNone(runtime._strip_wake_phrase("hello there"))

    def test_rms_is_zero_for_silence(self):
        self.assertEqual(AudioRuntime._rms(b"\x00\x00" * 100), 0)


if __name__ == "__main__":
    unittest.main()
