"""Offline audio adapters with push-to-talk and wake-phrase hands-free mode.

Hands-free mode is opt-in and active only while the user runs /voice-loop.
It uses local Vosk recognition and system ALSA capture/playback devices.
Raw recordings are temporary and deleted after recognition.
"""
import json
import math
import os
import re
from array import array
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave


class AudioUnavailable(RuntimeError):
    """Audio hardware, executable, or speech model is unavailable."""


class AudioRuntime:
    def __init__(self, config):
        self.config = dict(config or {})
        self.sample_rate = int(self.config.get("audio_sample_rate", 16000))
        self.record_seconds = int(self.config.get("audio_record_seconds", 6))
        self.vosk_model_path = self.config.get("vosk_model_path", "")
        self.piper_model_path = self.config.get("piper_model_path", "")
        self.capture_device = self.config.get("audio_capture_device", "default")
        self.playback_device = self.config.get("audio_playback_device", "default")
        self.vad_threshold = int(self.config.get("audio_vad_rms_threshold", 450))
        self.silence_seconds = float(self.config.get("audio_vad_silence_seconds", 1.0))
        self.max_utterance_seconds = float(self.config.get("audio_max_utterance_seconds", 12))
        self.wake_phrase = str(self.config.get("audio_wake_phrase", "achillus")).strip().casefold()
        self._vosk_model = None

        if not 8000 <= self.sample_rate <= 48000:
            raise ValueError("audio_sample_rate must be between 8000 and 48000")
        if not 1 <= self.record_seconds <= 30:
            raise ValueError("audio_record_seconds must be between 1 and 30")
        if not 0 <= self.vad_threshold <= 32767:
            raise ValueError("audio_vad_rms_threshold must be between 0 and 32767")
        if not 0.25 <= self.silence_seconds <= 5:
            raise ValueError("audio_vad_silence_seconds must be between 0.25 and 5")
        if not 1 <= self.max_utterance_seconds <= 30:
            raise ValueError("audio_max_utterance_seconds must be between 1 and 30")
        if not self.wake_phrase:
            raise ValueError("audio_wake_phrase cannot be empty")

    def _get_vosk_model(self):
        model_path = Path(self.vosk_model_path).expanduser() if self.vosk_model_path else None
        if model_path is None or not model_path.is_dir():
            raise AudioUnavailable(
                "Vosk model not configured. Run setup_audio.sh or set vosk_model_path."
            )
        try:
            from vosk import Model
        except ImportError as exc:
            raise AudioUnavailable("Python package 'vosk' is missing; run setup_audio.sh") from exc
        if self._vosk_model is None:
            self._vosk_model = Model(str(model_path))
        return self._vosk_model

    @staticmethod
    def _rms(pcm_bytes):
        samples = array("h")
        samples.frombytes(pcm_bytes[:len(pcm_bytes) - (len(pcm_bytes) % 2)])
        if not samples:
            return 0
        # ALSA's S16_LE is little-endian; Raspberry Pi is little-endian.
        return int(math.sqrt(sum(sample * sample for sample in samples) / len(samples)))

    def _record_wav(self, destination):
        arecord = shutil.which("arecord")
        if not arecord:
            raise AudioUnavailable("arecord is missing; install ALSA utilities")
        command = [
            arecord, "-q", "-D", self.capture_device,
            "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1",
            "-d", str(self.record_seconds), "-t", "wav", str(destination),
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True,
                timeout=self.record_seconds + 8, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioUnavailable("Audio recording timed out") from exc
        if result.returncode != 0:
            detail = (result.stderr or "").strip()[-500:]
            raise AudioUnavailable(f"Audio recording failed: {detail or result.returncode}")

    def _record_vad_wav(self, destination):
        """Capture until speech ends; wait for speech without a fixed timer."""
        arecord = shutil.which("arecord")
        if not arecord:
            raise AudioUnavailable("arecord is missing; install ALSA utilities")
        chunk_seconds = 0.25
        chunk_samples = max(1, int(self.sample_rate * chunk_seconds))
        chunk_bytes = chunk_samples * 2
        silence_chunks_required = max(1, int(self.silence_seconds / chunk_seconds))
        max_chunks = max(1, int(self.max_utterance_seconds / chunk_seconds))
        preroll = []
        max_preroll_chunks = 2
        captured = []
        speaking = False
        silence_chunks = 0
        chunks_after_start = 0
        command = [
            arecord, "-q", "-D", self.capture_device,
            "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1",
            "-t", "raw",
        ]
        process = None
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            while True:
                chunk = process.stdout.read(chunk_bytes)
                if not chunk:
                    raise AudioUnavailable("Audio capture stream ended unexpectedly")
                level = self._rms(chunk)
                if not speaking:
                    preroll.append(chunk)
                    if len(preroll) > max_preroll_chunks:
                        preroll.pop(0)
                    if level >= self.vad_threshold:
                        speaking = True
                        captured.extend(preroll)
                        chunks_after_start = len(captured)
                        silence_chunks = 0
                else:
                    captured.append(chunk)
                    chunks_after_start += 1
                    if level < self.vad_threshold:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0
                    if silence_chunks >= silence_chunks_required:
                        # Drop the trailing silence from the stored utterance.
                        if silence_chunks:
                            del captured[-silence_chunks:]
                        break
                    if chunks_after_start >= max_chunks:
                        break
        except OSError as exc:
            raise AudioUnavailable(f"Could not read audio stream: {exc}") from exc
        finally:
            if process is not None:
                if process.stdout:
                    process.stdout.close()
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

        if not captured:
            raise AudioUnavailable("No speech activity detected")
        with wave.open(str(destination), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(self.sample_rate)
            audio.writeframes(b"".join(captured))

    def _transcribe_wav(self, wav_path):
        try:
            from vosk import KaldiRecognizer
        except ImportError as exc:
            raise AudioUnavailable("Python package 'vosk' is missing") from exc
        model = self._get_vosk_model()
        try:
            with wave.open(str(wav_path), "rb") as audio:
                if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                    raise AudioUnavailable("Capture must be mono, 16-bit PCM")
                recognizer = KaldiRecognizer(model, audio.getframerate())
                recognizer.SetWords(False)
                while True:
                    frames = audio.readframes(4000)
                    if not frames:
                        break
                    recognizer.AcceptWaveform(frames)
                result = json.loads(recognizer.FinalResult())
        except (wave.Error, json.JSONDecodeError) as exc:
            raise AudioUnavailable(f"Could not decode captured audio: {exc}") from exc
        text = result.get("text", "")
        return text.strip() if isinstance(text, str) else ""

    def listen_once(self):
        """Record one fixed-duration utterance and transcribe it locally."""
        self._get_vosk_model()
        with tempfile.TemporaryDirectory(prefix="achillus-audio-") as tmp:
            wav_path = Path(tmp) / "capture.wav"
            self._record_wav(wav_path)
            return self._transcribe_wav(wav_path)

    def listen_vad_once(self):
        """Record one utterance bounded by speech activity and silence."""
        self._get_vosk_model()
        with tempfile.TemporaryDirectory(prefix="achillus-audio-") as tmp:
            wav_path = Path(tmp) / "capture.wav"
            self._record_vad_wav(wav_path)
            return self._transcribe_wav(wav_path)

    def speak(self, text):
        """Speak text with Piper, falling back to espeak-ng if available."""
        if not isinstance(text, str) or not text.strip():
            return False
        aplay = shutil.which("aplay")
        if not aplay:
            raise AudioUnavailable("aplay is missing; install ALSA utilities")

        with tempfile.TemporaryDirectory(prefix="achillus-audio-") as tmp:
            wav_path = Path(tmp) / "speech.wav"
            piper = shutil.which("piper")
            if piper and self.piper_model_path and Path(self.piper_model_path).expanduser().is_file():
                try:
                    result = subprocess.run(
                        [piper, "--model", str(Path(self.piper_model_path).expanduser()),
                         "--output_file", str(wav_path)],
                        input=text, capture_output=True, text=True, timeout=60, check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise AudioUnavailable("Piper speech synthesis timed out") from exc
                if result.returncode != 0 or not wav_path.is_file():
                    raise AudioUnavailable("Piper could not synthesize speech")
            else:
                espeak = shutil.which("espeak-ng") or shutil.which("espeak")
                if not espeak:
                    raise AudioUnavailable(
                        "No configured Piper voice or espeak-ng fallback is available"
                    )
                try:
                    result = subprocess.run(
                        [espeak, "-w", str(wav_path), text],
                        capture_output=True, text=True, timeout=30, check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise AudioUnavailable("Speech synthesis timed out") from exc
                if result.returncode != 0 or not wav_path.is_file():
                    raise AudioUnavailable("Speech synthesis failed")

            command = [aplay, "-q", "-D", self.playback_device, str(wav_path)]
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=90, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise AudioUnavailable("Audio playback timed out") from exc
            if result.returncode != 0:
                detail = (result.stderr or "").strip()[-500:]
                raise AudioUnavailable(f"Audio playback failed: {detail or result.returncode}")
        return True

    def _strip_wake_phrase(self, transcript):
        """Return the request only if the transcript starts with the wake phrase."""
        text = transcript.strip()
        match = re.match(
            r"^" + re.escape(self.wake_phrase) + r"(?:[\s,.:;!?-]+(.*)|$)",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return (match.group(1) or "").strip()

    def converse_once(self, conversation_engine):
        """Listen once with explicit progress and failure feedback."""
        output = print
        output("Audio test: loading local speech model...", flush=True)

        try:
            output(
                f"Audio test: recording for {self.record_seconds} seconds "
                f"from device '{self.capture_device}'...",
                flush=True,
            )
            transcript = self.listen_once()
        except Exception as exc:
            output(
                f"Audio test FAILED during capture/transcription: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return {"transcript": "", "reply": "", "spoken": False}

        if not transcript:
            output(
                "Audio test: recording completed, but Vosk recognized no words.",
                flush=True,
            )
            return {"transcript": "", "reply": "", "spoken": False}

        output(f"Audio test: recognized: {transcript}", flush=True)
        output("Audio test: generating reply...", flush=True)

        try:
            reply = conversation_engine.respond(transcript)
            output(f"Achillus: {reply}", flush=True)
        except Exception as exc:
            output(
                f"Audio test FAILED during reply generation: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return {"transcript": transcript, "reply": "", "spoken": False}

        try:
            output("Audio test: speaking reply...", flush=True)
            spoken = self.speak(reply)
        except Exception as exc:
            output(
                f"Audio test FAILED during playback: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return {
                "transcript": transcript,
                "reply": reply,
                "spoken": False,
            }

        return {
            "transcript": transcript,
            "reply": reply,
            "spoken": spoken,
        }
    def converse_hands_free(self, conversation_engine, output_fn=print):
        """Keep listening until interrupted; only act on utterances prefixed by wake phrase."""
        output_fn(
            f"Hands-free mode active. Say '{self.wake_phrase}' followed by your request. "
            "Press Ctrl+C to stop."
        )
        handled = 0
        while True:
            transcript = self.listen_vad_once()
            if not transcript:
                continue
            request = self._strip_wake_phrase(transcript)
            if request is None:
                continue
            if not request:
                self.speak("I'm listening.")
                continue
            output_fn(f"You (voice): {transcript}")
            reply = conversation_engine.respond(request)
            output_fn(f"Achillus: {reply}")
            self.speak(reply)
            handled += 1
