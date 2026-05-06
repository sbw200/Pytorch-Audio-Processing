from __future__ import annotations

import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio


_TORCHAUDIO_LOAD_AVAILABLE = True
_FFMPEG_CLI_AVAILABLE: bool | None = None


def _add_conda_ffmpeg_dll_directory() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    dll_dir = Path(conda_prefix) / "Library" / "bin"
    if dll_dir.exists():
        os.add_dll_directory(str(dll_dir))


def find_ffmpeg_executable() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        return ffmpeg_path

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "Library" / "bin" / "ffmpeg.exe"
        if candidate.exists():
            return str(candidate)

    return None


def ffmpeg_cli_available() -> bool:
    global _FFMPEG_CLI_AVAILABLE

    if _FFMPEG_CLI_AVAILABLE is not None:
        return _FFMPEG_CLI_AVAILABLE

    ffmpeg_path = find_ffmpeg_executable()
    if ffmpeg_path is None:
        _FFMPEG_CLI_AVAILABLE = False
        return _FFMPEG_CLI_AVAILABLE

    completed = subprocess.run(
        [ffmpeg_path, "-version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _FFMPEG_CLI_AVAILABLE = completed.returncode == 0
    return _FFMPEG_CLI_AVAILABLE


def get_wav_audio_format(audio_path: str | Path) -> int:
    path = Path(audio_path)
    with path.open("rb") as file:
        header = file.read(512)

    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise RuntimeError(f"Invalid RIFF/WAVE file: {path}")

    offset = 12
    while offset + 24 <= len(header):
        chunk_id = header[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", header, offset + 4)[0]
        chunk_start = offset + 8

        if chunk_id == b"fmt ":
            audio_format = struct.unpack_from("<H", header, chunk_start)[0]
            if audio_format == 65534 and chunk_size >= 26:
                return struct.unpack_from("<H", header, chunk_start + 24)[0]
            return audio_format

        offset = chunk_start + chunk_size + (chunk_size % 2)

    raise RuntimeError(f"Missing fmt chunk in WAV file '{path}'")


def audio_file_supported(audio_path: str | Path) -> bool:
    path = Path(audio_path)
    if path.suffix.lower() != ".wav":
        return True

    audio_format = get_wav_audio_format(path)
    if audio_format in {1, 3}:
        return True

    return ffmpeg_cli_available()


def load_waveform(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    """Load a WAV file as a waveform tensor with shape [channels, samples]."""
    global _TORCHAUDIO_LOAD_AVAILABLE

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Audio path is not a file: {path}")

    if not _TORCHAUDIO_LOAD_AVAILABLE and path.suffix.lower() == ".wav":
        return _load_wav_fallback(path)

    try:
        _add_conda_ffmpeg_dll_directory()
        return torchaudio.load(str(path))
    except ImportError as exc:
        if path.suffix.lower() == ".wav":
            _TORCHAUDIO_LOAD_AVAILABLE = False
            return _load_wav_fallback(path)
        raise ImportError(
            "torchaudio.load requires torchcodec with the installed torchaudio "
            "version for this file type. Run `uv pip install -e .` to install "
            "project dependencies."
        ) from None
    except RuntimeError as exc:
        if "Could not load libtorchcodec" in str(exc):
            if path.suffix.lower() == ".wav":
                _TORCHAUDIO_LOAD_AVAILABLE = False
                return _load_wav_fallback(path)
            raise RuntimeError(
                "torchaudio.load could not load TorchCodec's FFmpeg runtime. "
                "On Windows, install FFmpeg shared libraries, then rerun this script. "
                "Because Conda is active in your shell, try: "
                "`conda install -c conda-forge ffmpeg`."
            ) from None
        raise RuntimeError(f"Failed to load audio file '{path}': {exc}") from exc


def _load_wav_fallback(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    try:
        return load_wav_without_ffmpeg(audio_path)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from None


def load_wav_without_ffmpeg(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    """Load PCM or IEEE-float WAV without TorchCodec/FFmpeg."""
    path = Path(audio_path)

    with path.open("rb") as file:
        wav_bytes = file.read()

    audio_format, channels, sample_rate, bits_per_sample, audio_bytes = _read_wav_chunks(
        wav_bytes=wav_bytes,
        path=path,
    )

    if channels <= 0:
        raise RuntimeError(f"Invalid channel count in WAV file '{path}': {channels}")

    if audio_format == 1:
        samples = _pcm_bytes_to_float32(audio_bytes, bits_per_sample)
    elif audio_format == 3:
        samples = _float_bytes_to_float32(audio_bytes, bits_per_sample)
    else:
        return _load_wav_with_ffmpeg_cli(
            path=path,
            channels=channels,
            sample_rate=sample_rate,
            audio_format=audio_format,
        )

    if samples.size == 0:
        return torch.zeros((channels, 0), dtype=torch.float32), sample_rate

    if samples.size % channels != 0:
        raise RuntimeError(
            f"WAV data size is not divisible by channel count in '{path}'"
        )

    waveform = torch.from_numpy(samples.reshape(-1, channels).T.copy())
    return waveform, sample_rate


def _read_wav_chunks(
    wav_bytes: bytes,
    path: Path,
) -> tuple[int, int, int, int, bytes]:
    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise RuntimeError(f"Invalid RIFF/WAVE file: {path}")

    audio_format: int | None = None
    channels: int | None = None
    sample_rate: int | None = None
    bits_per_sample: int | None = None
    audio_data: bytes | None = None

    offset = 12
    while offset + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", wav_bytes, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size

        if chunk_end > len(wav_bytes):
            raise RuntimeError(f"Malformed WAV chunk in '{path}'")

        chunk_data = wav_bytes[chunk_start:chunk_end]
        if chunk_id == b"fmt ":
            if len(chunk_data) < 16:
                raise RuntimeError(f"Malformed fmt chunk in '{path}'")

            (
                audio_format,
                channels,
                sample_rate,
                _byte_rate,
                _block_align,
                bits_per_sample,
            ) = struct.unpack_from("<HHIIHH", chunk_data, 0)

            if audio_format == 65534 and len(chunk_data) >= 26:
                audio_format = struct.unpack_from("<H", chunk_data, 24)[0]
        elif chunk_id == b"data":
            audio_data = chunk_data

        offset = chunk_end + (chunk_size % 2)

    if (
        audio_format is None
        or channels is None
        or sample_rate is None
        or bits_per_sample is None
    ):
        raise RuntimeError(f"Missing fmt chunk in WAV file '{path}'")
    if audio_data is None:
        raise RuntimeError(f"Missing data chunk in WAV file '{path}'")

    return audio_format, channels, sample_rate, bits_per_sample, audio_data


def _pcm_bytes_to_float32(audio_bytes: bytes, bits_per_sample: int) -> np.ndarray:
    if bits_per_sample == 8:
        samples = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32)
        return (samples - 128.0) / 128.0

    if bits_per_sample == 16:
        samples = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32)
        return samples / 32768.0

    if bits_per_sample == 24:
        raw = np.frombuffer(audio_bytes, dtype=np.uint8).reshape(-1, 3)
        samples = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        samples = (samples ^ sign_bit) - sign_bit
        return samples.astype(np.float32) / 8388608.0

    if bits_per_sample == 32:
        samples = np.frombuffer(audio_bytes, dtype="<i4").astype(np.float32)
        return samples / 2147483648.0

    raise RuntimeError(f"Unsupported PCM WAV bit depth: {bits_per_sample}")


def _float_bytes_to_float32(audio_bytes: bytes, bits_per_sample: int) -> np.ndarray:
    if bits_per_sample == 32:
        return np.frombuffer(audio_bytes, dtype="<f4").astype(np.float32)
    if bits_per_sample == 64:
        return np.frombuffer(audio_bytes, dtype="<f8").astype(np.float32)

    raise RuntimeError(f"Unsupported float WAV bit depth: {bits_per_sample}")


def _load_wav_with_ffmpeg_cli(
    path: Path,
    channels: int,
    sample_rate: int,
    audio_format: int,
) -> tuple[torch.Tensor, int]:
    ffmpeg_path = find_ffmpeg_executable()

    if ffmpeg_path is None:
        raise RuntimeError(
            f"Unsupported WAV format {audio_format} in '{path}', and ffmpeg.exe "
            "was not found on PATH. Install FFmpeg or convert this file to PCM WAV."
        )

    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if completed.returncode != 0:
        error_message = completed.stderr.decode("utf-8", errors="replace").strip()
        if not error_message:
            error_message = f"ffmpeg exited with code {completed.returncode}"
        raise RuntimeError(f"ffmpeg failed to decode '{path}': {error_message}")

    samples = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32)
    if samples.size == 0:
        return torch.zeros((channels, 0), dtype=torch.float32), sample_rate
    if samples.size % channels != 0:
        raise RuntimeError(
            f"Decoded WAV data size is not divisible by channel count in '{path}'"
        )

    waveform = torch.from_numpy(samples.reshape(-1, channels).T.copy())
    return waveform, sample_rate


def stereo_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(
            f"Expected waveform with shape [channels, samples], got {tuple(waveform.shape)}"
        )
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample_waveform(
    waveform: torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    if original_sample_rate <= 0:
        raise ValueError(f"Invalid original sample rate: {original_sample_rate}")
    if target_sample_rate <= 0:
        raise ValueError(f"Invalid target sample rate: {target_sample_rate}")
    if original_sample_rate == target_sample_rate:
        return waveform

    resampler = torchaudio.transforms.Resample(
        orig_freq=original_sample_rate,
        new_freq=target_sample_rate,
    )
    return resampler(waveform)


def normalize_waveform(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak <= eps:
        return waveform
    return waveform / peak


def trim_or_pad_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    duration_seconds: float,
) -> torch.Tensor:
    if duration_seconds <= 0:
        raise ValueError(f"duration_seconds must be positive, got {duration_seconds}")

    target_samples = int(round(sample_rate * duration_seconds))
    current_samples = waveform.shape[-1]

    if current_samples > target_samples:
        return waveform[..., :target_samples]
    if current_samples < target_samples:
        pad_samples = target_samples - current_samples
        return torch.nn.functional.pad(waveform, (0, pad_samples))
    return waveform


def make_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64,
    power: float = 2.0,
    amplitude_to_db: bool = True,
) -> torch.Tensor:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )
    spectrogram = mel_transform(waveform)

    if amplitude_to_db:
        spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)

    return spectrogram


def preprocess_waveform(
    waveform: torch.Tensor,
    original_sample_rate: int,
    config: dict[str, Any],
) -> torch.Tensor:
    data_config = config.get("data", {})
    mel_config = config.get("mel", {})

    sample_rate = int(data_config.get("sample_rate", 16000))
    duration_seconds = float(data_config.get("duration_seconds", 4.0))

    waveform = stereo_to_mono(waveform)
    waveform = resample_waveform(waveform, original_sample_rate, sample_rate)

    if bool(data_config.get("normalize", True)):
        waveform = normalize_waveform(waveform)

    waveform = trim_or_pad_waveform(waveform, sample_rate, duration_seconds)

    return make_mel_spectrogram(
        waveform=waveform,
        sample_rate=sample_rate,
        n_fft=int(mel_config.get("n_fft", 1024)),
        hop_length=int(mel_config.get("hop_length", 512)),
        n_mels=int(mel_config.get("n_mels", 64)),
        power=float(mel_config.get("power", 2.0)),
        amplitude_to_db=bool(mel_config.get("amplitude_to_db", True)),
    )


def preprocess_audio_file(audio_path: str | Path, config: dict[str, Any]) -> torch.Tensor:
    waveform, sample_rate = load_waveform(audio_path)
    return preprocess_waveform(waveform, sample_rate, config)
