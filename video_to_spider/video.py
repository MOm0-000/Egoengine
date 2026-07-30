"""FFmpeg-backed video output shared by pipeline visualizations."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np


VIDEO_ENCODING = {
    "backend": "ffmpeg",
    "codec": "h264",
    "encoder": "libx264",
    "pixel_format": "yuv420p",
    "faststart": True,
}


class FFmpegVideoWriter:
    """Stream uint8 BGR frames to a broadly compatible H.264 MP4."""

    def __init__(
        self, path: str | Path, fps: float, size: tuple[int, int], *,
        overwrite: bool = False, crf: int = 18, preset: str = "fast",
    ) -> None:
        self.path = Path(path)
        self.fps = float(fps)
        self.width, self.height = (int(value) for value in size)
        if self.fps <= 0 or not np.isfinite(self.fps):
            raise ValueError(f"video FPS must be positive and finite, got {fps}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"video dimensions must be positive, got {size}")
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"output exists: {self.path}; pass --overwrite")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg executable not found on PATH")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary_path = self.path.with_name(
            f".{self.path.stem}.{uuid.uuid4().hex}.tmp{self.path.suffix}"
        )
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", f"{self.fps:.12g}", "-i", "pipe:0", "-an",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p", "-tag:v", "avc1", "-movflags", "+faststart",
            str(self._temporary_path),
        ]
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr,
            )
        except Exception:
            self._stderr.close()
            raise
        self._frame_count = 0
        self._closed = False

    def _error_output(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().decode("utf-8", errors="replace").strip()

    def _discard_temporary(self) -> None:
        self._temporary_path.unlink(missing_ok=True)

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed video writer")
        array = np.asarray(frame)
        expected_shape = (self.height, self.width, 3)
        if array.shape != expected_shape or array.dtype != np.uint8:
            raise ValueError(
                f"expected uint8 BGR frame with shape {expected_shape}, "
                f"got dtype={array.dtype} shape={array.shape}"
            )
        try:
            assert self._process.stdin is not None
            self._process.stdin.write(np.ascontiguousarray(array).tobytes())
        except BrokenPipeError as error:
            self._process.wait()
            details = self._error_output()
            self._closed = True
            self._discard_temporary()
            self._stderr.close()
            raise RuntimeError(f"ffmpeg stopped while writing {self.path}: {details}") from error
        self._frame_count += 1

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        assert self._process.stdin is not None
        self._process.stdin.close()
        return_code = self._process.wait()
        details = self._error_output()
        self._stderr.close()
        if return_code != 0 or self._frame_count == 0:
            self._discard_temporary()
            reason = details or "no frames were written"
            raise RuntimeError(f"ffmpeg failed to encode {self.path}: {reason}")
        os.replace(self._temporary_path, self.path)

    close = release

    def __enter__(self) -> FFmpegVideoWriter:
        return self

    def __exit__(self, error_type, error, traceback) -> None:
        if error_type is None:
            self.release()
            return
        if not self._closed:
            self._closed = True
            if self._process.stdin is not None:
                self._process.stdin.close()
            self._process.wait()
            self._stderr.close()
            self._discard_temporary()
