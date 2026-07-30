import json
import subprocess

import numpy as np

from video_to_spider.video import FFmpegVideoWriter


def test_ffmpeg_writer_exports_browser_compatible_h264(tmp_path):
    output = tmp_path / "diagnostic.mp4"
    writer = FFmpegVideoWriter(output, 7.5, (65, 49))
    for index in range(4):
        frame = np.full((49, 65, 3), index * 50, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_frames",
            "-of", "json", str(output),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert (stream["width"], stream["height"]) == (66, 50)
    assert int(stream["nb_frames"]) == 4


def test_ffmpeg_writer_preserves_existing_output_without_overwrite(tmp_path):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"existing")

    try:
        FFmpegVideoWriter(output, 5.0, (64, 48))
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output should require overwrite=True")

    assert output.read_bytes() == b"existing"
