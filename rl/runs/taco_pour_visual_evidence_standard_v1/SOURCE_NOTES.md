# Visual evidence source notes: Pour 0–60

- The active overlap renderer previously wrote all 61 endpoint states to its
  videos but published only selected keyframes; missing published frames did
  not imply missing trajectory arrays.
- `video_to_spider.rl.core.action.contracts.validate_reference_timeline`
  defines `frame_indices[row]` as an actual decoded-video frame ID. The robot
  and human references contain equal explicit frame IDs and timestamps.
- The local RGB is the original episode member copied from the released local
  archive. `ffprobe` PTS and sequential decoding are checked independently.
- The simulator cameras are fixed views and have no established pixel
  registration to the egocentric RGB camera.
- The saved trajectory covers endpoint 0 through 60. Saved substeps and
  contact rows cover 60 control intervals at ten substeps per interval.
- Saved substep qpos is post-integration while contact wrenches belong to the
  corresponding recorded solve. The standardized default does not draw force
  arrows unless a solve-time prestate is explicitly paired.
- The repository README and ignore rules intentionally exclude raw TACO RGB.
  Complete RGB comparisons remain on the controlled server; repository
  previews show a visible controlled-data placeholder.
- An old `visual_review.json` cannot certify newly rendered pixels. New review
  status is bound to the new image-manifest hash.
