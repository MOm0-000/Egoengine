# TACO Brush/Brush/Bowl 20231006_145 acquisition v1

The complete sequence-specific official package is stored at:

`/data_all/zzx/3.2RL/data/taco_v1/brush_brush_bowl_20231006_145`

Status: `COMPLETE_SEQUENCE_SPECIFIC_MODALITY_ACQUISITION`

- 55 manifest records, 629,890,780 bytes;
- egocentric RGB;
- original and resized egocentric depth;
- complete left/right MANO pose and shape records plus independent 3D joints;
- brush `171` and bowl `180` poses and meshes;
- egocentric and 12-camera allocentric calibration;
- 12 segmentation arrays;
- 12 original allocentric videos;
- 12 marker-removed allocentric videos;
- MANO left/right model files;
- official Hugging Face derived preview.

All extracted ZIP members passed CRC checks, and all 55 manifest output hashes were
verified after acquisition. RGB, hand/object poses, camera extrinsics and all 24
allocentric videos contain 122 frames. The independent hand-joint archives agree
bitwise.

Official-source timebase differences are retained without modification: original
depth declares 15 fps while resized depth declares 30 fps, although both contain 122
frames. Each official segmentation array contains 25 time samples rather than 122.

See the data package's `acquisition_manifest.json`, `validation.json`, `summary.md`
and `server_artifacts.sha256` for provenance and integrity details.
