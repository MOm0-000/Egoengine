# License audit

Open3D is sourced from the official repository `https://github.com/isl-org/Open3D`, tag `v0.20.0`, commit `b6c5e196384ad71e75b6e6f9c5da22d046221f1d`, under the MIT License.

The independent CPU environment uses the official release asset `open3d_cpu-0.20.0-cp311-cp311-manylinux_2_35_x86_64.whl`, SHA-256 `5b7b6ecd566e12fc7d55b1de476b5b35461deb632b40216ad52894cf402aed90`. The 100 MB wheel is not committed to this repository.

The adapter calls `open3d.pipelines.registration.TransformationEstimationPointToPlane.compute_transformation`; the one-group equivalence test also calls the official `registration_icp`. Robust mode uses the official `HuberLoss(k=0.01)`.
