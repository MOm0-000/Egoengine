# Open3D adapter contract

For a single static group, the project calls the official `registration_icp` in the equivalence test. Production comparison cases need one transform shared across multiple independently posed frames, so they use a transparent grouping adapter:

1. Apply the current common transform to each frame's measurements.
2. Form nearest-neighbour pairs only against that frame's sampled public model surface.
3. Merge explicit pairs without permitting cross-frame or cross-object matches.
4. Pass the merged pairs to Open3D 0.20.0 `TransformationEstimationPointToPlane.compute_transformation`.
5. Left-compose the returned official increment and apply frozen convergence/safety checks.

The project does not implement the point-to-plane loss, Huber weight, Jacobian, normal equations, rigid update or rotation solver. The adapter's one-group output matches official `registration_icp` to `1e-10`, and all recorded cross-group correspondence counts are zero.
