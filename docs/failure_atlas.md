# Failure Atlas

A living catalog of observed and anticipated failure modes. Each entry documents conditions, symptoms, and mitigation strategies.

---

## Template

**Status:** Planned
**Condition:** What triggers this failure.
**Symptom:** What the system does wrong.
**Detection:** How to identify it from logs or visualization.
**Mitigation:** Known or planned response.

---

## Low Texture

**Status:** Planned
**Condition:** Scene lacks gradient structure (blank walls, sky, uniform floors).
**Symptom:** Insufficient features tracked; frontend degeneracy.
**Detection:** Track count drops below threshold; reprojection variance spikes.
**Mitigation:** TBD.

## Motion Blur

**Status:** Planned
**Condition:** Fast camera motion or long exposure.
**Symptom:** Feature detection fails; tracked features drift.
**Detection:** Low feature response scores; tracking residuals increase.
**Mitigation:** TBD.

## Lighting Change

**Status:** Planned
**Condition:** Sudden illumination shift (entering/exiting room, HDR transition).
**Symptom:** Feature descriptors become inconsistent; place recognition fails.
**Detection:** Large photometric residual jumps.
**Mitigation:** TBD.

## Fast Rotation

**Status:** Planned
**Condition:** High angular velocity (> ~300 deg/s).
**Symptom:** IMU integration dominant; feature tracking fails; large parallax ambiguity.
**Detection:** IMU gyro saturation or aliasing; feature count collapse.
**Mitigation:** TBD.

## Timestamp Offset

**Status:** Planned
**Condition:** Camera and IMU timestamps not synchronized.
**Symptom:** IMU preintegration misaligned with image timestamps; inconsistent state.
**Detection:** Preintegration residuals biased in rotation direction of motion.
**Mitigation:** TBD.

## IMU Bias / Noise Corruption

**Status:** Planned
**Condition:** Accelerometer or gyro bias larger than modeled; sensor noise spike.
**Symptom:** Drift in gravity estimate; roll/pitch error; scale drift.
**Detection:** Gravity vector deviates from expected; large bias posterior.
**Mitigation:** TBD.

## Dropped Frames

**Status:** Planned
**Condition:** Camera frame drops due to I/O or processing overrun.
**Symptom:** Feature track gaps; spurious new track initialization; covariance jump.
**Detection:** Timestamp gaps in image stream.
**Mitigation:** TBD.

## Dynamic Objects

**Status:** Planned
**Condition:** Moving people, vehicles, or objects in scene.
**Symptom:** Moving features pass RANSAC; corrupt pose estimates.
**Detection:** Reprojection outliers localized to moving regions.
**Mitigation:** TBD.

## Low Parallax

**Status:** Planned
**Condition:** Camera barely translating (pure rotation or very slow motion).
**Symptom:** Triangulation depth is undefined or degenerate; scale unobservable.
**Detection:** Triangulated depth variance very high; depth estimates diverge.
**Mitigation:** TBD.

## Perceptual Aliasing in Loop Closure

**Status:** Planned
**Condition:** Visually similar but geometrically distinct places (long corridors, parking lots).
**Symptom:** False positive loop closure; catastrophic pose graph correction.
**Detection:** Geometric verification fails; loop correction inconsistent with odometry.
**Mitigation:** TBD.
