# Changelog

## Unreleased

### Robustness

- Normalize large-magnitude float inputs by `max(|I|)` before the FFT so the
  float32 power spectrum can no longer overflow to inf. Previously finite
  inputs around 1e15 magnitude failed with a misleading
  "功率谱包含 NaN 或 Inf" error; they now produce the same scale-invariant
  result as unit-scale data, and regular uint16-range inputs keep their
  existing bit-identical numeric path.
- Release the cached FFT radial grids after previews as well, not only after
  exports; a preview of a large frame no longer keeps a
  hundreds-of-megabytes grid resident while idle.
- Close the preview/export consistency gap: the export worker now verifies
  the input signature captured when the file was opened, so an input modified
  during confirmation dialogs aborts the run instead of silently processing
  data the user never previewed.
- Probe output-directory writability with a real create/delete attempt before
  starting the transaction (Windows `os.access` is unreliable for this), and
  include `.probe` leftovers in stale temporary-file cleanup.
- Honor cancellation during the reader's page-structure validation loop, so
  "cancel open" takes effect immediately instead of after validation
  completes.
- Publish the validated TIFF with a single atomic rename instead of the
  backup-then-rename pair transaction. A crash can no longer leave the target
  path missing with the previous output recoverable only from a `.backup`
  file; a failed rename always leaves the old output untouched.
- Retry provenance sidecar publication twice and report an explicit
  "new TIFF published, sidecar not updated" state instead of rolling back the
  validated output when the sidecar cannot be replaced.
- Release the cached FFT radial grids after every processing run so large
  frames do not keep hundreds of megabytes resident between tasks.
- Move file opening (full page validation plus first-frame decode) to a
  cancellable background thread with a dedicated "opening" state, so large
  compressed stacks no longer freeze the UI while loading.
- Add a 60-second close timeout with an explicit force-quit confirmation so a
  worker stuck on slow storage cannot block application exit forever.
- Flush the output directory after atomic publication so renames survive an
  abrupt power loss where the filesystem supports it.
- Detect and offer to remove leftover `.partial`/`.backup` files from previous
  abnormal exits before starting a new task.
- Cancel the pending queue poller during shutdown instead of relying on Tk
  callback cancellation.
- Capture the input file signature before replacing the active reader so an
  open failure cannot leave the UI in a half-switched state.

### Correctness and metadata

- Record library versions (numpy/scipy/opencv/tifffile), the CLAHE tile grid,
  start time, and processing duration in the provenance sidecar for
  reproducibility.
- Report progress when a frame finishes instead of one frame late.
- Add low/high stack-range percentile controls to the parameter panel; the
  cached preview range is keyed to the percentiles that produced it, and the
  provenance sidecar records the exact values used.
- Record the actual global-range percentiles in provenance instead of
  hard-coded literals; preview and export now pass the same constants.
- Validate ImageJ numeric metadata fields (`min`, `max`, `ranges`, `loop`,
  `spacing`, `fps`, `finterval`) and drop invalid values instead of copying
  them into the output.
- Preserve the input ImageDescription for generic TIFF output.
- Use a direct integer uint16→uint8 mapping for full-range previews, avoiding
  a whole-frame float64 allocation; derive the display maximum from
  `np.iinfo` instead of `itemsize`.
- Distinguish float32 representation overflow from true NaN/Inf input in the
  filter error message so finite-but-huge inputs are reported accurately.
- Remove the no-op bit-depth callback in the parameter panel.

### Internal

- Annotate the preview status line with the active display window (per-frame
  adaptive vs. stack-wide range) so the implicit switch after the first
  preview is visible instead of surprising.
- Record menu-entry indices when building the menus instead of hard-coded
  positional numbers, so menu reordering cannot silently break state toggles.
- Catch `tk.TclError` alongside `ValueError` when reading parameters for
  export, matching the preview path.
- Add `[project]` metadata with bounded dependency ranges to
  `pyproject.toml`, with a test that keeps its version in sync with
  `version.APP_VERSION`; exact build pins remain in the lock file.
- Name the CLAHE tile grid constant (`CLAHE_TILE_GRID`) instead of repeating
  the literal in both CLAHE entry points.
- Share one finite-real parameter validator (`filters.validate_real`) between
  the filter and the pipeline instead of two near-duplicate helpers.
- Derive parameter-panel defaults from `pipeline.DEFAULT_PARAMS` instead of
  duplicating the values.
- Remove the unused `kilaas_filter` alias and
  `pipeline.output_to_display_uint8` helper.

## 2.1.0 — 2026-07-29

### Data safety

- Reject input/output identity before opening a writer.
- Write to same-directory temporary files and atomically replace only after validation.
- Preserve existing outputs on cancellation or any read/filter/write/validation error.
- Abort on the first failed frame; removed original/zero-frame fallback.
- Wait for worker cleanup on application exit.
- Re-hash the input after processing and abort if its contents changed.
- Flush files before publication and roll back TIFF/sidecar as a pair.
- Decode every completed output page before publication.

### Scientific integrity

- Rebuild OME/ImageJ metadata for output dtype, shape, and axes.
- Preserve TIFF resolution and physical OME calibration fields.
- Add SHA-256 and parameter provenance sidecars.
- Use exact full-stack histograms for uint8/uint16 ranges.
- Make preview and export share one processing pipeline and range.
- Disable CLAHE by default and add nonlinear-processing warnings.
- Report low/high clipping fractions.

### Algorithm

- Replace the mislabeled global-mean deviation with true local standard deviation.
- Use a power-spectrum Wiener-style attenuation model.
- Use the real component of inverse FFT instead of absolute-value rectification.
- Support tiny images and validate all numeric parameters.
- Limit radial-grid cache to two FFT shapes.
- Rename user-facing claims from deconvolution to adaptive spectral denoising.
- Keep constant integer stacks on their native range instead of mapping to black.

### Engineering

- Add 31 regression tests, Ruff configuration, security rules, exact runtime
  pins, and Windows wheel hashes.
- Add imagecodecs for compressed TIFF support.
- Remove unused Pillow/Matplotlib from the build.
- Add Windows version resources and a signing-ready release script.
- Replace unbounded logs with rotating logs.
- Move the unrelated slab generator to a safe parameterized CLI.
- Make the release audit UTF-8 safe in Chinese paths and fail on every native
  command error.
- Add an in-package LZW/OME/transaction runtime self-test to the release gate.
- Parse untrusted OME-XML with entity expansion and external resources disabled.
- Audit the complete build environment and replace vulnerable pip/setuptools
  bootstrap versions with patched exact pins.
