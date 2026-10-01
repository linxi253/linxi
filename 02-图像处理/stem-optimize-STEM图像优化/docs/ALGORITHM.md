# Algorithm and interpretation

The v2.1 filter is an adaptive spectral denoiser, not a physical
deconvolution. A measured point-spread function or contrast-transfer function
is not supplied.

For each frame:

1. Convert finite numeric pixels to float32 (internally normalized by
   `max(|I|)` when magnitudes exceed the uint16 range so the float32 power
   spectrum cannot overflow; the result is scaled back linearly and the
   filter remains scale-invariant).
2. Estimate local texture using
   `sqrt(G(x²) - G(x)²)` at half resolution.
3. Reflect-pad to a fast FFT shape.
4. Compute the power spectrum.
5. Smooth it and form a rotationally averaged background power spectrum.
6. Estimate non-background signal power by non-negative subtraction.
7. Apply `S / (S + kB + epsilon)`, smooth the mask, and retain DC.
8. Inverse-transform using the real component.
9. Blend with the input using `local_texture × maximum_blend_ratio`.
10. Linearly map the stack-wide percentile range to uint8 or uint16.
11. Optionally apply CLAHE.

Consequences:

- `blend_ratio=0` is an exact identity before output normalization.
- `blend_ratio=1` is a maximum; smooth areas may receive less filtering.
- Rotational background estimation can attenuate isotropic rings or randomly
  oriented crystalline signal. Validate against domain-specific references.
- CLAHE destroys linear intensity comparability and is disabled by default.
- Percentile mapping can clip tails; the exact clipping fraction is reported.
- A constant uint8/uint16 stack uses its full native range instead of collapsing
  to an all-black output.

Recommended validation before publication includes synthetic Poisson noise,
known lattice spacings, peak-position preservation, FRC or equivalent
resolution measures, and comparison to an independently reviewed method.

On the 2026-07-29 Windows validation machine, the included benchmark processed
one synthetic 3072×3072 frame in 0.89 s with a 0.81 GiB peak working set. The
preflight estimate was 1.10 GiB and therefore remained conservative. Hardware
and image-dependent results will vary. Re-run with
`python -m tools.benchmark_filter --size 3072`.
