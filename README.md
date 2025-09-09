# Surrogate Model Documentation

This document explains the step-by-step procedure of building a
surrogate model for gravitational waveforms, adapted from the
implementation in `surrogate_model_3.py`.

------------------------------------------------------------------------

## Overview

The surrogate model approximates expensive-to-compute gravitational
waveforms using reduced-order modeling techniques. The main steps are:

1.  **Generate training data** (waveforms at different parameter
    values).
2.  **Create sparse interpolation grids** for amplitude and phase.
3.  **Apply Singular Value Decomposition (SVD)** to construct reduced
    bases.
4.  **Interpolate projection coefficients** across parameter space.
5.  **Reconstruct surrogate waveforms** using interpolated coefficients
    and bases.

------------------------------------------------------------------------

## Step I: Generating Training Data

-   Waveforms are generated for a grid of intrinsic parameters (e.g.,
    masses, spins).
-   The data is saved as frequency-domain waveforms.
-   Only valid waveforms are kept after filtering.

------------------------------------------------------------------------

## Step II: Sparse Grids & Interpolation

-   Construct sparse grids in parameter space for efficiency.
-   Interpolate amplitude and phase on these grids using Chebyshev
    interpolation or equivalent.

------------------------------------------------------------------------

## Step III: Singular Value Decomposition (SVD)

-   Decompose the training waveform set into orthonormal bases.
-   Keep only the most significant modes to reduce dimensionality.
-   These bases capture the main features of amplitude and phase.

------------------------------------------------------------------------

## Step IV: Coefficient Interpolation

-   Projection coefficients (from waveforms to bases) are computed.
-   These coefficients are interpolated across parameter space using
    sparse interpolation methods.
-   This step enables fast evaluation for unseen parameters.

------------------------------------------------------------------------

## Step V: Surrogate Waveform Reconstruction

-   For a new parameter point:
    -   Interpolated coefficients are obtained.
    -   A surrogate waveform is reconstructed by combining coefficients
        with reduced bases.
-   Optional inverse FFT transforms frequency-domain surrogate back to
    time-domain.

------------------------------------------------------------------------

## Notes

-   Interpolation can be performed separately for **amplitude** and
    **phase** rather than directly for complex waveform values.
-   Reconstruction accuracy improves with more training waveforms but at
    the cost of runtime.
-   The surrogate model significantly reduces computational cost
    compared to generating waveforms directly.

------------------------------------------------------------------------

## Example Usage

``` python
# Generate surrogate waveform
params = {"mass1": 30, "mass2": 25, "spin1z": 0.1, "spin2z": 0.2}
surrogate_h = surrogate_model(params)

# Compare with actual waveform
plt.plot(t, actual_h, label="Actual")
plt.plot(t, surrogate_h, "--", label="Surrogate")
plt.legend()
plt.show()
```

------------------------------------------------------------------------

## References

-   Field, S., et al. *Reduced basis catalogs for gravitational wave
    templates.* Phys. Rev. Lett. 106, 221102 (2011).
-   Pürrer, M. *Frequency domain reduced order model of aligned-spin
    effective-one-body waveforms.* Phys. Rev. D 93, 064041 (2016).
