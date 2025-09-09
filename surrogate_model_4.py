# rom_pipeline_with_planck_and_validation.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft, ifft
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy.signal import windows
from pycbc.waveform import get_td_waveform, get_fd_waveform
import warnings

# ------------------------
# Planck taper implementation
# ------------------------
def planck_taper(N, eps_left=0.02, eps_right=0.02):
    """
    Return a Planck-taper window of length N.
    eps_left/right are fractions of the waveform length to taper on each side.
    Formula (piecewise) follows the common Planck-taper window definition.
    """
    x = np.linspace(0, 1, N)
    w = np.ones_like(x)

    el = eps_left
    er = eps_right

    def s_func(z, eps):
        # Avoid dividing by zero at boundaries: clip z
        z = np.clip(z, 1e-16, 1 - 1e-16)
        a = eps / z
        b = eps / (eps - z)
        return 1.0 / (np.exp(a) + np.exp(b))

    # left region
    if el > 0:
        maskL = x < el
        if np.any(maskL):
            zL = x[maskL] / el
            w[maskL] = s_func(zL, 1.0)

    # right region
    if er > 0:
        maskR = x > (1 - er)
        if np.any(maskR):
            zR = (1 - x[maskR]) / er
            w[maskR] = s_func(zR, 1.0)

    # middle is 1
    return w

# ------------------------
# Hybridization helper (TaylorF2 low-f replacement)
# ------------------------
def hybridize_td_with_taylorf2(td_hp, delta_t, f_lower, f_hyb=30.0, f_blend_width=5.0,
                               approximant_td='SEOBNRv4_opt'):
    """
    Hybridize the time-domain waveform with a TaylorF2 frequency-domain approximant
    at low frequency. Returns a new time series (hp) long enough and smoothly blended.

    This is a simple hybridization:
      - obtain FD TaylorF2 (via get_fd_waveform)
      - transform TD to FD
      - below f_hyb - f_blend_width/2 use TaylorF2
      - above f_hyb + f_blend_width/2 use TD-FFT
      - cross-fade in the window
    """
    # Generate FD TaylorF2 from pycbc if available
    try:
        fd_hp_taylor, _ = get_fd_waveform(approximant='TaylorF2',
                                          mass1=1., mass2=1.,  # placeholder masses; we will rescale magnitude only
                                          delta_f=1.0 / (len(td_hp) * delta_t),
                                          f_lower=f_lower)
        fd_hp_taylor = fd_hp_taylor.numpy()
    except Exception as e:
        warnings.warn(f"Could not obtain TaylorF2 FD via pycbc.get_fd_waveform: {e}. Skipping hybridization.")
        return td_hp  # fallback: return original TD

    # FFT the TD waveform (we will double-pad later or assume padding handled outside)
    # This helper expects caller to pad appropriately. Return original if mismatch occurs.
    return td_hp  # For now we keep hybridization as optional stub (detailed hybridization is problem-specific).

# ------------------------
# Frequency-domain generator with Planck taper + doubling & optional hybridization
# ------------------------
def generate_fd_waveform_planck(params, f_lower, delta_t, nfft, planck_eps=0.03, hybridize=False):
    """
    Generate time-domain waveform (pycbc.get_td_waveform), apply Planck taper,
    double-pad (pad to 2*len) and then FFT to get frequency domain samples.
    Returns: freqs, h_fd (full rfft)
    """
    q = params['q']
    chi = params['chi']
    M_total = 40.0
    m2 = M_total / (1 + q)
    m1 = q * M_total / (1 + q)

    try:
        hp, _ = get_td_waveform(approximant='SEOBNRv4_opt',
                                mass1=m1, mass2=m2,
                                spin1z=chi, spin2z=chi,
                                delta_t=delta_t,
                                f_lower=f_lower)
        h_td_raw = hp.numpy()
    except Exception as e:
        print(f"Could not generate waveform for q={q}, chi={chi}: {e}")
        return None, None

    # Apply Planck taper to the actual signal portion
    w = planck_taper(len(h_td_raw), eps_left=planck_eps, eps_right=planck_eps)
    h_td_windowed = h_td_raw * w

    # Optional hybridization (left as a conservative stub; real hybridization is problem-specific)
    if hybridize:
        h_td_windowed = hybridize_td_with_taylorf2(h_td_windowed, delta_t, f_lower)

    # Double-pad: pad to next power of two >= 2*len(h_td_windowed) or use 'nfft' if provided
    desired_n = max(nfft, 2 * len(h_td_windowed))
    # round up to next power of two for FFT efficiency
    nextpow2 = 1 << (desired_n - 1).bit_length()
    pad_len = nextpow2 - len(h_td_windowed)
    h_td = np.pad(h_td_windowed, (0, pad_len))

    freqs = rfftfreq(len(h_td), delta_t)
    h_fd = rfft(h_td)
    return freqs, h_fd

# ------------------------
# Phase & amplitude extraction (with linear detrend + anchor)
# ------------------------
def get_amp_phase(freqs, h_fd):
    amp = np.abs(h_fd)
    phase = np.unwrap(np.angle(h_fd))

    # linear detrend
    if len(freqs) > 1:
        poly_fit = np.polyfit(freqs, phase, 1)
        linear_phase = np.polyval(poly_fit, freqs)
        phase_centered = phase - linear_phase
    else:
        phase_centered = phase

    # anchor so first point is zero
    if len(phase_centered) > 0:
        phase_anchored = phase_centered - phase_centered[0]
    else:
        phase_anchored = phase_centered

    return amp, phase_anchored

# ------------------------
# CSE grid generator with prefactor
# ------------------------
def generate_cse_grid(f_min, f_max, num_points, power, prefactor):
    """
    Generate a CSE-like sparse frequency grid with spacing Delta(f) = prefactor * f^power.
    We create nodes by inverting the cumulative integral of density rho(f) = 1/Delta(f).
    """
    if num_points < 2:
        return np.array([f_min])

    p = power
    pref = prefactor

    if p == 1.0:
        # density rho(f) = 1/(pref * f) => integral is (1/pref) * log(f)
        S_min = np.log(f_min) / pref
        S_max = np.log(f_max) / pref
        S_vals = np.linspace(S_min, S_max, num_points)
        fs = np.exp(pref * S_vals)
        return fs

    # p != 1
    # S(f) = ∫_{f_min}^{f} 1/(pref * f'^p) df' = (1/(pref*(1-p))) * (f^{1-p} - f_min^{1-p})
    S_min = 0.0
    S_max = (1.0 / (pref * (1.0 - p))) * (f_max**(1.0 - p) - f_min**(1.0 - p))
    S_vals = np.linspace(S_min, S_max, num_points)
    # invert:
    fs = (S_vals * pref * (1.0 - p) + f_min**(1.0 - p)) ** (1.0 / (1.0 - p))
    fs[-1] = f_max
    return fs

# ------------------------
# SVD rank selector based on singular values energy
# ------------------------
def select_rank_from_svals(svals, energy_tol=1e-6):
    """
    Choose the minimum k such that cumulative energy retained >= 1 - energy_tol.
    energy_tol ~ desired leftover energy after truncation.
    Returns integer rank.
    """
    s2 = svals ** 2
    cumulative = np.cumsum(s2)
    total = cumulative[-1]
    frac_retained = cumulative / total
    k = np.searchsorted(frac_retained, 1.0 - energy_tol) + 1
    # safety: at least 1 and at most len(svals)
    k = max(1, min(k, len(svals)))
    return k

# ------------------------
# Mismatch / match computation (PSD-weighted, maximized over time and phase)
# ------------------------
def get_psd(freqs, delta_f, psd_type='aLIGOZeroDetHighPower'):
    """
    Try to get an aLIGO PSD from pycbc; if not available, return flat PSD=1.
    """
    try:
        from pycbc.psd import aLIGOZeroDetHighPower
        # pycbc's aLIGOZeroDetHighPower accepts delta_f and length typically; here sample PSD at freqs
        psd = aLIGOZeroDetHighPower(len(freqs), delta_f, freqs[0])
        # psd is array length len(freqs)
        return psd
    except Exception:
        # Fallback: use 1 for all frequencies
        return np.ones_like(freqs)

def compute_mismatch(freqs, h1, h2, psd=None):
    """
    Compute mismatch = 1 - max_{t0,phi0} overlap(h1,h2).
    Inputs:
      freqs: array of frequencies (rfft sample frequencies)
      h1, h2: complex FD arrays sampled at these freqs
      psd: PSD array same length as freqs (if None, will use get_psd fallback)
    """
    # Use only positive freq bins where PSD defined
    df = freqs[1] - freqs[0]
    if psd is None:
        psd = get_psd(freqs, df)

    # inner product <a|b> = 4 Re sum (a*(f) b(f) / S_n(f)) df
    # To get time-maximization, compute cross-correlation via IFFT of product:
    # z(f) = 4 * a*(f) b(f) / S_n(f)
    z = 4.0 * np.conjugate(h1) * h2 / psd
    # zero-pad z to symmetric (invert rfft -> ifft expects full complex spectrum), but we can do ifft of the
    # Hermitian-symmetric time-series implicitly by creating full complex array for ifft:
    # build a full-spectrum array for ifft (length N) using inverse rfft trick
    N = (len(freqs) - 1) * 2
    # create array of length N with rfft bins z
    # construct full spectrum Z_full of length N where rfft(Z_full) == z. We can use irfft to go back to time domain:
    corr_t = ifft(np.concatenate([z, np.conjugate(z[-2:0:-1])]))  # produce time-domain correlation (complex)
    # Norms:
    norm1_sq = 4.0 * np.sum(np.abs(h1) ** 2 / psd) * df
    norm2_sq = 4.0 * np.sum(np.abs(h2) ** 2 / psd) * df
    norm1 = np.sqrt(np.real(norm1_sq))
    norm2 = np.sqrt(np.real(norm2_sq))
    if norm1 == 0 or norm2 == 0:
        return 1.0  # maximal mismatch if one waveform is zero

    # maximize overlap over time by taking max absolute of corr_t (phase maximized by absolute value)
    max_corr = np.max(np.abs(corr_t))
    overlap = max_corr / (norm1 * norm2)
    overlap = float(np.clip(overlap, 0.0, 1.0))
    mismatch = 1.0 - overlap
    return mismatch

# ------------------------
# Full pipeline: generate training set, build ROM, validate
# ------------------------
def build_and_validate_rom(do_plots=True, energy_tol=1e-6, hybridize=False):
    # Parameter space grid
    q_vals = np.linspace(1, 10, 30)
    chi_vals = np.linspace(-0.8, 0.6, 30)
    param_grid_q, param_grid_chi = np.meshgrid(q_vals, chi_vals)
    params_list = [{'q': q, 'chi': chi} for q, chi in zip(param_grid_q.flatten(), param_grid_chi.flatten())]

    f_lower = 20.0
    f_min_grid = 25.0
    f_max_grid = 1024.0
    delta_t = 1/4096
    nfft = 16 * 4096

    raw_amps = []
    raw_phases = []
    raw_freqs = []
    amp_norms = []
    valid_params = []

    print("Generating training waveforms with Planck taper and doubled padding...")
    for params in params_list:
        freqs, h_fd = generate_fd_waveform_planck(params, f_lower, delta_t, nfft, planck_eps=0.03, hybridize=hybridize)
        if freqs is None:
            continue
        mask = (freqs >= f_min_grid) & (freqs <= f_max_grid)
        freqs_masked = freqs[mask]
        if freqs_masked.size == 0:
            continue

        amp, phase = get_amp_phase(freqs_masked, h_fd[mask])

        norm = np.linalg.norm(amp)
        if norm <= 0:
            continue
        raw_amps.append(amp / norm)
        amp_norms.append(norm)
        raw_phases.append(phase)
        raw_freqs.append(freqs_masked)
        valid_params.append(params)

    print(f"Generated {len(raw_amps)} valid training waveforms.")

    # CSE grids using paper prefactors:
    # amplitude spacing prefactor: 0.1 (Delta_A(f) = 0.1 * f^1)
    # phase spacing prefactor: 0.304 (Delta_phi(f) = 0.304 * f^(4/3))
    sparse_freq_amp = generate_cse_grid(f_min_grid, f_max_grid, num_points=100, power=1.0, prefactor=0.1)
    sparse_freq_phase = generate_cse_grid(f_min_grid, f_max_grid, num_points=200, power=4.0/3.0, prefactor=0.304)

    # Interpolate onto grids
    A_mat = np.zeros((len(sparse_freq_amp), len(raw_amps)))
    Phi_mat = np.zeros((len(sparse_freq_phase), len(raw_phases)))
    for i, (amp, phase, freqs) in enumerate(zip(raw_amps, raw_phases, raw_freqs)):
        spline_amp = UnivariateSpline(freqs, amp, s=0, k=3, ext='raise')
        spline_phase = UnivariateSpline(freqs, phase, s=0, k=3, ext='raise')
        A_mat[:, i] = spline_amp(sparse_freq_amp)
        Phi_mat[:, i] = spline_phase(sparse_freq_phase)

    print("Performing SVD...")
    Ua, sa, Vta = np.linalg.svd(A_mat, full_matrices=False)
    Up, sp, Vtp = np.linalg.svd(Phi_mat, full_matrices=False)

    # Choose ranks based on singular values energy retention
    rank_a = select_rank_from_svals(sa, energy_tol=energy_tol)
    rank_p = select_rank_from_svals(sp, energy_tol=energy_tol)
    print(f"Selected ranks: amplitude rank = {rank_a}, phase rank = {rank_p}")

    B_a = Ua[:, :rank_a]
    B_p = Up[:, :rank_p]

    Ca = B_a.T @ A_mat
    Cp = B_p.T @ Phi_mat

    # Interpolation grid
    q_unique = np.unique(param_grid_q)
    chi_unique = np.unique(param_grid_chi)
    # pack amp_norms
    amp_norms_grid = np.array(amp_norms).reshape(len(chi_unique), len(q_unique))

    # Create interpolants
    print("Building coefficient interpolants (RectBivariateSpline)...")
    interpolants_a = []
    for i in range(rank_a):
        coeff_grid = Ca[i, :].reshape(len(chi_unique), len(q_unique))
        interp = RectBivariateSpline(chi_unique, q_unique, coeff_grid, kx=3, ky=3)
        interpolants_a.append(interp)

    interpolants_p = []
    for i in range(rank_p):
        coeff_grid = Cp[i, :].reshape(len(chi_unique), len(q_unique))
        interp = RectBivariateSpline(chi_unique, q_unique, coeff_grid, kx=3, ky=3)
        interpolants_p.append(interp)

    interp_amp_norm = RectBivariateSpline(chi_unique, q_unique, amp_norms_grid, kx=3, ky=3)

    # Surrogate evaluator
    def evaluate_surrogate_fd(q_star, chi_star, freqs_out):
        ca_star = np.array([interp(chi_star, q_star)[0, 0] for interp in interpolants_a])
        cp_star = np.array([interp(chi_star, q_star)[0, 0] for interp in interpolants_p])

        amp_recon_sparse = B_a @ ca_star
        phase_recon_sparse = B_p @ cp_star

        spline_amp = UnivariateSpline(sparse_freq_amp, amp_recon_sparse, s=0, k=3, ext='raise')
        spline_phase = UnivariateSpline(sparse_freq_phase, phase_recon_sparse, s=0, k=3, ext='raise')
        amp_final = spline_amp(freqs_out)
        phase_final = spline_phase(freqs_out)

        norm_star = interp_amp_norm(chi_star, q_star)[0, 0]
        amp_final *= norm_star
        h_fd_recon = amp_final * np.exp(1j * phase_final)
        return freqs_out, h_fd_recon

    # Diagnostics plots if requested
    if do_plots:
        # singular values
        plt.figure(figsize=(8,5))
        plt.semilogy(sa / sa[0], label='amp svals')
        plt.semilogy(sp / sp[0], label='phase svals')
        plt.xlabel('mode index')
        plt.ylabel('normalized singular value')
        plt.legend()
        plt.title('Singular values (normalized)')
        plt.grid(True)
        plt.show()

        # cumulative energy
        plt.figure(figsize=(8,5))
        plt.plot(np.cumsum(sa**2) / np.sum(sa**2), label='amp energy')
        plt.plot(np.cumsum(sp**2) / np.sum(sp**2), label='phase energy')
        plt.xlabel('mode index')
        plt.ylabel('cumulative energy')
        plt.legend()
        plt.title('Cumulative energy of singular values')
        plt.grid(True)
        plt.show()

    # Validation: pick a test point off-grid
    test_params = {'q': 8.23, 'chi': -0.5}
    true_freqs, true_h_fd = generate_fd_waveform_planck(test_params, f_lower, delta_t, nfft, planck_eps=0.03, hybridize=hybridize)
    mask = (true_freqs >= f_min_grid) & (true_freqs <= f_max_grid)
    freqs_masked = true_freqs[mask]
    true_h_fd_masked = true_h_fd[mask]

    surr_freqs, surr_h_fd = evaluate_surrogate_fd(test_params['q'], test_params['chi'], freqs_masked)

    # compute mismatch
    df = freqs_masked[1] - freqs_masked[0]
    try:
        psd = get_psd(freqs_masked, df)
    except Exception:
        psd = np.ones_like(freqs_masked)

    mismatch_val = compute_mismatch(freqs_masked, true_h_fd_masked, surr_h_fd, psd=psd)
    print(f"Validation mismatch (max over time & phase) = {mismatch_val:.3e}")

    if do_plots:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Surrogate Model Validation for q={test_params['q']}, χ={test_params['chi']}", fontsize=16)

        true_amp = np.abs(true_h_fd_masked)
        true_phase = np.unwrap(np.angle(true_h_fd_masked))
        surr_amp = np.abs(surr_h_fd)
        surr_phase = np.unwrap(np.angle(surr_h_fd))

        axs[0].loglog(freqs_masked, true_amp, label='True Waveform', lw=3, alpha=0.8)
        axs[0].loglog(surr_freqs, surr_amp, '--', label='Surrogate Model', lw=2, color='red')
        axs[0].set_ylabel('Amplitude', fontsize=12)
        axs[0].legend(fontsize=11)
        axs[0].set_title('Amplitude Comparison', fontsize=14)
        axs[0].grid(True, which="both", ls="--")

        axs[1].semilogx(freqs_masked, true_phase, label='True Waveform (centered)', lw=3, alpha=0.8)
        axs[1].semilogx(surr_freqs, surr_phase, '--', label='Surrogate Model (centered)', lw=2, color='red')
        axs[1].set_xlabel('Frequency (Hz)', fontsize=12)
        axs[1].set_ylabel('Phase (rad)', fontsize=12)
        axs[1].legend(fontsize=11)
        axs[1].set_title('Phase Comparison (Linearly Centered)', fontsize=14)
        axs[1].grid(True, which="both", ls="--")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # Return objects for further inspection if caller wants them
    return {
        'B_a': B_a, 'B_p': B_p,
        'Ua': Ua, 'Up': Up, 'sa': sa, 'sp': sp,
        'interpolants_a': interpolants_a, 'interpolants_p': interpolants_p,
        'interp_amp_norm': interp_amp_norm,
        'sparse_freq_amp': sparse_freq_amp, 'sparse_freq_phase': sparse_freq_phase,
        'mismatch': mismatch_val
    }

# ------------------------
# If run as script, execute pipeline
# ------------------------
if __name__ == "__main__":
    res = build_and_validate_rom(do_plots=True, energy_tol=1e-6, hybridize=False)
    print("Done. Returned result keys:", list(res.keys()))
