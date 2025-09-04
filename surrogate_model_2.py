import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from pycbc.waveform import get_td_waveform

# -----------------------------------------------------------------------------
# ## Setup and Helper Functions
# -----------------------------------------------------------------------------

def generate_fd_waveform(params, f_lower, delta_t, nfft):
    """Generates a time-domain waveform and converts it to the frequency domain."""
    q = params['q']
    chi = params['chi']
    # Mass ratio q = m1/m2, so m1 = q*m2. For a fixed total mass M_total = m1+m2:
    # m2 = M_total / (1+q) and m1 = q * M_total / (1+q)
    # The paper generates waveforms at a fixed total mass for the SVD step.
    # We'll use a representative total mass of 40 M_sun.
    M_total = 40.0
    m2 = M_total / (1 + q)
    m1 = q * M_total / (1 + q)

    # Generate the time-domain waveform using a model like SEOBNRv1 (here, we use a substitute)
    try:
        hp, _ = get_td_waveform(approximant='SEOBNRv4_opt', # A more modern EOB model available in PyCBC
                                  mass1=m1, mass2=m2,
                                  spin1z=chi, spin2z=chi, # Using equal spins as in the paper's single-spin model
                                  delta_t=delta_t,
                                  f_lower=f_lower)
        h_td = hp.numpy()
    except Exception as e:
        print(f"Could not generate waveform for q={q}, chi={chi}: {e}")
        return None, None

    # Zero-pad to a consistent length for FFT
    if len(h_td) < nfft:
        h_td = np.pad(h_td, (0, nfft - len(h_td)))
    else:
        h_td = h_td[:nfft]

    # Convert to frequency domain
    freqs = rfftfreq(nfft, delta_t)
    h_fd = rfft(h_td)
    return freqs, h_fd

def get_amp_phase(freqs, h_fd):
    """
    Extracts amplitude and unwrapped phase.
    It now performs a two-step centering:
    1. Removes a linear fit to detrend the phase.
    2. Anchors the phase to start at zero for numerical stability.
    """
    amp = np.abs(h_fd)
    phase = np.unwrap(np.angle(h_fd))

    # 1. Subtract a linear fit to remove arbitrary time/phase shifts
    if len(freqs) > 1:
        poly_fit = np.polyfit(freqs, phase, 1)
        linear_phase = np.polyval(poly_fit, freqs)
        phase_centered = phase - linear_phase
    else:
        phase_centered = phase # Cannot fit a line to a single point

    # 2. Anchor the centered phase to start at 0
    # This provides a consistent reference point for the SVD model.
    if len(phase_centered) > 0:
        phase_offset = phase_centered[0]
        phase_anchored = phase_centered - phase_offset
    else:
        phase_anchored = phase_centered

    return amp, phase_anchored

def generate_sparse_grid(f_min, f_max, num_points, power=4/3):
    """
    Implements a simplified version of the 'Constant Spline Error' (CSE) grid generation
    from Algorithm 1 and Section 5.1 of the paper. [cite: 324, 332, 345]
    The grid spacing Delta(f) is proportional to f^power.
    """
    # This is a practical implementation of the paper's method.
    # We generate points such that the integral of 1/Delta(f) is uniform.
    if power == 1:
        return np.geomspace(f_min, f_max, num_points)
    
    # Integrated spacing function
    def integrated_spacing(f):
        return f**(1-power)

    # Solve for the constant C to get the correct number of points
    C = (integrated_spacing(f_max) - integrated_spacing(f_min)) / (num_points - 1)

    grid = []
    f_current = f_min
    for _ in range(num_points):
        grid.append(f_current)
        if f_current >= f_max:
            break
        # Step size is Delta(f) ~ C * f^power
        f_next_integrated = integrated_spacing(f_current) + C
        f_current = f_next_integrated**(1 / (1-power))
        
    grid = np.array(grid)
    grid[-1] = f_max # Ensure the grid ends exactly at f_max
    return grid

# -----------------------------------------------------------------------------
# ## Step I: Generate and Pre-process Training Waveforms [cite: 143]
# -----------------------------------------------------------------------------
print("Step I: Generating training data...")

# Parameter space grid [cite: 145]
q_vals = np.linspace(1, 10, 15)
chi_vals = np.linspace(-0.8, 0.6, 15)
param_grid_q, param_grid_chi = np.meshgrid(q_vals, chi_vals)
params_list = [{'q': q, 'chi': chi} for q, chi in zip(param_grid_q.flatten(), param_grid_chi.flatten())]

# Waveform generation settings
f_lower = 20.0
f_min_grid = 25.0 # Start of frequency grid for model
f_max_grid = 1024.0
delta_t = 1/4096
nfft = 16 * 4096 # Use a longer FFT for better frequency resolution

# Store pre-processed data
raw_amps = []
raw_phases = []
raw_freqs = []
amp_norms = []
valid_params = []

for params in params_list:
    freqs, h_fd = generate_fd_waveform(params, f_lower, delta_t, nfft)
    if freqs is None: continue

    mask = (freqs >= f_min_grid) & (freqs <= f_max_grid)
    freqs_masked = freqs[mask]
    
    amp, phase = get_amp_phase(freqs_masked, h_fd[mask])
    
    # Normalize amplitude and store the norm 
    norm = np.linalg.norm(amp)
    raw_amps.append(amp / norm)
    amp_norms.append(norm)

    raw_phases.append(phase)
    raw_freqs.append(freqs_masked)
    valid_params.append(params)

print(f"Generated {len(raw_amps)} valid waveforms.")

# -----------------------------------------------------------------------------
# ## Step II: Define Sparse Frequency Grids and Interpolate [cite: 151]
# -----------------------------------------------------------------------------
print("Step II: Creating sparse grids and interpolating...")

# Generate separate sparse grids for amplitude and phase [cite: 152]
# Using the paper's suggestion of ~100-200 points.
sparse_freq_amp = generate_sparse_grid(f_min_grid, f_max_grid, num_points=100, power=1.3)
sparse_freq_phase = generate_sparse_grid(f_min_grid, f_max_grid, num_points=200, power=4/3)

# Interpolate each waveform onto the sparse grids 
A_mat = np.zeros((len(sparse_freq_amp), len(raw_amps)))
Phi_mat = np.zeros((len(sparse_freq_phase), len(raw_phases)))

for i, (amp, phase, freqs) in enumerate(zip(raw_amps, raw_phases, raw_freqs)):
    # Use cubic splines for interpolation
    spline_amp = UnivariateSpline(freqs, amp, s=0, k=3, ext='raise')
    spline_phase = UnivariateSpline(freqs, phase, s=0, k=3, ext='raise')
    A_mat[:, i] = spline_amp(sparse_freq_amp)
    Phi_mat[:, i] = spline_phase(sparse_freq_phase)

# -----------------------------------------------------------------------------
# ## Step III: Compute Reduced Bases via SVD [cite: 155]
# -----------------------------------------------------------------------------
print("Step III: Performing SVD to find reduced bases...")

Ua, sa, Vta = np.linalg.svd(A_mat, full_matrices=False)
Up, sp, Vtp = np.linalg.svd(Phi_mat, full_matrices=False)

# Truncate the basis. The paper notes this is an option for compression.
# Let's keep a fixed number of modes for simplicity, e.g., 20.
rank_a = 20
rank_p = 60
B_a = Ua[:, :rank_a]  # Amplitude basis matrix [cite: 401]
B_p = Up[:, :rank_p]  # Phase basis matrix

# -----------------------------------------------------------------------------
# ## Step IV: Interpolate Projection Coefficients [cite: 158]
# -----------------------------------------------------------------------------
print("Step IV: Interpolating projection coefficients...")

# Calculate projection coefficients for all input waveforms [cite: 159]
# M_ji = (B^T * T)_ji, where T is the training matrix (A_mat or Phi_mat)
Ca = B_a.T @ A_mat
Cp = B_p.T @ Phi_mat

# Create interpolants for the coefficients over the parameter space.
# The paper recommends tensor product splines. 
q_unique = np.unique(param_grid_q)
chi_unique = np.unique(param_grid_chi)

# Interpolate amplitude coefficients
interpolants_a = []
for i in range(rank_a):
    coeff_grid = Ca[i, :].reshape(len(chi_unique), len(q_unique))
    # RectBivariateSpline is a tensor product spline interpolator
    interp = RectBivariateSpline(chi_unique, q_unique, coeff_grid, kx=3, ky=3)
    interpolants_a.append(interp)

# Interpolate phase coefficients
interpolants_p = []
for i in range(rank_p):
    coeff_grid = Cp[i, :].reshape(len(chi_unique), len(q_unique))
    interp = RectBivariateSpline(chi_unique, q_unique, coeff_grid, kx=3, ky=3)
    interpolants_p.append(interp)

# Also create an interpolant for the amplitude normalization factors [cite: 161]
amp_norms_grid = np.array(amp_norms).reshape(len(chi_unique), len(q_unique))
interp_amp_norm = RectBivariateSpline(chi_unique, q_unique, amp_norms_grid, kx=3, ky=3)

# -----------------------------------------------------------------------------
# ## Step V: Assemble and Evaluate the Surrogate Model [cite: 162]
# -----------------------------------------------------------------------------
print("Step V: Assembling the surrogate model evaluator.")

def evaluate_surrogate_fd(q_star, chi_star, freqs_out):
    """
    Evaluates the surrogate model at a new parameter point (q*, chi*).
    This function implements Eq. (6.10) from the paper. 
    """
    # 1. Get projection coefficients from interpolants
    ca_star = np.array([interp(chi_star, q_star)[0, 0] for interp in interpolants_a])
    cp_star = np.array([interp(chi_star, q_star)[0, 0] for interp in interpolants_p])

    # 2. Reconstruct amplitude and phase on their sparse grids
    amp_recon_sparse = B_a @ ca_star
    phase_recon_sparse = B_p @ cp_star

    # 3. Interpolate from sparse grids to the desired output frequency grid (the I_f step)
    spline_amp = UnivariateSpline(sparse_freq_amp, amp_recon_sparse, s=0, k=3, ext='raise')
    spline_phase = UnivariateSpline(sparse_freq_phase, phase_recon_sparse, s=0, k=3, ext='raise')
    
    amp_final = spline_amp(freqs_out)
    phase_final = spline_phase(freqs_out)

    # 4. Restore amplitude normalization
    norm_star = interp_amp_norm(chi_star, q_star)[0, 0]
    amp_final *= norm_star

    # 5. Combine into a complex frequency-domain waveform
    h_fd_recon = amp_final * np.exp(1j * phase_final)
    
    return freqs_out, h_fd_recon

# -----------------------------------------------------------------------------
# ## Validation and Plotting
# -----------------------------------------------------------------------------
print("\nValidating model with a test waveform...")

# Pick a test parameter set NOT on the training grid
test_params = {'q': 4.5, 'chi': 0.45}

# Generate the "true" waveform for comparison
true_freqs, true_h_fd = generate_fd_waveform(test_params, f_lower, delta_t, nfft)
mask = (true_freqs >= f_min_grid) & (true_freqs <= f_max_grid)
true_freqs_masked = true_freqs[mask]
true_h_fd_masked = true_h_fd[mask]
true_amp, true_phase = get_amp_phase(true_freqs_masked, true_h_fd_masked)


# Generate the surrogate waveform on the same frequency grid
surr_freqs, surr_h_fd = evaluate_surrogate_fd(test_params['q'], test_params['chi'], true_freqs_masked)
surr_amp = np.abs(surr_h_fd)
surr_phase = np.unwrap(np.angle(surr_h_fd))

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle(f"Surrogate Model Validation for q={test_params['q']}, χ={test_params['chi']}", fontsize=16)

# Amplitude Plot
axs[0].loglog(true_freqs_masked, true_amp, label='True Waveform', lw=3, alpha=0.8)
axs[0].loglog(surr_freqs, surr_amp, '--', label='Surrogate Model', lw=2, color='red')
axs[0].set_ylabel('Amplitude', fontsize=12)
axs[0].legend(fontsize=11)
axs[0].set_title('Amplitude Comparison', fontsize=14)
axs[0].grid(True, which="both", ls="--")

# Phase Plot
axs[1].semilogx(true_freqs_masked, true_phase, label='True Waveform (centered)', lw=3, alpha=0.8)
axs[1].semilogx(surr_freqs, surr_phase, '--', label='Surrogate Model (centered)', lw=2, color='red')
axs[1].set_xlabel('Frequency (Hz)', fontsize=12)
axs[1].set_ylabel('Phase (rad)', fontsize=12)
axs[1].legend(fontsize=11)
axs[1].set_title('Phase Comparison (Linearly Centered)', fontsize=14)
axs[1].grid(True, which="both", ls="--")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Surrogate_Model_vs_True_Model.pdf')
plt.show()