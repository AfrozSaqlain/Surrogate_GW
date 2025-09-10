import pycbc
import pickle
import pycbc.psd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.fft import rfft, rfftfreq
from pycbc.waveform import get_td_waveform
from scipy.interpolate import UnivariateSpline, RectBivariateSpline

# ----------------------------------------------------
# Load surrogate model
# ----------------------------------------------------
def load_surrogate(filename):
    """Load surrogate model and rebuild interpolants."""
    with open(filename, "rb") as f:
        data = pickle.load(f)

    interpolants_a = []
    for i in range(data["Ca"].shape[0]):
        coeff_grid = data["Ca"][i, :].reshape(len(data["chi_unique"]), len(data["q_unique"]))
        interp = RectBivariateSpline(data["chi_unique"], data["q_unique"], coeff_grid, kx=3, ky=3)
        interpolants_a.append(interp)

    interpolants_p = []
    for i in range(data["Cp"].shape[0]):
        coeff_grid = data["Cp"][i, :].reshape(len(data["chi_unique"]), len(data["q_unique"]))
        interp = RectBivariateSpline(data["chi_unique"], data["q_unique"], coeff_grid, kx=3, ky=3)
        interpolants_p.append(interp)

    interp_amp_norm = RectBivariateSpline(data["chi_unique"], data["q_unique"], data["amp_norms_grid"], kx=3, ky=3)

    return {
        "sparse_freq_amp": data["sparse_freq_amp"],
        "sparse_freq_phase": data["sparse_freq_phase"],
        "B_a": data["B_a"],
        "B_p": data["B_p"],
        "interpolants_a": interpolants_a,
        "interpolants_p": interpolants_p,
        "interp_amp_norm": interp_amp_norm
    }

# ----------------------------------------------------
# Surrogate evaluation function
# ----------------------------------------------------
def evaluate_surrogate_fd(q_star, chi_star, freqs_out, surrogate):
    ca_star = np.array([interp(chi_star, q_star)[0, 0] for interp in surrogate["interpolants_a"]])
    cp_star = np.array([interp(chi_star, q_star)[0, 0] for interp in surrogate["interpolants_p"]])

    amp_recon_sparse = surrogate["B_a"] @ ca_star
    phase_recon_sparse = surrogate["B_p"] @ cp_star

    spline_amp = UnivariateSpline(surrogate["sparse_freq_amp"], amp_recon_sparse, s=0, k=3, ext=3)
    spline_phase = UnivariateSpline(surrogate["sparse_freq_phase"], phase_recon_sparse, s=0, k=3, ext=3)

    amp_final = spline_amp(freqs_out)
    phase_final = spline_phase(freqs_out)

    norm_star = surrogate["interp_amp_norm"](chi_star, q_star)[0, 0]
    amp_final *= norm_star

    return freqs_out, amp_final * np.exp(1j * phase_final)


# ----------------------------------------------------
# Generate frequency-domain waveform using PyCBC
# ----------------------------------------------------
def generate_fd_waveform(params, f_lower, delta_t, nfft):
    """Generates a time-domain waveform, applies a window, and converts to frequency domain."""
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

    tukey_window = windows.tukey(len(h_td_raw), alpha=0.1)
    h_td_windowed = h_td_raw * tukey_window

    if len(h_td_windowed) < nfft:
        h_td = np.pad(h_td_windowed, (0, nfft - len(h_td_windowed)))
    else:
        h_td = h_td_windowed[:nfft]

    freqs = rfftfreq(nfft, delta_t)
    h_fd = rfft(h_td)
    return freqs, h_fd

# ----------------------------------------------------
# Load trained surrogate
# ----------------------------------------------------
surrogate = load_surrogate("Models/surrogate_model.pkl")
print("Surrogate model loaded successfully!")

# ----------------------------------------------------
# Setup
# ----------------------------------------------------
f_lower = 20.0
f_min_grid = 25.0
f_max_grid = 1024.0
delta_t = 1/4096
nfft = 16 * 4096

total_mass = 40
q_vals = np.linspace(1, 10, 100)
chi_vals = np.linspace(-0.8, 0.6, 100)

mismatch_vals = np.zeros((len(chi_vals), len(q_vals)))

for iq, q in enumerate(q_vals):
    for ichi, chi in enumerate(chi_vals):
        params = {'q': q, 'chi': chi}

        true_freqs, true_h_fd = generate_fd_waveform(params, f_lower, delta_t, nfft)
        if true_freqs is None:
            mismatch_vals[ichi, iq] = np.nan
            continue

        mask = (true_freqs >= f_min_grid) & (true_freqs <= f_max_grid)
        true_freqs_masked = true_freqs[mask]
        true_h_fd_masked = true_h_fd[mask]

        surr_freqs, surr_h_fd = evaluate_surrogate_fd(params['q'], params['chi'], true_freqs_masked, surrogate)

        pycbc_surr_h_fd = pycbc.types.FrequencySeries(surr_h_fd, delta_f=true_freqs_masked[1]-true_freqs_masked[0], epoch=0)
        pycbc_true_h_fd = pycbc.types.FrequencySeries(true_h_fd_masked, delta_f=true_freqs_masked[1]-true_freqs_masked[0], epoch=0)

        mismatch = 1 - pycbc.filter.matchedfilter.match(
            pycbc_surr_h_fd, pycbc_true_h_fd,
            psd=pycbc.psd.aLIGOZeroDetHighPower(len(pycbc_true_h_fd), pycbc_true_h_fd.delta_f, f_lower),
            low_frequency_cutoff=f_min_grid
        )[0]

        mismatch_vals[ichi, iq] = mismatch

Q, Chi = np.meshgrid(q_vals, chi_vals)

interp_mismatch = RectBivariateSpline(chi_vals, q_vals, mismatch_vals, kx=3, ky=3, s=3)

q_fine = np.linspace(q_vals.min(), q_vals.max(), 400)
chi_fine = np.linspace(chi_vals.min(), chi_vals.max(), 400)
Q_fine, Chi_fine = np.meshgrid(q_fine, chi_fine)

mismatch_fine = interp_mismatch(chi_fine, q_fine)

plt.pcolormesh(Q_fine, Chi_fine, mismatch_fine, shading='auto', cmap='viridis')
plt.colorbar(label='Mismatch')
plt.xlabel('Mass Ratio q')
plt.ylabel('Spin χ')
plt.title('Mismatch between Surrogate and True Waveforms')
plt.savefig('Results/mismatch.pdf', dpi=400)
plt.show()
