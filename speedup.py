import time
import pickle
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
# Load trained surrogate
# ----------------------------------------------------
surrogate = load_surrogate("Models/surrogate_model.pkl")
print("Surrogate model loaded successfully!")

# ----------------------------------------------------
# Benchmark setup
# ----------------------------------------------------
f_lower = 20.0
f_min_grid = 25.0
f_max_grid = 1024.0
delta_t = 1/4096
nfft = 16 * 4096

masses = np.linspace(10, 400, 100) 
q_fixed = 5.0
chi_fixed = 0.0
n_trials = 5   

true_times = []
surr_times = []
speedups = []

# ----------------------------------------------------
# Benchmark loop
# ----------------------------------------------------
print("\nBenchmarking speedup for varying total masses...")

for M_total in masses:
    m2 = M_total / (1 + q_fixed)
    m1 = q_fixed * M_total / (1 + q_fixed)

    t_true_runs, t_surr_runs = [], []

    for _ in range(n_trials):
        # --- True waveform timing ---
        start_true = time.time()
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    spin1z=chi_fixed, spin2z=chi_fixed,
                                    delta_t=delta_t,
                                    f_lower=f_lower)
            h_td_raw = hp.numpy()
            tukey_window = windows.tukey(len(h_td_raw), alpha=0.1)
            h_td_windowed = h_td_raw * tukey_window
            if len(h_td_windowed) < nfft:
                h_td = np.pad(h_td_windowed, (0, nfft - len(h_td_windowed)))
            else:
                h_td = h_td_windowed[:nfft]
            freqs_true = rfftfreq(nfft, delta_t)
            h_fd_true = rfft(h_td)
            mask_true = (freqs_true >= f_min_grid) & (freqs_true <= f_max_grid)
            freqs_true = freqs_true[mask_true]
            h_fd_true = h_fd_true[mask_true]
        except Exception as e:
            print(f"Failed true waveform at M={M_total}: {e}")
            continue
        end_true = time.time()
        t_true_runs.append(end_true - start_true)

        # --- Surrogate timing ---
        start_surr = time.time()
        _, h_fd_surr = evaluate_surrogate_fd(q_fixed, chi_fixed, freqs_true, surrogate)
        end_surr = time.time()
        t_surr_runs.append(end_surr - start_surr)

    if not t_true_runs or not t_surr_runs:
        continue

    t_true_avg = np.mean(t_true_runs)
    t_surr_avg = np.mean(t_surr_runs)

    true_times.append(t_true_avg)
    surr_times.append(t_surr_avg)
    speedups.append(t_true_avg / t_surr_avg if t_surr_avg > 0 else np.nan)

    print(f"M={M_total:.1f}: True={t_true_avg:.3f}s, Surrogate={t_surr_avg:.3e}s, "
          f"Speedup={t_true_avg/t_surr_avg:.1f}x (averaged over {n_trials} runs)")

# ----------------------------------------------------
# Plot absolute runtimes
# ----------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(masses[:len(true_times)], true_times, marker="o", lw=2, label="True Model")
plt.plot(masses[:len(surr_times)], surr_times, marker="s", lw=2, label="Surrogate Model")
plt.xlabel("Total Mass $M_{tot} \\, (M_\\odot)$", fontsize=12)
plt.ylabel("Runtime (s)", fontsize=12)
plt.title(f"Runtime Comparison (q={q_fixed}, χ={chi_fixed})", fontsize=14)
plt.legend()
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.savefig("Results/Runtime_vs_Mass.pdf")
plt.show()

# ----------------------------------------------------
# Plot speedup
# ----------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(masses[:len(speedups)], speedups, marker="o", lw=2, color="purple")
plt.xlabel("Total Mass $M_{tot} \\, (M_\\odot)$", fontsize=12)
plt.ylabel("Speedup (True / Surrogate)", fontsize=12)
plt.title(f"Surrogate Speedup vs. True Model (q={q_fixed}, χ={chi_fixed})", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.savefig("Results/Speedup_vs_Mass.pdf")
plt.show()
