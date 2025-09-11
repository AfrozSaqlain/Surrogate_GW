import os
import pycbc
import pickle
import argparse
import pycbc.psd
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.signal import windows
import matplotlib.colors as colors
from scipy.fft import rfft, rfftfreq
from pycbc.waveform import get_td_waveform
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import UnivariateSpline, RectBivariateSpline

parser = argparse.ArgumentParser(description="Train a surrogate model for gravitational waveforms.")
parser.add_argument('--results-dir', type=str, default='Results', help='Directory to save results and plots.')
args = parser.parse_args()

if args.results_dir != 'Results' and not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

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

    spline_amp = UnivariateSpline(surrogate["sparse_freq_amp"], amp_recon_sparse, s=0, k=3, ext=0)
    spline_phase = UnivariateSpline(surrogate["sparse_freq_phase"], phase_recon_sparse, s=0, k=3, ext=0)

    amp_final = spline_amp(freqs_out)
    phase_final = spline_phase(freqs_out)

    norm_star = surrogate["interp_amp_norm"](chi_star, q_star)[0, 0]
    amp_final *= norm_star

    return freqs_out, amp_final * np.exp(1j * phase_final)


# ----------------------------------------------------
# Generate frequency-domain waveform
# ----------------------------------------------------
LALSIMULATION_RINGING_EXTENT = 19
def Planck_window_LAL(data, taper_method='LAL_SIM_INSPIRAL_TAPER_STARTEND', num_extrema_start=2, num_extrema_end=2):
    """
    Parameters:
    -----------
    data: 1D numpy array 
        data to taper
    taper_method: string
        Tapering method. Available methods are: 
        "LAL_SIM_INSPIRAL_TAPER_START"
        "LAL_SIM_INSPIRAL_TAPER_END"
        "LAL_SIM_INSPIRAL_TAPER_STARTEND"
    num_extrema_start: int
        number of extrema till which to taper from the start
    num_extrema_end: int
        number of extrema till which to taper from the end
        
    Returns:
    --------
    window: 1D numpy array
        Planck tapering window
    """
    start=0
    end=0
    n=0
    length = len(data)

    # Search for start and end of signal
    flag = 0
    i = 0
    while(flag == 0 and i < length):
        if (data[i] != 0.):
            start = i
            flag = 1
        i+=1
    if (flag == 0):
        raise ValueError("No signal found in the vector. Cannot taper.\n")

    flag = 0
    i = length - 1
    while( flag == 0 ):
        if( data[i] != 0. ):
                end = i
                flag = 1
        i-=1

    # Check we have more than 2 data points 
    if( (end - start) <= 1 ):
        raise RuntimeError( "Data less than 3 points, cannot taper!\n" )

    # Calculate middle point in case of short waveform
    mid = int((start+end)/2)

    window = np.ones(length)
    # If requested search for num_extrema_start-th peak from start and taper
    if( taper_method != "LAL_SIM_INSPIRAL_TAPER_END" ):
        flag = 0
        i = start+1
        while ( flag < num_extrema_start and i != mid ):
            if( abs(data[i]) >= abs(data[i-1]) and
                abs(data[i]) >= abs(data[i+1]) ):
            
                if( abs(data[i]) == abs(data[i+1]) ):
                    i+=1
                # only count local extrema more than 19 samples in
                if ( i-start > LALSIMULATION_RINGING_EXTENT ):
                    flag+=1
                n = i - start
            i+=1

        # Have we reached the middle without finding `num_extrema_start` peaks?
        if( flag < num_extrema_start ):
            n = mid - start
            print(f"""WARNING: Reached the middle of waveform without finding {num_extrema_start} extrema. Tapering only till the middle from the beginning.""")

        # Taper to that point
        realN = n
        window[:start+1] = 0.0
        realI = np.arange(1, n - 1)
        z = (realN - 1.0)/realI + (realN - 1.0)/(realI - (realN - 1.0))
        window[start+1: start+n-1] = 1.0/(np.exp(z) + 1.0)

    # If requested search for num_extrema_end-th peak from end
    if( taper_method == "LAL_SIM_INSPIRAL_TAPER_END" or taper_method == "LAL_SIM_INSPIRAL_TAPER_STARTEND" ):
        i = end - 1
        flag = 0
        while( flag < num_extrema_end and i != mid ):
            if( abs(data[i]) >= abs(data[i+1]) and
                abs(data[i]) >= abs(data[i-1]) ):
                if( abs(data[i]) == abs(data[i-1]) ):
                    i-=1
                # only count local extrema more than 19 samples in
                if ( end-i > LALSIMULATION_RINGING_EXTENT ):
                    flag+=1
                n = end - i
            i-=1

        # Have we reached the middle without finding `num_extrema_end` peaks?
        if( flag < num_extrema_end ):
            n = end - mid
            print(f"""WARNING: Reached the middle of waveform without finding {num_extrema_end} extrema. Tapering only till the middle from the end.""")

        # Taper to that point
        realN = n
        window[end:] = 0.0        
        realI = -np.arange(-n+2, 0)
        z = (realN - 1.0)/realI + (realN - 1.0)/(realI - (realN - 1.0))
        window[end-n+2:end] = 1.0/(np.exp(z) + 1.0)

    return window

def planck_taper(N, epsilon=0.1):
    """
    Stable Planck-taper window.
    """
    if not (0 < epsilon < 0.5):
        raise ValueError("epsilon must be between 0 and 0.5")

    w = np.ones(N)
    L = int(epsilon * N)

    if L == 0:
        return w

    n = np.arange(1, L)
    x = L/n - L/(L-n)
    w[:L-1] = expit(-x)
    w[0] = 0.0

    x = L/(L-n) - L/n
    w[-(L-1):] = expit(-x)
    w[-1] = 0.0

    return w

def generate_fd_waveform(params, f_lower, delta_t, window_type='planck', epsilon=0.1):
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

    if window_type == "tukey":
        window = windows.tukey(len(h_td_raw), alpha=0.1)
    elif window_type == "planck":
        # window = planck_taper(len(h_td_raw), epsilon=epsilon)
        window = Planck_window_LAL(h_td_raw, taper_method='LAL_SIM_INSPIRAL_TAPER_STARTEND', num_extrema_start=2, num_extrema_end=2)
    else:
        window = np.ones(len(h_td_raw))

    h_td_windowed = h_td_raw * window

    L = len(h_td_windowed)
    nfft = 2 ** int(np.ceil(np.log2(2 * L)))

    h_td = np.pad(h_td_windowed, (0, nfft - L))

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
# nfft = 16 * 4096

total_mass = 40
q_vals = np.linspace(1, 10, 100)
chi_vals = np.linspace(-0.8, 0.6, 100)

mismatch_vals = np.zeros((len(chi_vals), len(q_vals)))

for iq, q in enumerate(q_vals):
    for ichi, chi in enumerate(chi_vals):
        params = {'q': q, 'chi': chi}

        true_freqs, true_h_fd = generate_fd_waveform(params, f_lower, delta_t)
        if true_freqs is None:
            mismatch_vals[ichi, iq] = np.nan
            continue

        mask = (true_freqs >= f_min_grid) & (true_freqs <= f_max_grid)
        true_freqs_masked = true_freqs[mask]
        true_h_fd_masked = true_h_fd[mask]

        surr_freqs, surr_h_fd = evaluate_surrogate_fd(params['q'], params['chi'], true_freqs_masked, surrogate)

        pycbc_surr_h_fd = pycbc.types.FrequencySeries(surr_h_fd, delta_f=true_freqs_masked[1]-true_freqs_masked[0], epoch=0)
        pycbc_true_h_fd = pycbc.types.FrequencySeries(true_h_fd_masked, delta_f=true_freqs_masked[1]-true_freqs_masked[0], epoch=0)

        mismatch = 1 - pycbc.filter.matchedfilter.optimized_match(
            pycbc_surr_h_fd, pycbc_true_h_fd,
            psd=pycbc.psd.aLIGOZeroDetHighPower(len(pycbc_true_h_fd), pycbc_true_h_fd.delta_f, f_lower),
            low_frequency_cutoff=f_min_grid
        )[0]

        mismatch_vals[ichi, iq] = mismatch

Q, Chi = np.meshgrid(q_vals, chi_vals)

interp_mismatch = RegularGridInterpolator(
    (chi_vals, q_vals), mismatch_vals,
    method="linear", bounds_error=False, fill_value=None
)

q_fine = np.linspace(q_vals.min(), q_vals.max(), 400)
chi_fine = np.linspace(chi_vals.min(), chi_vals.max(), 400)
Q_fine, Chi_fine = np.meshgrid(q_fine, chi_fine)

mismatch_fine = interp_mismatch((Chi_fine, Q_fine))

plt.pcolormesh(
    Q, Chi, mismatch_vals,
    shading='auto',
    cmap='viridis',
    norm=colors.LogNorm(vmin=mismatch.min(), vmax=mismatch.max())
)
plt.colorbar(label='Mismatch (log scale)')
plt.xlabel('Mass Ratio q')
plt.ylabel('Spin χ')
plt.title('Mismatch between Surrogate and True Waveforms')
plt.savefig(f'{args.results_dir}/mismatch.pdf', dpi=400)
plt.show()


plt.pcolormesh(
    Q_fine, Chi_fine, mismatch_fine,
    shading='auto',
    cmap='viridis',
    norm=colors.LogNorm(vmin=mismatch_fine.min(), vmax=mismatch_fine.max())
)
plt.colorbar(label='Mismatch (log scale)')
plt.xlabel('Mass Ratio q')
plt.ylabel('Spin χ')
plt.title('Mismatch between Surrogate and True Waveforms')
plt.savefig(f'{args.results_dir}/interpolated_mismatch.pdf', dpi=400)
plt.show()
