import numpy as np
import spikeinterface.full as si
import pynapple as nap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import scipy
import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path
from sklearn.mixture import GaussianMixture
import ast
import os

default_dinmap = {
    'licks': 0,
    'timing_pulse': 1
}

default_ainmap = {
    'photometry': 0,
    'accelerometer_x': 1,
    'accelerometer_y': 2,
    'accelerometer_z': 3,
    'solenoid': 4,
    'tone': 5,
    'chrimson': 6,
    'chr2': 7
}

# ints = [10,20,40,100,200,400] #light intensities in microwatts
default_ints = [20, 40, 60, 80, 100]  # light intensities in microwatts
default_optoidx = {'chrimson': default_ints,
                   'chr2': default_ints}


def get_nidq_aligner(imec_rec, digital_events, pulse_din=1):

    digchids = digital_events.channel_ids
    imec_sync = imec_rec.select_channels([imec_rec.channel_ids[-1]])
    imec_time = imec_sync.get_times()
    imec_time = imec_time - imec_time[0]  # zeroing time
    sync_trace = imec_sync.get_traces()
    #convert sync trace to boolean, get the min and max to define a threshold
    threshold = (np.max(sync_trace) + np.min(sync_trace)) / 2
    sync_trace_bool = sync_trace > threshold
    sync_trace_bool = sync_trace_bool.astype(int).squeeze()
    #get the rising/falling edges of the boolean trace
    off_edges = np.where((np.diff(sync_trace_bool) == 1))[0] #| (np.diff(sync_trace_bool) == -1))[0]
    on_edges = np.where((np.diff(sync_trace_bool) == -1))[0]
    imec_onsets = imec_time[on_edges+1]  # +1 because diff reduces length by 1
    imec_offsets = imec_time[off_edges+1]

    nidq_edges = digital_events.get_event_times(digchids[pulse_din])
    #the first half -1 are onsets, the second half are offsets
    nind = len(nidq_edges)
    nidq_offsets = nidq_edges[:nind//2]
    nidq_onsets = nidq_edges[nind//2:]

    #concatenate onsets and offsets
    pulseA = np.sort(np.concatenate((imec_onsets, imec_offsets)))
    pulseB = np.sort(np.concatenate((nidq_onsets, nidq_offsets)))

    # Often, clocks drift linearly. Let's fit tA = slope * tB + intercept
    coef = np.polyfit(pulseB, pulseA, 1)
    slope, intercept = coef
    print(f"Clock alignment fit: tA ≈ {slope:.9f} * tB + {intercept:.6f}")

    # Create a function to map any DAQ B timestamp into DAQ A time
    map_B_to_A = np.poly1d(coef)

    return map_B_to_A

def binarize_trace(trace, threshold=None):
    if threshold is None:
        threshold = (np.max(trace) + np.min(trace)) / 2
    digital = trace > threshold
    digital = digital.astype(int)
    return digital

def get_edge_times(digital_trace, time_array, edge_type='rising', return_idxs=False):
    if edge_type == 'rising':
        edge_idxs = np.where((np.diff(digital_trace) == 1))[0]
    elif edge_type == 'falling':
        edge_idxs = np.where((np.diff(digital_trace) == -1))[0]
    else:
        raise ValueError("edge_type must be 'rising' or 'falling'")
    edge_times = time_array[edge_idxs + 1]  # +1 because diff reduces length by 1
    if return_idxs:
        return edge_times, edge_idxs + 1
    else:
        return edge_times

def parse_solenoid_analog(analog_events, nidq_t_aligned, idx={'solenoid':4}, plot=False):
    time_support = nap.IntervalSet(start=nidq_t_aligned[0], end=nidq_t_aligned[-1])
    sol_ch = idx['solenoid']
    solenoid_trace = analog_events[:, sol_ch]
    thresh = find_mode_thresholds(solenoid_trace, n_modes=2, nbin=1000, plot=plot)
    #thresh = _find_thresholds_gmm(solenoid_trace, n_modes=2, plot=plot)
    # threshold solenoid trace to get digital values
    solenoid_digital = binarize_trace(solenoid_trace, threshold=thresh[0])
    solenoid_onsets = get_edge_times(solenoid_digital, nidq_t_aligned, edge_type='rising')
    solenoid_offsets = get_edge_times(solenoid_digital, nidq_t_aligned, edge_type='falling')
    solenoid_onsets = nap.Ts(t=solenoid_onsets, time_support=time_support)
    solenoid_offsets = nap.Ts(t=solenoid_offsets, time_support=time_support)
    solenoid_ts_list = [solenoid_onsets, solenoid_offsets]
    solenoid_ts_index = ['solenoid_on', 'solenoid_off']
    return solenoid_ts_list, solenoid_ts_index

def parse_tones_digital(digital_events, solenoid_ts_list, map2imectime, time_support, idx={'tone':3}):
    time_support = nap.IntervalSet(start=time_support.start, end=time_support.end)
    solenoid_onsets = solenoid_ts_list[0].t

    tone_din = idx['tone']

    digchids = digital_events.channel_ids
    times = digital_events.get_event_times(digchids[tone_din])
    tone_times = map2imectime(times)

    mididx = len(tone_times) // 2
    tone_onsets = tone_times[:mididx]
    tone_offsets = tone_times[mididx:]

    return parse_tones(tone_onsets, tone_offsets, solenoid_onsets, time_support)


def parse_tones_analog(analog_events, solenoid_ts_list, nidq_t_aligned, idx={'tone':5}):
    time_support = nap.IntervalSet(start=nidq_t_aligned[0], end=nidq_t_aligned[-1])
    solenoid_onsets = solenoid_ts_list[0].t

    tone_ch = idx['tone']
    tone_trace = analog_events[:,tone_ch]
    tone_digital = binarize_trace(tone_trace)
    tone_onsets = get_edge_times(tone_digital, nidq_t_aligned, edge_type='rising')

    #get the inter onset intervals
    tone_offsets = get_edge_times(tone_digital, nidq_t_aligned, edge_type='falling')

    return parse_tones(tone_onsets, tone_offsets, solenoid_onsets, time_support)

def parse_tones(tone_onsets, tone_offsets, solenoid_onsets, time_support, verbose=False):
    inter_onset_intervals = np.diff(tone_onsets)
    inter_offset_intervals = np.diff(tone_offsets)

    #append infinity to the end of inter_onset_intervals and inter_offset_intervals to make them the same length as tone_onsets and tone_offsets
    inter_onset_intervals = np.append(np.inf, inter_onset_intervals)
    inter_offset_intervals = np.append(inter_offset_intervals, np.inf)

    #keep only tone onsets preceded by an interval of at least 0.001 seconds, and keep only tone offsets followed by an interval of at least 0.01
    min_isi = 0.009  # minimum inter-stimulus interval in seconds
    valid_onset_intervals = inter_onset_intervals >= min_isi
    valid_offset_intervals = inter_offset_intervals >= min_isi
    valid_tone_onsets = tone_onsets[valid_onset_intervals]
    valid_tone_offsets = tone_offsets[valid_offset_intervals]
    #require that the number of valid tone onsets and offsets are the same
    if len(valid_tone_onsets) != len(valid_tone_offsets):
        raise ValueError("Number of valid tone onsets does not match number of valid tone offsets")
    valid_tone_durations = valid_tone_offsets - valid_tone_onsets


    #get the duration of each tone
    rounded_tone_durations = np.round(valid_tone_durations, 2)
    #get indices of tone durations that are not 2, 0.1, or 0.2s
    odd_tone_idxs = np.where((rounded_tone_durations != 2.0) & (rounded_tone_durations != 0.1) & (rounded_tone_durations != 0.2))[0]

    #round durations to nearest 0.0001
    #get indices of tone onsets that are found in solenoid onsets
    reward_tone_onset_idxs = np.where(np.isin(np.round(valid_tone_onsets,1), np.round(solenoid_onsets,1)))[0]

    reward_tone_onsets = valid_tone_onsets[reward_tone_onset_idxs]
    reward_tone_offsets = valid_tone_offsets[reward_tone_onset_idxs]


    #if length of reward tone onsets is not equal to length of solenoid onsets, print a warning
    if len(reward_tone_onsets) != len(solenoid_onsets):
        print("Warning: number of reward tone onsets does not match number of solenoid onsets")
        if verbose:
            print('reward_tone_onsets', reward_tone_onsets)
            print('number of reward tones', len(reward_tone_onsets))
            print('solenoid_onsets', solenoid_onsets)
            print('number of solenoid onsets', len(solenoid_onsets))

    long_tone_bool = np.round(valid_tone_durations) == 2
    long_tone_idxs = np.where(long_tone_bool)[0]
    timeout_tone_onsets = valid_tone_onsets[long_tone_idxs]
    timeout_tone_offsets = valid_tone_offsets[long_tone_idxs]

    #cue tones are tones where tone duration is 0.1 seconds
    cue_tone_bool = np.round(valid_tone_durations,1) == 0.1
    cue_tone_idxs = np.where(cue_tone_bool)[0]
    cue_tone_onsets = valid_tone_onsets[cue_tone_idxs]
    cue_tone_offsets = valid_tone_offsets[cue_tone_idxs]

    #error tones are the tones that are not reward, timeout, or cue tones
    all_special_tone_idxs = np.unique(np.concatenate((reward_tone_onset_idxs, long_tone_idxs, cue_tone_idxs, odd_tone_idxs)))
    error_tone_idxs = np.setdiff1d(np.arange(len(valid_tone_onsets)), all_special_tone_idxs)
    error_tone_onsets = valid_tone_onsets[error_tone_idxs]
    error_tone_offsets = valid_tone_offsets[error_tone_idxs]

    #for all odd tone idxs, add the start to cue onset and the end - 200 to error onset
    for idx in odd_tone_idxs:
        ontime = valid_tone_onsets[idx]
        offtime = valid_tone_offsets[idx]
        cue_tone_onsets = np.append(cue_tone_onsets, ontime)
        cue_tone_offsets = np.append(cue_tone_offsets, offtime - 0.2)
        error_tone_onsets = np.append(error_tone_onsets, offtime - 0.2)
        error_tone_offsets = np.append(error_tone_offsets, offtime)
    error_tone_onsets.sort()
    error_tone_offsets.sort()
    cue_tone_onsets.sort()
    cue_tone_offsets.sort()

    #check that outcomes don't happen between cues
    for i, cue in enumerate(cue_tone_onsets):
        #get the next cue
        next_cue = cue_tone_onsets[i+1] if i+1 < len(cue_tone_onsets) else np.inf
        #check if any reward, timeout, or error tones happen between cue and next cue
        reward_in_between = np.where((reward_tone_onsets > cue) & (reward_tone_onsets < next_cue))[0]
        timeout_in_between = np.where((timeout_tone_onsets > cue) & (timeout_tone_onsets < next_cue))[0]
        error_in_between = np.where((error_tone_onsets > cue) & (error_tone_onsets < next_cue))[0]
        if len(reward_in_between) + len(timeout_in_between) + len(error_in_between) > 1:
            raise ValueError(f"Multiple outcome tones found between cue at index {i} and next cue.")

    n_outcomes = len(reward_tone_onsets) + len(timeout_tone_onsets) + len(error_tone_onsets)
    n_cues = len(cue_tone_onsets)
    if n_outcomes != n_cues:
        outcomes = np.concatenate((reward_tone_onsets, timeout_tone_onsets, error_tone_onsets))
        outcomes.sort()

        for i in range(min(n_outcomes, n_cues)):
            #if the cue tone onset is greater than the outcome tone onset, print a warning
            if cue_tone_onsets[i] > outcomes[i]:
                raise ValueError(f"Cue tone onset at index {i} occurs after outcome tone onset.")

        print("Warning: number of outcomes does not match number of cue onsets")
        if verbose:
            print('reward_tone_onsets', reward_tone_onsets)
            print('number of reward tones', len(reward_tone_onsets))
            print('timeout_tone_onsets', timeout_tone_onsets)
            print('number of timeout tones', len(timeout_tone_onsets))
            print('error_tone_onsets', error_tone_onsets)
            print('number of error tones', len(error_tone_onsets))
            print('cue_tone_onsets', cue_tone_onsets)
            print('number of cue tones', len(cue_tone_onsets))

    cue_tone_on_ts = nap.Ts(t=cue_tone_onsets, time_support=time_support)
    cue_tone_off_ts = nap.Ts(t=cue_tone_offsets, time_support=time_support)
    reward_tone_on_ts = nap.Ts(t=reward_tone_onsets, time_support=time_support)
    reward_tone_off_ts = nap.Ts(t=reward_tone_offsets, time_support=time_support)
    timeout_tone_on_ts = nap.Ts(t=timeout_tone_onsets, time_support=time_support)
    timeout_tone_off_ts = nap.Ts(t=timeout_tone_offsets, time_support=time_support)
    error_tone_on_ts = nap.Ts(t=error_tone_onsets, time_support=time_support)
    error_tone_off_ts = nap.Ts(t=error_tone_offsets, time_support=time_support)

    ts_list = [cue_tone_on_ts, cue_tone_off_ts,
               reward_tone_on_ts, reward_tone_off_ts,
               timeout_tone_on_ts, timeout_tone_off_ts,
               error_tone_on_ts, error_tone_off_ts]
    ts_index = ['start_tone_on', 'start_tone_off',
                'reward_tone_on', 'reward_tone_off',
                'timeout_tone_on', 'timeout_tone_off',
                'error_tone_on', 'error_tone_off']

    return ts_list, ts_index


def get_digital_ts(digital_events, map2imectime, time_support, din=None, event_name='lick'):
    if din is None:
        print("Warning: din not specified, using default channel 0 for lick events.")
        din = 0

    #lick_ontimes, lickofftimes = get_licks(digital_events, map2imectime)
    digchids = digital_events.channel_ids
    times = digital_events.get_event_times(digchids[din])
    times = map2imectime(times)
    n_times = len(times)
    offtimes = times[n_times//2:]
    ontimes = times[:n_times//2]

    on_ts = nap.Ts(t=ontimes, time_support=time_support)
    off_ts = nap.Ts(t=offtimes, time_support=time_support)
    tslist = [on_ts, off_ts]
    indexlist = [f'{event_name}_on', f'{event_name}_off']
    return tslist, indexlist


def _find_thresholds_gmm(data, n_modes=7, plot=False):
    """Find thresholds using Gaussian Mixture Model."""
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_modes, random_state=42)
    gmm.fit(data.reshape(-1, 1))
    vals = gmm.sample(1000)[0].flatten()

    # Get peak locations (means of Gaussians)
    peaks = gmm.means_.flatten()
    sort_idx = np.argsort(peaks)
    peaks = peaks[sort_idx]

    # Calculate thresholds as midpoints between adjacent peaks
    thresholds = []
    for i in range(len(peaks) - 1):
        # Find the minimum probability point between two peaks
        x_range = np.linspace(peaks[i], peaks[i + 1], 1000)
        probs = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
        threshold = x_range[np.argmin(probs)]
        thresholds.append(threshold)

    if plot:
        nb = 1000
        pl = plt.figure(figsize=(8, 4))
        plt.hist(data, bins=nb, density=True, alpha=0.3, label='Data Histogram (log scale)')
        #xaxis = np.linspace(np.min(data), np.max(data), nb)
        #plt.plot(xaxis, vals)
        for peak in peaks:
            plt.axvline(peak, color='green', linestyle=':', label='Peak' if peak == peaks[0] else "")
        for thresh in thresholds:
            plt.axvline(thresh, color='red', linestyle='-', label='Threshold' if thresh == thresholds[0] else "")
        #log scale y axis
        plt.yscale('log')
        for t in thresholds:
            plt.axvline(t, color='gray', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

    return sorted(thresholds)


def find_mode_thresholds(data, n_modes=7, nbin=100, plot=False):
    """
    Identify threshold cutoffs separating modes in a noisy 1D time series.

    Parameters:
        data : array-like
            The input time series values.
        n_modes : int
            Expected number of modes (peaks).
        bandwidth : float or None
            Bandwidth for KDE smoothing (if None, chosen automatically).
        plot : bool
            Whether to plot the estimated density and thresholds.

    Returns:
        thresholds : list of floats
            Sorted values dividing the modes.
    """
    #all data that is less than 0 are set to 1
    #floor all data to 0
    data = np.array(data)
    hist = np.histogram(data, nbin)
    counts = hist[0]
    counts_smooth = scipy.ndimage.gaussian_filter1d(counts, nbin/500)
    #smooth counts with gaussian filte
    bin_edges = hist[1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    xs = bin_centers

    # Find peaks (modes)
    peak_idxs, _ = find_peaks(counts_smooth)

    peaks = xs[peak_idxs]
    if len(peaks) < n_modes:
        raise ValueError(f"Found only {len(peaks)} modes, fewer than expected {n_modes} modes.")
    if len(peaks) > n_modes:
        prominences = scipy.signal.peak_prominences(counts_smooth, peak_idxs)[0]
        #get the n_modes most prominent peaks
        top_n_idxs = np.argsort(prominences)[-n_modes:]
        peaks = peaks[top_n_idxs]

    peaks = np.sort(peaks)
    #If COUNTS smooth between peaks is not zero, find the local minima between peaks
    thresholds = []
    for i in range(len(peaks)-1):
        peak1_idx = np.where(xs == peaks[i])[0][0]
        peak2_idx = np.where(xs == peaks[i+1])[0][0]
        segment = counts_smooth[peak1_idx:peak2_idx]
        if np.min(segment) > 0:
            #find local minima in segment
            local_min_idxs, _ = find_peaks(-segment)
            if len(local_min_idxs) > 0:
                #get the global minimum
                global_min_idx = local_min_idxs[np.argmin(segment[local_min_idxs])]
                threshold = xs[peak1_idx + global_min_idx]
                thresholds.append(threshold)
        else:
            #get midpoint between peaks
            threshold = (peaks[i] + peaks[i+1]) / 2.0
            thresholds.append(threshold)
    if len(thresholds) != n_modes - 1:
        raise ValueError(f"Number of thresholds found ({len(thresholds)}) does not match expected ({n_modes - 1}).")

    # Optionally plot
    if plot:
        plt.figure(figsize=(8, 4))
        plt.hist(data, bins=nbin, density=True, alpha=0.3, label='Data Histogram (log scale)')
        plt.plot(xs, counts_smooth)
        for peak in peaks:
            plt.axvline(peak, color='green', linestyle=':', label='Peak' if peak == peaks[0] else "")
        for thresh in thresholds:
            plt.axvline(thresh, color='red', linestyle='-', label='Threshold' if thresh == thresholds[0] else "")
        #log scale y axis
        plt.yscale('log')
        for t in thresholds:
            plt.axvline(t, color='gray', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

    return sorted(thresholds)

#get on and off edges for each level
def get_opto_edges(trace, time_array, powers, plot=False):
    n_powers = len(powers)
    max_power = max(powers)
    lower_cutoff = 500#np.mean(trace) * 2
    #scaling_factor = (max_trace - mean_trace) ** 0.66
    #lower_cutoff = (scaling_factor * (min_power**0.25))/2 + np.mean(trace)
    filtered_trace = trace[trace >= lower_cutoff]
    thresholds = _find_thresholds_gmm(np.log(filtered_trace), n_modes=n_powers, plot=plot)
    thresholds = np.exp(thresholds)
    thresholds = np.concatenate((np.array([lower_cutoff]), thresholds))

    #get all edges greater than the first threshold
    min_threshold = min(thresholds)
    max_threshold = max(thresholds)
    digital_trace = trace > min_threshold
    digital_trace = digital_trace.astype(int)
    onsets, onidxs = get_edge_times(digital_trace, time_array, edge_type='rising', return_idxs=True)
    offsets, offidxs = get_edge_times(digital_trace, time_array, edge_type='falling', return_idxs=True)
    durations = offsets - onsets
    #filter out any onsets/offsets where duration is less than 0.01 seconds
    valid_durations = durations >= 0.009
    onidxs = np.array(onidxs)[valid_durations]
    offidxs = np.array(offidxs)[valid_durations]

    opto_trace = np.zeros(len(trace))
    for onidx, offidx in zip(onidxs, offidxs):
        section = trace[onidx:offidx]
        section_level = np.mean(section)
        #identify which theshold this section corresponds to
        for pidx, level in enumerate(powers):
            if level == max_power:
                if section_level > max_threshold:
                    opto_trace[onidx:offidx] = level
            else:
                if (section_level > thresholds[pidx]) and (section_level <= thresholds[pidx + 1]):
                    opto_trace[onidx:offidx] = level
    #break optoidx into binary traces for each level
    onset_times = []
    offset_times = []
    for pidx, level in enumerate(powers):
        binary_trace = (opto_trace == level).astype(int)
        ontimes, _ = get_edge_times(binary_trace, time_array, edge_type='rising', return_idxs=True)
        offtimes, _ = get_edge_times(binary_trace, time_array, edge_type='falling', return_idxs=True)
        onset_times.append(ontimes)
        offset_times.append(offtimes)
    return onset_times, offset_times, opto_trace


def get_opto_ts(analog_events, nidq_t_aligned, idx=None, pwrs=None, plot=False):
    if idx is None:
        print("Warning: idx not specified, using default channels for chrimson and chr2.")
        idx = {'chrimson': 6, 'chr2': 7}
    if pwrs is None:
        print("Warning: power levels not specified, using default power levels.")
        ints = [10,20,40,100,200,400]
        pwrs = {'chrimson': ints, 'chr2': ints}

    #get items from idx that contain 'chrimson' or 'chr2'
    idx = {k: v for k, v in idx.items() if k in ['chrimson', 'chr2']}
    ts_list = []
    ts_idx = []
    time_support = nap.IntervalSet(start=nidq_t_aligned[0], end=nidq_t_aligned[-1])
    for i in idx:
        ch = idx[i]
        pw = pwrs[i]
        trace = analog_events[:,ch]
        on_times, off_times, opto_trace = get_opto_edges(trace, nidq_t_aligned, pw, plot=plot)
        ts_list.append(nap.Tsd(t=nidq_t_aligned, d=opto_trace, time_support=time_support))
        ts_idx.append(f"{i}_trace")
        for pidx, level in enumerate(pw):
            on = nap.Ts(t=on_times[pidx], time_support=time_support)
            off = nap.Ts(t=off_times[pidx], time_support=time_support)
            ts_list.append(on)
            ts_idx.append(f"{i}_on_{level}")
            ts_list.append(off)
            ts_idx.append(f"{i}_off_{level}")
    return ts_list, ts_idx

#for each start tone timestamp, find the closest outcome tone timestamp that follows
def parse_trials(start_tone_ts, outcome_tone_ts, lick_ts, require_first_licks=True):
    #for each timestamp in outcome_tone_ts, find the closest timestamp in start_tone_ts that is less than it
    time_support = lick_ts.time_support
    trial_starts = []
    trial_ends = []
    first_licks = []
    cues = []
    for t in outcome_tone_ts.t:
        #get timestamps in start_tone_ts.t that are less than t
        possible_starts = start_tone_ts.t[start_tone_ts.t < t]
        if len(possible_starts) > 0:
            trial_start = possible_starts[-1]
            trial_starts.append(trial_start-1.5)
            cues.append(trial_start)
            trial_ends.append(t+10)

            assert trial_start < t, "Trial start {} is not less than trial end {}".format(trial_start, t)
            trial_licks = lick_ts.t[(lick_ts.t >= trial_start) & (lick_ts.t <= t)]
            if len(trial_licks) > 0:
                first_lick = trial_licks[0]
            else:
                first_lick = np.nan
            first_licks.append(first_lick)

        else:
            raise ValueError("No matching start tone found for outcome tone at time {}".format(t))


    #ensure that the lengths of trial_starts and trial_ends are the same
    assert len(trial_starts) == len(trial_ends)

    trial_starts = nap.Ts(np.array(trial_starts), time_support=time_support)
    trial_ends = nap.Ts(np.array(trial_ends), time_support=time_support)
    cues_ts = nap.Ts(np.array(cues), time_support=time_support)

    # if any first licks are nan, raise an error
    if require_first_licks:
        if any(np.isnan(first_licks)):
            raise ValueError("Some trials are missing first licks: {}".format(first_licks))
        first_licks_ts = nap.Ts(np.array(first_licks), time_support=time_support)
    else:
        first_licks_ts = None
    return trial_starts,trial_ends, cues_ts, first_licks_ts

def parse_lick_bouts(lick_ts, ili_thresh=0.5):
    #thresh from https://pmc.ncbi.nlm.nih.gov/articles/PMC6063358/
    time_support = lick_ts.time_support
    lick_times = lick_ts.t
    if len(lick_times) == 0:
        return [], []
    ilis = np.diff(lick_times)
    bout_starts = [lick_times[0]]
    bout_ends = []
    for idx, ili in enumerate(ilis):
        if ili > ili_thresh:
            bout_ends.append(lick_times[idx])
            bout_starts.append(lick_times[idx + 1])
    bout_ends.append(lick_times[-1])

    bout_start_ts = nap.Ts(np.array(bout_starts), time_support=time_support)
    bout_end_ts = nap.Ts(np.array(bout_ends), time_support=time_support)
    return bout_start_ts, bout_end_ts

def match_starts_and_ends(starts_ts, ends_ts):
    #check if starts and ends are pynapple ts
    if isinstance(starts_ts, nap.Ts) and isinstance(ends_ts, nap.Ts):
        starts = starts_ts.t
        ends = ends_ts.t
    elif isinstance(starts_ts, np.ndarray) and isinstance(ends_ts, np.ndarray):
        starts = starts_ts
        ends = ends_ts
    else:
        raise ValueError("starts_ts and ends_ts must be either pynapple Ts or numpy arrays")

    matched_starts = []
    matched_ends = []
    for s in starts:
        possible_ends = ends[ends > s]
        if len(possible_ends) > 0:
            matched_starts.append(s)
            matched_ends.append(possible_ends[0])

    if isinstance(starts_ts, nap.Ts) and isinstance(ends_ts, nap.Ts):
        time_support = starts_ts.time_support
        matched_starts = nap.Ts(np.array(matched_starts), time_support=time_support)
        matched_ends = nap.Ts(np.array(matched_ends), time_support=time_support)

    return matched_starts, matched_ends

def parse_base_folder(base_folder, test_mode=False):
    #base folder is where the recording files live
    if base_folder is None:
        if test_mode:
            #base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250822")
            base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905")
        else:
            #query user to enter base folder in command line
            print(r"base folder example: C:\Users\assad\Documents\analysis_files\DS13\DS13_20250822")
            base_folder_input = input("Please enter the base folder path: ")
            base_folder = Path(base_folder_input)

    elif not isinstance(base_folder, Path):
        base_folder = Path(base_folder)

    return base_folder

def get_spikeglx_folder(base_folder):
    #spikeglx folder is where the _g# folder lives
    #infer the spikeglx folder by looking for a folder that ends with the same pattern as the base folder, plust "_g#"
    subfolders = [f for f in base_folder.iterdir() if f.is_dir() and f.name.startswith(base_folder.name) and '_g' in f.name]
    if len(subfolders) == 0:
        raise FileNotFoundError(f"No SpikeGLX folder found in {base_folder}")
    elif len(subfolders) > 1:
        print(f"Multiple SpikeGLX folders found in {base_folder}, enter the number corresponding to the desired _g# folder:")
        gnum = input(f"Available folders: {[f.name for f in subfolders]} \n Enter g# number: ")
        spikeglx_folder = [f for f in subfolders if f"_g{gnum}" in f.name][0]
    else:
        spikeglx_folder = subfolders[0]
    return spikeglx_folder

def get_timing_pulse_din(dinmap):
    if 'timing_pulse' not in dinmap.keys():
        print("Warning: 'timing_pulse' not found in dinmap keys. Using default channel 1 for timing pulse.")
        pulse_din = 1
    else:
        pulse_din = dinmap['timing_pulse']
    return pulse_din

def get_maps_from_txt(base_folder):
    #get 'config.txt' from base_folder
    config_file = base_folder / 'nidaq_map.txt'
    if not config_file.exists():
        raise FileNotFoundError(f"No nidaq_map.txt file found in {base_folder}")

    vars_dict = {}
    with open(config_file) as f:
        tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                name = node.targets[0].id
                value = ast.literal_eval(node.value)
                vars_dict[name] = value

    #check that dinmap, ainmap, and optoidx are in vars_dict
    required_keys = ['dinmap', 'ainmap', 'optoidx']
    for key in required_keys:
        if key not in vars_dict.keys():
            raise KeyError(f"{key} not found in nidaq_map.txt")

    dinmap = vars_dict["dinmap"]
    ainmap = vars_dict["ainmap"]
    optoidx = vars_dict["optoidx"]

    return dinmap, ainmap, optoidx

def parse_channel_maps(base_folder, dinmap, ainmap, optoidx):
    #if all of dinmap, ainmap, and optoidx are None, try to get them from nidaq_map.txt
    #dinmap should be a dictionary mapping digital channel ids to event names
    #ainmap should be a dictionary mapping analog channel ids to event names
    if (dinmap is not None) and (ainmap is not None) and (optoidx is not None):
        return dinmap, ainmap, optoidx
    elif (dinmap is None) and (ainmap is None) and (optoidx is None):
        #check that nidaq_map.txt exists in base_folder
        config_file = base_folder / 'nidaq_map.txt'
        if config_file.exists():
            dinmap, ainmap, optoidx = get_maps_from_txt(base_folder)
        else:
            print("Warning: No nidaq_map.txt file found in base folder, using default channel maps.")
            dinmap = default_dinmap
            ainmap = default_ainmap
            optoidx = default_optoidx
    else:
        raise ValueError("Either provide all of dinmap, ainmap, and optoidx, or none to use nidaq_map.txt or defaults.")

    return dinmap, ainmap, optoidx

def get_pynapple_folder_and_file(base_folder, overwrite=False):
    #pynapple folder is where pynapple outputs will be saved
    pynapple_folder = base_folder / "pynapple"
    #if pynapple_folder does not exist, create it
    if not pynapple_folder.exists():
        pynapple_folder.mkdir(parents=True, exist_ok=True)
    pynapple_file = pynapple_folder / "binary_signals.npz"

    return pynapple_folder, pynapple_file

def load_pynapple_file(pynapple_file):
    if pynapple_file.exists():
        print(f"Loading existing binary signals from {pynapple_file}, set overwrite=True to re-parse.")
        bs = nap.load_file(str(pynapple_file))
        return bs
    else:
        print(f"No existing binary signals found at {pynapple_file}, parsing new signals.")
        return None


def get_binary_signals(base_folder=None, dinmap=None,ainmap=None, optoidx=None, overwrite=False, plot=False, test_mode=False):
    #section 1: folder handling
    base_folder = parse_base_folder(base_folder, test_mode=test_mode)
    spikeglx_folder = get_spikeglx_folder(base_folder)
    pynapple_folder, pynapple_file = get_pynapple_folder_and_file(base_folder, overwrite=overwrite)

    if not overwrite:
        bs = load_pynapple_file(pynapple_file)
        if bs is not None:
            return bs

    #section 2: channel mapping
    dinmap, ainmap, optoidx = parse_channel_maps(base_folder, dinmap, ainmap, optoidx)

    #section 3: read in spikeglx data
    imec_rec = si.read_spikeglx(spikeglx_folder, stream_name='imec0.ap', load_sync_channel=True)
    nidq_rec = si.read_spikeglx(spikeglx_folder, stream_id='nidq')

    chids = nidq_rec.channel_ids #get all channel ids--channels are indexed using alphanumeric string identifier
    anachids = [chid for chid in chids if 'D' not in str(chid)] #get analog channel ids (ones without 'D' in them)

    digital_events = si.read_spikeglx_event(spikeglx_folder) #get raw digital events
    analog_events = nidq_rec.get_traces(channel_ids=anachids)

    #section 4: align nidq to imec time
    pulse_din = get_timing_pulse_din(dinmap)
    map2imectime = get_nidq_aligner(imec_rec, digital_events, pulse_din)
    nidq_t = nidq_rec.get_times()
    nidq_t = nidq_t - nidq_t[0]  # zeroing time
    nidq_t_aligned = map2imectime(nidq_t)
    time_support = nap.IntervalSet(start=nidq_t_aligned[0], end=nidq_t_aligned[-1])

    #analog event index
    ts_list = []
    ts_idx = []
    if 'solenoid' in dinmap.keys():
        solenoid_din = dinmap['solenoid']
        solenoid_ts_list, solenoid_ts_idx = get_digital_ts(digital_events, map2imectime, time_support, solenoid_din, event_name='solenoid')
    elif 'solenoid' in ainmap.keys():
        solenoid_ts_list, solenoid_ts_idx = parse_solenoid_analog(analog_events, nidq_t_aligned, idx=ainmap, plot=plot)
    else:
        raise ValueError("Solenoid channel not found in either dinmap or ainmap.")
    ts_list += solenoid_ts_list
    ts_idx += solenoid_ts_idx

    if 'tone' in dinmap.keys():
        tones_ts_list, tones_ts_idx = parse_tones_digital(digital_events, solenoid_ts_list, map2imectime, time_support, idx=dinmap)
    elif 'tone' in ainmap.keys():
        tones_ts_list, tones_ts_idx = parse_tones_analog(analog_events, solenoid_ts_list, nidq_t_aligned, idx=ainmap)
    else:
        raise ValueError("Tone channel not found in either dinmap or ainmap.")
    ts_list += tones_ts_list
    ts_idx += tones_ts_idx

    if 'licks' in dinmap.keys():
        lick_din = dinmap['licks']
        lick_ts_list, lick_ts_idx = get_digital_ts(digital_events, map2imectime, time_support, lick_din)
    else:
        raise ValueError("Lick channel not found in dinmap.")

    ts_list += lick_ts_list
    ts_idx += lick_ts_idx

    if ('chrimson' in ainmap.keys()) or ('chr2' in ainmap.keys()):
        opto_ts_list, opto_ts_idx = get_opto_ts(analog_events, nidq_t_aligned, idx=ainmap, pwrs=optoidx, plot=plot)
        ts_list += opto_ts_list
        ts_idx += opto_ts_idx

    if ('chrimson' in dinmap.keys()):
        chrimson_ts_list, chrimson_ts_idx = get_digital_ts(digital_events, map2imectime, time_support, dinmap['chrimson'], event_name='chrimson')
        ts_list += chrimson_ts_list
        ts_idx += chrimson_ts_idx

    if ('chr2' in dinmap.keys()):
        chr2_ts_list, chr2_ts_idx = get_digital_ts(digital_events, map2imectime, time_support, dinmap['chr2'], event_name='chr2')
        ts_list += chr2_ts_list
        ts_idx += chr2_ts_idx

    ts_numbers = [i for i in range(len(ts_list))]
    ts_dict = dict(zip(ts_numbers, ts_list))
    ts_idx_dict = dict(zip(ts_numbers, ts_idx))
    ts_idx_df = pd.DataFrame.from_dict(ts_idx_dict, orient='index', columns=['event'])

    tsg = nap.TsGroup(ts_dict, metadata=ts_idx_df)

    metadata = tsg.metadata
    # get row of metadata where event is called "start tone on"
    start_tone_idx = metadata[metadata['event'] == 'start_tone_on'].index[0]
    timeout_tone_idx = metadata[metadata['event'] == 'timeout_tone_on'].index[0]
    error_tone_idx = metadata[metadata['event'] == 'error_tone_on'].index[0]
    reward_tone_idx = metadata[metadata['event'] == 'reward_tone_on'].index[0]
    lick_ts_idx = metadata[metadata['event'] == 'lick_on'].index[0]

    start_tone_ts = tsg[start_tone_idx]
    timeout_tone_ts = tsg[timeout_tone_idx]
    error_tone_ts = tsg[error_tone_idx]
    reward_tone_ts = tsg[reward_tone_idx]
    all_outcome_tone_ts = nap.Ts(np.sort(np.concatenate((reward_tone_ts.t, error_tone_ts.t, timeout_tone_ts.t))),
                                 time_support=time_support)
    lick_ts = tsg[lick_ts_idx]


    if len(all_outcome_tone_ts.t) != len(start_tone_ts.t):
        start_tone_ts, all_outcome_tone_ts = match_starts_and_ends(start_tone_ts, all_outcome_tone_ts)

    ITI_starts = nap.Ts(all_outcome_tone_ts.t[:-1] + 3.7)
    ITI_ends = nap.Ts(start_tone_ts.t[1:] - 1.5)

    reward_trials_starts, reward_trials_ends, rewarded_cues_ts, rewarded_first_licks_ts = parse_trials(start_tone_ts, reward_tone_ts, lick_ts)
    error_trials_starts, error_trials_ends, early_cues_ts, early_first_licks_ts = parse_trials(start_tone_ts, error_tone_ts, lick_ts)
    timeout_trials_starts, timeout_trials_ends, timeout_cues_ts, _  = parse_trials(start_tone_ts, timeout_tone_ts, lick_ts, require_first_licks=False)

    lick_bout_starts_ts, lick_bout_ends_ts = parse_lick_bouts(lick_ts)

    first_licks_metadata = pd.DataFrame.from_dict({
        1: 'rewarded_first_licks',
        2: 'early_first_licks',
        3: 'rewarded_cues',
        4: 'early_cues',
        5: 'start_cues',
        6: 'outcome_cues',
        7: 'ITI_starts',
        8: 'ITI_ends',
        9: 'lick_bout_starts',
        10: 'lick_bout_ends'}, orient='index', columns=['event'])
    first_licks_ts = nap.TsGroup({1: rewarded_first_licks_ts,
                                  2: early_first_licks_ts,
                                  3: rewarded_cues_ts,
                                  4: early_cues_ts,
                                  5: start_tone_ts,
                                  6: all_outcome_tone_ts,
                                  7: ITI_starts,
                                  8: ITI_ends,
                                  9: lick_bout_starts_ts,
                                  10: lick_bout_ends_ts
                                  },
                                 metadata=first_licks_metadata, time_support=time_support)




    # mege first_licks_ts into tsg
    tsg = tsg.merge(first_licks_ts, reset_index=True)
    tsg.save(str(pynapple_file))

    return tsg

def test():
    dinmap = None
    ainmap = None
    optoidx = None
    base_folder = None
    test_mode=False
    overwrite=True
    plot=True
    base_folder = str(r'C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250824')
    tsg = get_binary_signals(base_folder,dinmap,ainmap,optoidx,overwrite,plot,test_mode)