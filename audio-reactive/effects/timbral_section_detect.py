"""
Timbral section-change detection — shared between the streaming effect
and the offline viewer.

Detects section boundaries by tracking timbral *shape* drift using
L2-normalized MFCCs with an anchored-reference architecture:

  anchor EMA (240s TC) = "what section we're in" — barely moves within
                          a section, re-snapped to probe on commit
  probe  EMA (20s TC)  = "what is happening now" — tracks current timbre

Euclidean distance between anchor and probe on the unit sphere measures
how far the current timbral shape has drifted from the section reference,
independent of overall energy level.

A soft-fade blend variable accumulates when distance > threshold and rolls
back when it drops. Sustained divergence (blend > 0.9) commits a switch.
Transients roll back before committing.

The algorithm is CAUSAL — it only uses past frames, never future data.
Running it in batch on a pre-computed MFCC matrix produces identical
results to the streaming version.
"""

import numpy as np


def detect_timbral_sections(mfcc_matrix, fps,
                            anchor_tc=240.0, probe_tc=20.0,
                            eagerness_floor=1.05, eagerness_ceiling=2.0,
                            eagerness_tau=30.0,
                            dist_ema_rise_tc=120.0, dist_ema_fall_tc=15.0,
                            dead_zone_s=15.0,
                            warmup_s=25.0, fade_in_rate=0.03,
                            fade_out_rate=0.05, render_fps=30.0,
                            return_diagnostics=False):
    """Run timbral section detection on a pre-computed MFCC matrix.

    Parameters
    ----------
    mfcc_matrix : ndarray, shape (n_coeffs, n_frames)
        MFCC coefficients over time. Pass mfccs[0:13] to include MFCC 0
        (overall loudness) alongside shape coefficients 1-12. L2
        normalization makes MFCC 0 just one of 13 dimensions rather than
        the dominant signal, so energy-driven boundaries (EDM drops) are
        captured through MFCC 0 while shape-driven boundaries (ambient/
        dark electronic) are captured through MFCCs 1-12.
    fps : float
        Analysis frame rate (sr / hop_length).
    anchor_tc, probe_tc : float
        Time constants in seconds for the anchor and probe EMAs.
    eagerness_floor : float
        Minimum threshold (most eager, ratio to distance EMA baseline).
    eagerness_ceiling : float
        Maximum threshold (most reluctant, ratio to distance EMA baseline).
    eagerness_tau : float
        Exponential decay time constant in seconds.
    dist_ema_rise_tc : float
        Time constant (seconds) for EMA when distance is rising (slow,
        spike-resistant). Keeps the baseline low so real spikes stand out.
    dist_ema_fall_tc : float
        Time constant (seconds) for EMA when distance is falling (fast,
        tracks baseline drops).
    dead_zone_s : float
        Seconds of suppression after each commit.
    warmup_s : float
        Seconds before detection starts (first section = baseline).
    fade_in_rate, fade_out_rate : float
        Blend accumulation/rollback rates (per render frame at render_fps).
    render_fps : float
        Reference frame rate for fade rate scaling.
    return_diagnostics : bool
        If True, return a dict with transitions, distances, distances_norm,
        dist_ema, thresholds, and times arrays for visualization.

    Returns
    -------
    transitions : list of float
        Timestamps (in seconds) where section boundaries were detected.
        If return_diagnostics is True, returns a dict instead.
    """
    n_coeffs, n_frames = mfcc_matrix.shape

    anchor_alpha = 1.0 / (anchor_tc * fps)
    probe_alpha = 1.0 / (probe_tc * fps)
    dist_ema_rise_alpha = 1.0 / (dist_ema_rise_tc * fps)
    dist_ema_fall_alpha = 1.0 / (dist_ema_fall_tc * fps)
    fade_in = fade_in_rate * (render_fps / fps)
    fade_out = fade_out_rate * (render_fps / fps)
    dead_zone = int(dead_zone_s * fps)
    warmup = int(warmup_s * fps)
    eagerness_range = eagerness_ceiling - eagerness_floor

    # Initialize from first frame
    x0 = mfcc_matrix[:, 0].astype(np.float64)
    x0_norm = np.linalg.norm(x0)
    x0_unit = x0 / (x0_norm + 1e-10)

    anchor = x0_unit.copy()
    probe = x0_unit.copy()

    blend = 0.0
    dist_ema = 0.0
    warmup_dist_sum = 0.0
    warmup_dist_count = 0
    frames_since_commit = 0
    need_ema_reseed = False
    transitions = []

    if return_diagnostics:
        distances_raw = np.zeros(n_frames, dtype=np.float64)
        distances_norm = np.zeros(n_frames, dtype=np.float64)
        dist_ema_trace = np.zeros(n_frames, dtype=np.float64)
        thresholds = np.zeros(n_frames, dtype=np.float64)
        blend_trace = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        x = mfcc_matrix[:, i].astype(np.float64)
        x_norm = np.linalg.norm(x)
        x_unit = x / (x_norm + 1e-10)

        anchor += anchor_alpha * (x_unit - anchor)
        probe += probe_alpha * (x_unit - probe)

        diff = anchor - probe
        shape_dist = np.sqrt(np.dot(diff, diff))

        frames_since_commit += 1
        in_dead = frames_since_commit < dead_zone

        # Accumulate warmup distances to seed EMA
        if i > 0 and i < warmup:
            warmup_dist_sum += shape_dist
            warmup_dist_count += 1
        if i == warmup and warmup_dist_count > 0:
            dist_ema = warmup_dist_sum / warmup_dist_count
            frames_since_commit = 0  # enter dead zone at warmup end

        # Re-seed EMA when dead zone ends after a commit, so the EMA
        # reflects the new section's baseline rather than the old one.
        if need_ema_reseed and not in_dead:
            dist_ema = shape_dist
            need_ema_reseed = False

        # Asymmetric EMA normalization: slow rise (spike-resistant),
        # fast fall (tracks baseline). Frozen during dead zone and warmup.
        if not in_dead and i >= warmup:
            alpha = dist_ema_rise_alpha if shape_dist > dist_ema else dist_ema_fall_alpha
            dist_ema += alpha * (shape_dist - dist_ema)
        shape_dist_norm = shape_dist / (dist_ema + 1e-10)

        # Eagerness curve: threshold decays from ceiling to floor
        time_since_commit = frames_since_commit / fps
        threshold = eagerness_floor + eagerness_range * np.exp(-time_since_commit / eagerness_tau)

        if return_diagnostics:
            distances_raw[i] = shape_dist
            distances_norm[i] = shape_dist_norm
            dist_ema_trace[i] = dist_ema
            thresholds[i] = threshold
            blend_trace[i] = blend

        if not in_dead and i >= warmup:
            if shape_dist_norm > threshold:
                blend += fade_in
            else:
                blend -= fade_out
            blend = max(0.0, min(1.0, blend))
        else:
            blend = max(0.0, blend - fade_out)

        if blend > 0.9:
            t = i / fps
            transitions.append(t)
            anchor[:] = probe
            blend = 0.0
            frames_since_commit = 0
            need_ema_reseed = True

    if return_diagnostics:
        times = np.arange(n_frames) / fps
        return {
            'transitions': transitions,
            'distances': distances_raw,
            'distances_norm': distances_norm,
            'dist_ema': dist_ema_trace,
            'thresholds': thresholds,
            'blend': blend_trace,
            'times': times,
        }
    return transitions


def detect_simple_eagerness(mfcc_matrix, fps, l2_normalize=True,
                            fast_tc=5.0, slow_tc=45.0,
                            cooldown_s=15.0,
                            sigma_ceiling=3.0, sigma_floor=1.0,
                            eagerness_tau=150.0):
    """Dual-EMA novelty detector with eagerness.

    Two EMAs: a fast one (5s) that tracks the current timbre, and a slow one
    (45s) that holds the section reference. Detection compares the current
    frame against the SLOW EMA, so the reference doesn't absorb changes
    before eagerness can act. The fast EMA is available for rendering.

    Threshold: dist > dist_ema + N * dist_std, where N decays linearly
    from sigma_ceiling to sigma_floor over eagerness_tau seconds.

    Parameters
    ----------
    mfcc_matrix : ndarray, shape (n_coeffs, n_frames)
    fps : float
    l2_normalize : bool
        If True, L2-normalize each frame (shape only). If False, use raw MFCCs.
    fast_tc : float
        Fast EMA time constant (seconds) — tracks current timbre.
    slow_tc : float
        Slow EMA time constant (seconds) — holds section reference for detection.
    cooldown_s : float
        Minimum seconds between switches.
    sigma_ceiling : float
        N multiplier right after a switch (reluctant).
    sigma_floor : float
        N multiplier after eagerness_tau has elapsed (eager).
    eagerness_tau : float
        Seconds for threshold to ramp from ceiling to floor.
    """
    n_coeffs, n_frames = mfcc_matrix.shape
    fast_alpha = 1.0 / (fast_tc * fps)
    slow_alpha = 1.0 / (slow_tc * fps)
    cooldown = int(cooldown_s * fps)
    sigma_range = sigma_ceiling - sigma_floor

    # Initialize from first frame
    x0 = mfcc_matrix[:, 0].astype(np.float64)
    if l2_normalize:
        x0 = x0 / (np.linalg.norm(x0) + 1e-10)

    fast_ema = x0.copy()
    slow_ema = x0.copy()
    dist_ema = 0.0
    dist_var_ema = 0.0
    dist_alpha = 1.0 / (fast_tc * fps)

    frames_since_switch = cooldown + 1  # start past cooldown
    transitions = []

    for i in range(n_frames):
        x = mfcc_matrix[:, i].astype(np.float64)
        if l2_normalize:
            x = x / (np.linalg.norm(x) + 1e-10)

        # Update both EMAs
        fast_ema += fast_alpha * (x - fast_ema)
        slow_ema += slow_alpha * (x - slow_ema)

        # Distance: current frame vs SLOW EMA (holds the section reference)
        diff = x - slow_ema
        dist = np.sqrt(np.dot(diff, diff))

        # Update distance stats (using fast alpha so stats track recent behavior)
        dist_ema += dist_alpha * (dist - dist_ema)
        dist_var_ema += dist_alpha * ((dist - dist_ema) ** 2 - dist_var_ema)
        dist_std = np.sqrt(max(dist_var_ema, 1e-20))

        frames_since_switch += 1

        # Eagerness: p^1.5 decay (gentle curve, eases down ~30-45s)
        t = frames_since_switch / fps
        progress = min(t / eagerness_tau, 1.0)
        N = sigma_ceiling - sigma_range * (progress ** 1.5)

        # Trigger check
        if frames_since_switch > cooldown and dist > dist_ema + N * dist_std:
            transitions.append(i / fps)
            frames_since_switch = 0
            # Snap slow EMA to current state so it references the new section
            slow_ema[:] = fast_ema

    return transitions


def detect_log_eagerness(mfcc_matrix, fps, l2_normalize=False,
                         ema_tc=5.0, cooldown_s=15.0,
                         sigma_ceiling=3.0, sigma_floor=1.0,
                         eagerness_tau=120.0, power=3.0):
    """Single-EMA novelty detector with power-curve eagerness decay.

    Threshold stays near ceiling then drops at rate controlled by power.
    power=3 (cubic): very patient. power=1: linear. power=0.5: eager early.
    """
    n_coeffs, n_frames = mfcc_matrix.shape
    alpha = 1.0 / (ema_tc * fps)
    cooldown = int(cooldown_s * fps)
    sigma_range = sigma_ceiling - sigma_floor

    x0 = mfcc_matrix[:, 0].astype(np.float64)
    if l2_normalize:
        x0 = x0 / (np.linalg.norm(x0) + 1e-10)

    ema = x0.copy()
    dist_ema = 0.0
    dist_var_ema = 0.0
    dist_alpha = 1.0 / (ema_tc * fps)

    frames_since_switch = cooldown + 1
    transitions = []

    for i in range(n_frames):
        x = mfcc_matrix[:, i].astype(np.float64)
        if l2_normalize:
            x = x / (np.linalg.norm(x) + 1e-10)

        ema += alpha * (x - ema)

        diff = x - ema
        dist = np.sqrt(np.dot(diff, diff))

        dist_ema += dist_alpha * (dist - dist_ema)
        dist_var_ema += dist_alpha * ((dist - dist_ema) ** 2 - dist_var_ema)
        dist_std = np.sqrt(max(dist_var_ema, 1e-20))

        frames_since_switch += 1

        # Cubic decay: stays near ceiling, drops sharply near tau
        t = frames_since_switch / fps
        progress = min(t / eagerness_tau, 1.0)
        N = sigma_ceiling - sigma_range * (progress ** power)

        if frames_since_switch > cooldown and dist > dist_ema + N * dist_std:
            transitions.append(i / fps)
            frames_since_switch = 0

    return transitions


def detect_bucket_novelty(mfcc_matrix, fps, l2_normalize=True,
                          n_buckets=10, bucket_tc=2.0,
                          cooldown_s=15.0, novelty_threshold=0.8,
                          eagerness_tau=120.0):
    """Bucket-based novelty detector: track N recent timbral states,
    fire when current timbre doesn't fit any of them.

    Maintains N EMA-smoothed MFCC centroids representing the distinct
    timbral "states" heard in the current section. Each new frame is
    compared to all buckets -- minimum distance is the novelty score.
    When the best-matching bucket is still far away, we're hearing
    something genuinely new.

    Handles repeated patterns: ABCDABCD fills buckets for A,B,C,D and
    never fires. When E arrives, it doesn't match any bucket -> fire.
    """
    n_coeffs, n_frames = mfcc_matrix.shape
    cooldown = int(cooldown_s * fps)
    bucket_alpha = 1.0 / (bucket_tc * fps)
    threshold_floor = novelty_threshold * 0.3

    def norm_frame(x):
        x = x.astype(np.float64)
        if l2_normalize:
            x = x / (np.linalg.norm(x) + 1e-10)
        return x

    # Initialize all buckets to the first frame
    x0 = norm_frame(mfcc_matrix[:, 0])
    buckets = np.tile(x0, (n_buckets, 1))
    bucket_age = np.zeros(n_buckets, dtype=np.float64)

    frames_since_switch = cooldown + 1
    warmup = int(20.0 * fps)
    transitions = []

    for i in range(n_frames):
        x = norm_frame(mfcc_matrix[:, i])

        # Distance to each bucket
        diffs = buckets - x[np.newaxis, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        best_idx = np.argmin(dists)
        min_dist = dists[best_idx]

        # Update best-matching bucket (EMA toward current frame)
        buckets[best_idx] += bucket_alpha * (x - buckets[best_idx])
        bucket_age[best_idx] = 0
        bucket_age += 1

        frames_since_switch += 1

        # Eagerness: threshold ramps down linearly
        t = frames_since_switch / fps
        threshold = novelty_threshold - (novelty_threshold - threshold_floor) * min(t / eagerness_tau, 1.0)

        if frames_since_switch > cooldown and i >= warmup and min_dist > threshold:
            transitions.append(i / fps)
            frames_since_switch = 0
            # Replace oldest bucket with the new state
            oldest_idx = np.argmax(bucket_age)
            buckets[oldest_idx] = x.copy()
            bucket_age[oldest_idx] = 0

    return transitions


def detect_bucket_novelty_v2(mfcc_matrix, fps, l2_normalize=True,
                              max_entries=16, entry_tc=3.0,
                              cooldown_s=15.0, warmup_s=20.0,
                              admit_mult=2.5, fire_mult=2.0,
                              eagerness_tau=90.0,
                              dist_ema_tc=10.0,
                              maturity_hits=None):
    """Adaptive-codebook timbral novelty detector (v2).

    Instead of pre-allocating N buckets, grows a codebook on demand:
    - Starts empty. First frame becomes entry #0.
    - Each frame finds the nearest codebook entry. If the distance
      exceeds admit_mult * dist_ema (the running baseline distance),
      a new entry is created from the current frame.
    - While the codebook is still growing (learning phase), new entries
      are added silently — no transition fired.
    - Once the codebook has been stable for a while (no new entries
      for cooldown_s seconds), it's considered "mature". After that,
      any frame that forces a new entry fires a transition — the timbre
      doesn't fit the learned vocabulary.
    - Codebook entries are EMA-updated toward matching frames, but the
      alpha decays with hit count so well-established entries become
      anchors that resist drift.
    - When the codebook is full (max_entries), the least-used entry
      is evicted to make room.

    Parameters
    ----------
    mfcc_matrix : ndarray, shape (n_coeffs, n_frames)
    fps : float
    l2_normalize : bool
    max_entries : int
        Maximum codebook size. Keeps memory bounded.
    entry_tc : float
        Base EMA time constant (seconds) for updating entries.
        Effective alpha = base_alpha / (1 + hits/maturity_hits),
        so entries stiffen as they accumulate matches.
    cooldown_s : float
        Minimum seconds between transitions AND the quiet period
        required before the codebook is considered mature.
    warmup_s : float
        Seconds before detection starts (build initial codebook).
    admit_mult : float
        Multiplier on dist_ema to decide "this frame is far enough
        from everything to deserve its own entry".
    fire_mult : float
        Multiplier on dist_ema for the transition threshold. Only
        used post-maturity. Decays toward a floor via eagerness_tau.
    eagerness_tau : float
        Seconds for threshold to decay from fire_mult toward floor.
    dist_ema_tc : float
        Time constant for the running distance baseline EMA.
    maturity_hits : int or None
        Number of matches before an entry's EMA alpha halves.
        Defaults to int(5 * fps) — roughly 5 seconds of continuous
        matching.
    """
    n_coeffs, n_frames = mfcc_matrix.shape
    cooldown = int(cooldown_s * fps)
    warmup = int(warmup_s * fps)
    base_alpha = 1.0 / (entry_tc * fps)
    dist_alpha = 1.0 / (dist_ema_tc * fps)
    if maturity_hits is None:
        maturity_hits = int(5.0 * fps)
    fire_floor = fire_mult * 0.4  # eagerness floor

    def norm_frame(x):
        x = x.astype(np.float64)
        if l2_normalize:
            x = x / (np.linalg.norm(x) + 1e-10)
        return x

    # Codebook storage — pre-allocate max, track how many are live
    codebook = np.zeros((max_entries, n_coeffs), dtype=np.float64)
    hits = np.zeros(max_entries, dtype=np.int64)      # match count per entry
    last_hit = np.zeros(max_entries, dtype=np.int64)   # frame of last match
    n_entries = 0

    # Add first frame
    x0 = norm_frame(mfcc_matrix[:, 0])
    codebook[0] = x0
    hits[0] = 1
    last_hit[0] = 0
    n_entries = 1

    dist_ema = 0.0
    frames_since_switch = cooldown + 1
    frames_since_new_entry = 0  # tracks codebook stability
    transitions = []

    for i in range(1, n_frames):
        x = norm_frame(mfcc_matrix[:, i])

        # Find nearest entry (scalar loop mirrors ESP32 implementation)
        min_dist = np.inf
        best_idx = 0
        for j in range(n_entries):
            d = 0.0
            for k in range(n_coeffs):
                diff_k = codebook[j, k] - x[k]
                d += diff_k * diff_k
            d = np.sqrt(d)
            if d < min_dist:
                min_dist = d
                best_idx = j

        # Update distance baseline EMA (always, including warmup)
        dist_ema += dist_alpha * (min_dist - dist_ema)

        # Admission threshold: is this frame far enough from everything?
        admit_thresh = admit_mult * (dist_ema + 1e-10)
        is_novel = min_dist > admit_thresh

        frames_since_switch += 1
        frames_since_new_entry += 1

        if is_novel:
            # This frame doesn't fit the codebook
            codebook_mature = (i >= warmup and
                               frames_since_new_entry > cooldown)

            if codebook_mature and frames_since_switch > cooldown:
                # Post-maturity: fire a transition
                # Eagerness: threshold decays from fire_mult toward floor
                t = frames_since_switch / fps
                threshold = fire_floor + (fire_mult - fire_floor) * np.exp(-t / eagerness_tau)
                effective_thresh = threshold * (dist_ema + 1e-10)

                if min_dist > effective_thresh:
                    transitions.append(i / fps)
                    frames_since_switch = 0

            # Either way, add (or replace) an entry for this new state
            if n_entries < max_entries:
                codebook[n_entries] = x.copy()
                hits[n_entries] = 1
                last_hit[n_entries] = i
                n_entries += 1
            else:
                # Evict least-used entry
                worst = 0
                worst_hits = hits[0]
                for j in range(1, n_entries):
                    if hits[j] < worst_hits:
                        worst = j
                        worst_hits = hits[j]
                codebook[worst] = x.copy()
                hits[worst] = 1
                last_hit[worst] = i
            frames_since_new_entry = 0
        else:
            # Update best-matching entry — alpha decays with maturity
            effective_alpha = base_alpha / (1.0 + hits[best_idx] / maturity_hits)
            codebook[best_idx] += effective_alpha * (x - codebook[best_idx])
            hits[best_idx] += 1
            last_hit[best_idx] = i

    return transitions


def detect_core_candidate(mfcc_matrix, fps, l2_normalize=True,
                          n_core=3, n_candidates=3,
                          ema_tc=3.0, cooldown_s=15.0, warmup_s=20.0,
                          admit_mult=2.0, dist_ema_tc=10.0):
    """Core+candidate bucket detector: fire when multiple candidates activate.

    3 core buckets = the current section's timbral vocabulary.
    3 candidate buckets = testing ground for new states.

    A frame that doesn't match any core bucket goes to a candidate slot.
    When 2+ candidate slots are active (have been hit recently), the
    section vocabulary is changing → fire a transition.

    On fire: promote the active candidates to core, reset candidates.
    """
    n_coeffs, n_frames = mfcc_matrix.shape
    cooldown = int(cooldown_s * fps)
    warmup = int(warmup_s * fps)
    base_alpha = 1.0 / (ema_tc * fps)
    dist_alpha = 1.0 / (dist_ema_tc * fps)
    candidate_window = int(5.0 * fps)  # candidate is "active" if hit within 5s

    def norm_frame(x):
        x = x.astype(np.float64)
        if l2_normalize:
            x = x / (np.linalg.norm(x) + 1e-10)
        return x

    # Core buckets
    x0 = norm_frame(mfcc_matrix[:, 0])
    core = np.tile(x0, (n_core, 1))
    core_hits = np.zeros(n_core, dtype=np.int64)
    core_hits[0] = 1
    n_core_live = 1  # how many core slots are actually filled

    # Candidate buckets
    candidates = np.zeros((n_candidates, n_coeffs), dtype=np.float64)
    cand_last_hit = np.full(n_candidates, -candidate_window * 2, dtype=np.int64)
    n_cand_live = 0

    dist_ema = 0.0
    frames_since_switch = cooldown + 1
    transitions = []

    for i in range(1, n_frames):
        x = norm_frame(mfcc_matrix[:, i])

        # Find nearest core bucket
        min_core_dist = np.inf
        best_core = 0
        for j in range(n_core_live):
            d = np.sqrt(np.sum((core[j] - x) ** 2))
            if d < min_core_dist:
                min_core_dist = d
                best_core = j

        dist_ema += dist_alpha * (min_core_dist - dist_ema)
        admit_thresh = admit_mult * (dist_ema + 1e-10)

        frames_since_switch += 1

        if min_core_dist <= admit_thresh:
            # Matches a core bucket — update it
            core[best_core] += base_alpha * (x - core[best_core])
            core_hits[best_core] += 1
        else:
            # Doesn't match core — check candidates
            min_cand_dist = np.inf
            best_cand = -1
            for j in range(n_cand_live):
                d = np.sqrt(np.sum((candidates[j] - x) ** 2))
                if d < min_cand_dist:
                    min_cand_dist = d
                    best_cand = j

            if n_cand_live > 0 and min_cand_dist <= admit_thresh:
                # Matches an existing candidate
                candidates[best_cand] += base_alpha * (x - candidates[best_cand])
                cand_last_hit[best_cand] = i
            elif n_cand_live < n_candidates:
                # New candidate
                candidates[n_cand_live] = x.copy()
                cand_last_hit[n_cand_live] = i
                n_cand_live += 1
            else:
                # All candidate slots full — replace oldest
                oldest = 0
                for j in range(1, n_cand_live):
                    if cand_last_hit[j] < cand_last_hit[oldest]:
                        oldest = j
                candidates[oldest] = x.copy()
                cand_last_hit[oldest] = i

            # Count active candidates (hit within the window)
            active_count = 0
            for j in range(n_cand_live):
                if (i - cand_last_hit[j]) < candidate_window:
                    active_count += 1

            # Fire if 2+ candidates are active (vocabulary is shifting)
            if (active_count >= 2 and i >= warmup and
                    frames_since_switch > cooldown):
                transitions.append(i / fps)
                frames_since_switch = 0

                # Promote active candidates to core (replace least-used cores)
                promoted = 0
                for j in range(n_cand_live):
                    if (i - cand_last_hit[j]) < candidate_window and promoted < n_core:
                        # Replace the least-hit core
                        worst_core = 0
                        for k in range(1, n_core_live):
                            if core_hits[k] < core_hits[worst_core]:
                                worst_core = k
                        if n_core_live < n_core:
                            core[n_core_live] = candidates[j].copy()
                            core_hits[n_core_live] = 1
                            n_core_live += 1
                        else:
                            core[worst_core] = candidates[j].copy()
                            core_hits[worst_core] = 1
                        promoted += 1

                # Reset candidates
                n_cand_live = 0
                cand_last_hit[:] = -candidate_window * 2

    return transitions
