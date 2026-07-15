_EPSILON = 1e-6

def bpm_to_spb(bpm):
    return 60.0 / bpm

def calc_segment_lengths(bpms):
    """
Calculates the duration (in seconds) of each time segment between consecutive BPM change points.
Converts beat-based segments to time-based durations for accurate mapping of beat positions to real-world time.
Essential for timing synchronization and further calculations involving absolute time in music analysis.
"""

    assert len(bpms) > 0
    segment_lengths = []
    for i in range(len(bpms) - 1):
        spb = bpm_to_spb(bpms[i][1])
        segment_lengths.append(spb * (bpms[i + 1][0] - bpms[i][0]))
    return segment_lengths

def calc_abs_for_beat(offset, bpms, stops, segment_lengths, beat):
    """
Calculates the absolute time (in seconds) for a specific beat in a musical piece, accounting for BPM changes, stops, and an initial time offset.
Iterates through the BPM change points to find the relevant segment and computes the cumulative time up to that beat, including adjustments for stops.
Returns the absolute time in seconds, facilitating precise synchronization of beats and events in music.
"""
    bpm_idx = 0
    while bpm_idx < len(bpms) and beat + _EPSILON > bpms[bpm_idx][0]:
        bpm_idx += 1
    bpm_idx -= 1

    stop_len_cumulative = 0.0
    for stop_beat, stop_len in stops:
        diff = beat - stop_beat
        # We are at this stop which should not count to its timing
        if abs(diff) < _EPSILON:
            break
        # We are before this stop so we don't count it
        elif diff < 0:
            break
        # We are after this stop so count it
        else:
            stop_len_cumulative += stop_len
    """
    full_segment_total: The total time up to the start of the current BPM segment (before beat).
    partial_segment_spb: The duration of each beat in the current segment.
    partial_segment: The time from the start of the current BPM segment to the specified beat.
    """

    full_segment_total = sum(segment_lengths[:bpm_idx])
    partial_segment_spb = bpm_to_spb(bpms[bpm_idx][1])
    partial_segment = partial_segment_spb * (beat - bpms[bpm_idx][0])

    return full_segment_total + partial_segment - offset + stop_len_cumulative

def calc_note_beats_and_abs_times(offset, bpms, stops, note_data):
    segment_lengths = calc_segment_lengths(bpms)

    # copy bpms
    bpms = bpms[:]
    inc = None
    inc_prev = None
    time = offset

    # beat loop
    note_beats_abs_times = []
    beat_times = []
    for measure_num, measure in enumerate(note_data):
        ppm = len(measure)
        for i, code in enumerate(measure):
            beat = measure_num * 4.0 + 4.0 * (float(i) / ppm)
            # TODO: This could be much more efficient but is not the bottleneck for the moment.
            beat_abs = calc_abs_for_beat(offset, bpms, stops, segment_lengths, beat)
            note_beats_abs_times.append(((measure_num, ppm, i), beat, beat_abs, code))
            beat_times.append(beat_abs)

    # handle negative stops
    beat_time_prev = float('-inf')
    del_idxs = []
    for i, beat_time in enumerate(beat_times):
        if beat_time_prev > beat_time:
            del_idxs.append(i)
        else:
            beat_time_prev = beat_time
    for del_idx in sorted(del_idxs, reverse=True):
        del note_beats_abs_times[del_idx]
        del beat_times[del_idx]

    #TODO: remove when stable
    assert sorted(beat_times) == beat_times

    return note_beats_abs_times
