import os
import pickle
import numpy as np
from essentia.standard import MonoLoader, FrameGenerator, Windowing, Spectrum, MelBands, RhythmExtractor2013
import essentia.standard as es
import math
from typing import List, Tuple, Dict
from tensorflow.keras.metrics import Metric
import tensorflow as tf

def set_bpm(audio_fp, min_tempo = 100, max_tempo = 200, maxstep = 12, bpm_method = 'DDCL'):
    if bpm_method == 'DDCL':
        beat_dict = extract_BPM(audio_fp, min_tempo, max_tempo)
        bpm = beat_dict['BPM']
        beats = beat_dict['beats']

        beat_intervals = list(beat_dict['beat_intervals'])
        bpm_intervals = [60/gap for gap in beat_intervals]

        avg_point = 0
        bpm_shift_times = []
        first_line = True
        for i in range(1,len(bpm_intervals)):
            cur_bpm = bpm_intervals[i]
            avg_bpm = 60/np.mean(beat_intervals[avg_point:i])
            if abs(cur_bpm-avg_bpm)>25 or i == len(bpm_intervals)-1:
                if first_line:
                    bpm_str = '0.0={}'.format(avg_bpm)
                    first_line=False
                else:
                    bpm_str += '\n'+',{}={}'.format(avg_point, avg_bpm)
                bpm_shift_times.append((avg_point, avg_bpm))
                beats[avg_point:i] = np.linspace(beats[avg_point], beats[i], endpoint = False, num = i-avg_point)
                avg_point = i
                
        if first_line:
            bpm_str = '0.0={}'.format(avg_bpm)

        print('bpm shifts: {}'.format(bpm_shift_times))

        offset = beats[0]

        song_length = beats[-1]/60

        subdiv_beats = []
        for j in range(len(beats)-1):
            for i in range(maxstep):
                subdiv_beats.append(((beats[j]*(maxstep-i))/maxstep)+(beats[j+1]*i)/maxstep)
        subdiv_beats.append(beats[-1])

        print('offset: {}'.format(offset))

        return beats, np.array(subdiv_beats), bpm_shift_times, offset, bpm_str, song_length, bpm
    elif bpm_method == 'AV':
        beat_dict = arrow_vortex_get_bpm(audio_fp, bpm_range = (min_tempo, max_tempo))
        bpm = beat_dict[0]['bpm']
        offset = beat_dict[0]['offset']
        beats = beat_dict[0]['beats']
        bpm_shift_times = [(0,bpm)]
        subdiv_beats = []
        for j in range(len(beats)-1):
            for i in range(maxstep):
                subdiv_beats.append(((beats[j]*(maxstep-i))/maxstep)+(beats[j+1]*i)/maxstep)
        subdiv_beats.append(beats[-1])
        bpm_str = '0.0={}'.format(bpm)
        song_length = beats[-1]/60
        return beats, np.array(subdiv_beats), bpm_shift_times, offset, bpm_str, song_length, bpm
    elif bpm_method == 'SMEdit':
        beat_dict = smedit_analyze_audio(audio_fp, bpm_range = (min_tempo, max_tempo))
        bpm = beat_dict['bpm_results'][0]['bpm']
        offset = beat_dict['offset_results'][0]['offset']
        beats = beat_dict['beat_times'][bpm]
        bpm_shift_times = [(0,bpm)]
        subdiv_beats = []
        for j in range(len(beats)-1):
            for i in range(maxstep):
                subdiv_beats.append(((beats[j]*(maxstep-i))/maxstep)+(beats[j+1]*i)/maxstep)
        subdiv_beats.append(beats[-1])
        bpm_str = '0.0={}'.format(bpm)
        song_length = beats[-1]/60
        return beats, np.array(subdiv_beats), bpm_shift_times, offset, bpm_str, song_length, bpm

def weighted_median(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0

def hamming_window(w):
    return [0.54 - 0.46 * math.cos(2 * math.pi * n / (w - 1)) for n in range(w)]

def arrow_vortex_get_bpm(audio_file, 
                        window_size=1024, 
                        hop_size=256,
                        bpm_range=(89, 205),
                        silence_threshold=-70,
                        threshold_weight=0.1,
                        threshold_window_size=7):
    """
    This is the algorithm described in Bram van de Wetering's Non-Causal Beat Tracking for Rhythm Games. 
    Essentia is used but it would be easy to use Librosa too.
    """
    #load audio
    loader = es.AudioLoader(filename=audio_file)
    audio, sample_rate, _, _, _, _ = loader()
    
    #stereo to mono
    if len(audio.shape) > 1:
        audio = es.MonoMixer()(audio, 1)  # Mix to mono
    
    #PHASE 1: ONSET EXTRACTION
    #phase vocoder for spectral flux
    windowing = es.Windowing(type='hann', size=window_size)
    fft = es.FFT(size=window_size)
    magnitude = es.Magnitude()
    spectral_flux = es.Flux()
    
    #compute spectral flux
    detection_function = []
    
    for frame in es.FrameGenerator(audio, frameSize=window_size, hopSize=hop_size):
        windowed_frame = windowing(frame)
        spectrum = fft(windowed_frame)
        mag_spectrum = magnitude(spectrum)
        flux = spectral_flux(mag_spectrum)
        detection_function.append(flux)
    
    detection_function = np.array(detection_function)
    
    #thresholding
    thresholds = []
    
    for n in range(len(detection_function)):
        start = max(0, n - 5)
        end = min(len(detection_function), n + 1)
        window_data = detection_function[start:end]
        
        threshold = weighted_median(window_data) + threshold_weight * np.mean(window_data)
        thresholds.append(threshold)
    
    thresholds = np.array(thresholds)

    onsets = []
    for i in range(1, len(detection_function) - 1):
        if (detection_function[i] > thresholds[i] and 
            detection_function[i] > detection_function[i-1] and
            detection_function[i] > detection_function[i+1]):
            onsets.append(i)
    
    #silence gate
    windowing = es.Windowing(type='hann', size=window_size)
    spectrum = es.Spectrum(size=window_size)
    frame_gen = es.FrameGenerator

    filtered_onsets = []

    for onset in onsets:
        frame_start = onset * hop_size
        frame_end = min(frame_start + window_size, len(audio))
        frame_audio = audio[frame_start:frame_end]

        if len(frame_audio) == 0:
            continue

        if len(frame_audio) < window_size:
            frame_audio = np.pad(frame_audio, (0, window_size - len(frame_audio)), mode='constant')

        stft_frames = frame_gen(frame_audio, frameSize=window_size, hopSize=hop_size, startFromZero=True)

        mean_energy = np.mean([
            np.sum(spectrum(windowing(frame))**2)
            for frame in stft_frames
        ])

        mean_energy_db = 10 * np.log10(max(mean_energy, 1e-10))

        if mean_energy_db > silence_threshold:
            filtered_onsets.append(onset)

    onsets = np.array(filtered_onsets)
    
    #convert onsets to time
    onset_times = onsets * hop_size / sample_rate
    last_onset_time = onset_times[-1] if len(onset_times) > 0 else 0
    
    #PHASE 2: BPM DETECTION
    #BPM range to interval range
    frame_rate = sample_rate / hop_size
    i_min = int(frame_rate * 60 / bpm_range[1])
    i_max = int(frame_rate * 60 / bpm_range[0])
    
    
    #test every 10th interval
    test_intervals = range(i_min, i_max + 1, 10)
    fitness_scores = {}
    
    for interval in test_intervals:
        #histogram for onset positions mod interval
        histogram = np.zeros(interval)
        for onset in onsets:
            bin_idx = onset % interval
            histogram[bin_idx] += 1
        
        #evidence function using Hamming window
        hamming_size = min(interval // 4, 10)
        hamming_weights = hamming_window(hamming_size)
        
        evidence = np.zeros(interval)
        for p in range(interval):
            for n in range(hamming_size):
                bin_idx = int((p - hamming_size // 2 + n) % interval)
                if n < len(hamming_weights):
                    evidence[p] += hamming_weights[n] * histogram[bin_idx]
        
        #confidence function
        confidence = np.zeros(interval)
        for p in range(interval):
            conf1 = evidence[p]
            conf2 = evidence[int((p + interval // 2) % interval)]
            confidence[p] = conf1 + conf2 / 2
        
        fitness_scores[interval] = np.max(confidence)
    
    if not fitness_scores:
        print("BPM detection has failed, check max and min tempos, and also whether that was really a song you put in.")
        return [], [], {}
    
    #fit third-degree polynomial to fitness values
    intervals = np.array(list(fitness_scores.keys()))
    fitness_values = np.array(list(fitness_scores.values()))
    
    if len(intervals) >= 4:
        poly_coeffs = np.polyfit(intervals, fitness_values, 3)
        poly_func = np.poly1d(poly_coeffs)
        
        for interval in fitness_scores.keys():
            bias_estimate = poly_func(interval)
            fitness_scores[interval] = fitness_scores[interval] - bias_estimate
    
    #candidates after polynomial fitting
    max_fitness = max(fitness_scores.values())
    threshold_fitness = 0.4 * max_fitness
    
    candidates = []
    for interval, fitness in fitness_scores.items():
        if fitness > threshold_fitness:
            #interval to BPM
            bpm = frame_rate * 60 / interval
            candidates.append((bpm, fitness, interval))
    
    #sort by fitness
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    #remove similar candidates
    filtered_candidates = []
    for bpm, fitness, interval in candidates:
        is_unique = True
        for existing_bpm, _, _ in filtered_candidates:
            if abs(bpm - existing_bpm) < 0.1:
                is_unique = False
                break
        if is_unique:
            filtered_candidates.append((bpm, fitness, interval))
    
    candidates = filtered_candidates[:5]
    
    #PHASE 3: OFFSET DETECTION
    final_results = []
    
    for bpm, fitness, interval in candidates:
        histogram = np.zeros(interval)
        for onset in onsets:
            bin_idx = onset % interval
            histogram[bin_idx] += 1

        hamming_size = min(interval // 4, 10)
        hamming_weights = hamming_window(hamming_size)
        
        confidence = np.zeros(interval)
        for p in range(interval):
            evidence = 0
            for n in range(hamming_size):
                bin_idx = int((p - hamming_size // 2 + n) % interval)
                if n < len(hamming_weights):
                    evidence += hamming_weights[n] * histogram[bin_idx]
            
            conf2 = 0
            for n in range(hamming_size):
                bin_idx = int((p + interval // 2 - hamming_size // 2 + n) % interval)
                if n < len(hamming_weights):
                    conf2 += hamming_weights[n] * histogram[bin_idx]
            
            confidence[p] = evidence + conf2 / 2
        
        p_max = np.argmax(confidence)
        
        #initial offset in seconds
        offset_frames = p_max
        offset_seconds = offset_frames * hop_size / sample_rate

        interval_seconds = interval * hop_size / sample_rate
        window_samples = int(0.05 * sample_rate)
        
        beat_slopes = []
        offbeat_slopes = []
        
        current_time = offset_seconds
        while current_time < len(audio) / sample_rate - 0.1:
            beat_sample = int(current_time * sample_rate)
            offbeat_sample = int((current_time + interval_seconds / 2) * sample_rate)
            
            if beat_sample + window_samples < len(audio) and beat_sample - window_samples >= 0:
                forward_sum = sum(abs(audio[beat_sample + n]) for n in range(1, window_samples + 1))
                backward_sum = sum(abs(audio[beat_sample - window_samples + n]) for n in range(1, window_samples + 1))
                beat_slope = max(forward_sum - backward_sum, 0)
                beat_slopes.append(beat_slope)

            if offbeat_sample + window_samples < len(audio) and offbeat_sample - window_samples >= 0:
                forward_sum = sum(abs(audio[offbeat_sample + n]) for n in range(1, window_samples + 1))
                backward_sum = sum(abs(audio[offbeat_sample - window_samples + n]) for n in range(1, window_samples + 1))
                offbeat_slope = max(forward_sum - backward_sum, 0)
                offbeat_slopes.append(offbeat_slope)
            
            current_time += interval_seconds

        if beat_slopes and offbeat_slopes:
            avg_beat = np.mean(beat_slopes)
            avg_offbeat = np.mean(offbeat_slopes)
            
            if avg_offbeat > avg_beat:
                offset_seconds += interval_seconds / 2
        
        #beat timings to make my life easier
        beat_timings = []
        beat_interval_seconds = 60.0 / bpm
 
        current_beat_time = offset_seconds

        while current_beat_time > 0:
            current_beat_time -= beat_interval_seconds
        
        while current_beat_time <= last_onset_time + beat_interval_seconds:
            if current_beat_time >= 0:
                beat_timings.append(current_beat_time)
            current_beat_time += beat_interval_seconds
        
        final_results.append({
            'bpm': bpm,
            'offset': offset_seconds,
            'fitness': fitness,
            'confidence': np.max(confidence),
            'beats': beat_timings,
        })
    
    #maybe you want onsets too idk
    #you ever think about the fact that OFFset is computed with something called an ONset haha
    #return final_results, onset_times
    print(f"Best bpm, offset: {final_results[0]['bpm']}, {final_results[0]['offset']}")
    return final_results

class SMEditAudioSyncDetector:
    '''
    Instantiates a bpm detector imitating the one used in SMEditor. Similar to AV but with some process improvements.
    '''
    def __init__(self, 
                 window_step: int = 512,
                 fft_size: int = 1024,
                 tempo_fft_size: int = 4096,
                 tempo_step: int = 2,
                 min_bpm: float = 125,
                 max_bpm: float = 250,
                 sample_rate: int = 44100):
        
        self.window_step = window_step
        self.fft_size = fft_size
        self.tempo_fft_size = tempo_fft_size
        self.tempo_step = tempo_step
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.sample_rate = sample_rate
        
        self.AVERAGE_WINDOW_RADIUS = 3
        self.TEMPOGRAM_SMOOTHING = 3
        self.TEMPOGRAM_OFFSET_THRESHOLD = 0.02
        self.TEMPOGRAM_GROUPING_WINDOW = 6
        self.OFFSET_LOOKAHEAD = 800
        
        # essentia init
        self.windowing = es.Windowing(type='hann')
        self.spectrum = es.Spectrum()
        self.fft = es.FFT()
        
        # magic SMEdit numbers. I think these are supposed to replace the polyfit AV uses.
        self.weight_data = [
            (20, 0.4006009013520281), (25, 0.4258037044922291), (31.5, 0.4536690484291709),
            (40, 0.4840856831659204), (50, 0.5142710208279764), (63, 0.5473453749315819),
            (80, 0.5841121495327103), (100, 0.6214074879602299), (125, 0.6601749463607856),
            (160, 0.7054673721340388), (200, 0.7489234225800412), (250, 0.7936507936507937),
            (315, 0.8406893652795292), (400, 0.889284126278346), (500, 0.9291521486643438),
            (630, 0.9675858732462506), (800, 0.9985022466300548), (1000, 0.9997500624843789),
            (1250, 0.9564801530368244), (1600, 0.9409550693954364), (2000, 1.0196278358399185),
            (2500, 1.0955902492467817), (3150, 1.1232799775344005), (4000, 1.0914051841746248),
            (5000, 0.9997500624843789), (6300, 0.8727907484180668), (8000, 0.7722007722007722),
            (10000, 0.7369196757553427), (12500, 0.7768498737618955), (16000, 0.7698229407236336),
            (20000, 0.4311738708634257), (22550, 0.2), (25000, 0)
        ]
        
        self.spectro_weights = None
        self.audio_data = None
        self.spectrogram = []
        self.spectogram_difference = []
        self.novelty_curve = []
        self.novelty_curve_isolated = []
        self.tempogram = []
        self.tempogram_groups = []
        
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file and convert to mono"""
        loader = es.MonoLoader(filename=audio_path, sampleRate=self.sample_rate)
        self.audio_data = loader()
        #print(f"Loaded audio: {len(self.audio_data)} samples, {len(self.audio_data)/self.sample_rate:.2f} seconds")
        return self.audio_data
    
    def _calculate_spectro_weights(self) -> np.ndarray:
        """Calculate frequency weights based on ISO 226 approximation"""
        weights = np.zeros(self.fft_size)
        
        for i in range(self.fft_size):
            freq = (i / (self.fft_size / 2)) * self.sample_rate / 2
            
            # Find the closest weight points
            weight_index = None
            for j, (weight_freq, _) in enumerate(self.weight_data):
                if weight_freq > freq:
                    weight_index = j
                    break
            
            if weight_index is None or weight_index == 0:
                weights[i] = 0
                continue
                
            # Linear interpolation in log space
            lower_freq, lower_weight = self.weight_data[weight_index - 1]
            higher_freq, higher_weight = self.weight_data[weight_index]
            
            log_freq = np.log(1 + freq)
            log_lower = np.log(1 + lower_freq)
            log_higher = np.log(1 + higher_freq)
            
            if log_higher != log_lower:
                t = (log_freq - log_lower) / (log_higher - log_lower)
                weights[i] = lower_weight + t * (higher_weight - lower_weight)
            else:
                weights[i] = lower_weight
                
        return weights
    
    def _render_block(self, block_num: int) -> np.ndarray:
        """Process a single audio block to get spectrogram"""
        if self.audio_data is None:
            raise ValueError("Audio data not loaded")
            
        # Extract audio slice
        start_idx = max(0, block_num * self.window_step - self.fft_size // 2)
        end_idx = block_num * self.window_step + self.fft_size // 2
        
        audio_slice = np.zeros(self.fft_size, dtype=np.float32)
        
        # Handle boundaries
        audio_start = max(0, start_idx)
        audio_end = min(len(self.audio_data), end_idx)
        slice_start = audio_start - start_idx
        slice_end = slice_start + (audio_end - audio_start)
        
        if audio_end > audio_start:
            audio_slice[slice_start:slice_end] = self.audio_data[audio_start:audio_end]
        
        # Apply Hann window and FFT
        windowed = self.windowing(audio_slice)
        spectrum_complex = self.fft(windowed)
        
        # Convert to magnitude spectrum
        spectrum_mag = np.abs(spectrum_complex[:self.fft_size//2])
        
        # log scale
        response = np.log(1 + spectrum_mag)
        
        return response
    
    def _calc_difference(self, block_num: int, current_spectrum: np.ndarray) -> np.ndarray:
        """Calculate spectral difference with previous block"""
        if block_num == 0:
            previous = np.zeros_like(current_spectrum)
        else:
            previous = self.spectrogram[block_num - 1]
            
        # Spectral flux with weighting
        diff = np.maximum(0, current_spectrum - previous) * self.spectro_weights[:len(current_spectrum)]
        return diff
    
    def _calc_isolated_novelty(self, block_num: int):
        """Calculate isolated novelty function with local averaging"""
        # Update blocks around current position
        for i in range(max(0, block_num - self.AVERAGE_WINDOW_RADIUS), block_num + 1):
            if i >= len(self.novelty_curve):
                continue
                
            # Calculate local average
            start_idx = max(0, i - self.AVERAGE_WINDOW_RADIUS)
            end_idx = min(len(self.novelty_curve), i + self.AVERAGE_WINDOW_RADIUS + 1)
            
            local_sum = sum(self.novelty_curve[start_idx:end_idx])
            local_avg = local_sum / (end_idx - start_idx)
            
            # Isolated novelty
            isolated = max(0, self.novelty_curve[i] - local_avg)
            
            # Extend list if necessary
            while len(self.novelty_curve_isolated) <= i:
                self.novelty_curve_isolated.append(0)
                
            self.novelty_curve_isolated[i] = np.log(1 + isolated)
    
    def detect_onsets(self, threshold: float = 0.3) -> List[float]:
        """Detect onset times based on novelty curve"""
        if self.audio_data is None:
            raise ValueError("Audio data not loaded")
            
        print("Computing spectrogram and novelty curve...")
        
        # Initialize weights
        self.spectro_weights = self._calculate_spectro_weights()
        
        # Calculate number of blocks
        max_blocks = int(np.ceil(len(self.audio_data) / self.window_step))
        
        # Reset arrays
        self.spectrogram = []
        self.spectogram_difference = []
        self.novelty_curve = []
        self.novelty_curve_isolated = []
        
        # Process each block
        for block_num in range(max_blocks):                
            # Get spectrum
            spectrum = self._render_block(block_num)
            self.spectrogram.append(spectrum)
            
            # Calculate difference
            diff = self._calc_difference(block_num, spectrum)
            self.spectogram_difference.append(diff)
            
            # Sum for novelty curve
            novelty_sum = np.sum(diff)
            self.novelty_curve.append(novelty_sum)
            
            # Calculate isolated novelty
            self._calc_isolated_novelty(block_num)
        
        # Detect peaks
        peaks = []
        for i in range(1, len(self.novelty_curve_isolated)):
            if (self.novelty_curve_isolated[i] > threshold and 
                self.novelty_curve_isolated[i] > self.novelty_curve_isolated[i-1]):
                
                # Convert block to time
                time_seconds = (i * self.window_step) / self.sample_rate
                peaks.append(time_seconds)
        
        #print(f"Found {len(peaks)} onsets")
        return peaks
    
    def detect_tempo_and_offset(self) -> Tuple[List[Dict], List[Dict]]:
        """Detect tempo and offset information"""
        if not self.novelty_curve_isolated:
            raise ValueError("Must run detect_onsets first")
        
        # Scale novelty curve to max 1
        novelty_array = np.array(self.novelty_curve_isolated)
        max_val = np.max(novelty_array)
        if max_val > 0:
            scaled_novelty = novelty_array / max_val
        else:
            scaled_novelty = novelty_array
        
        # Calculate tempogram
        max_tempo_blocks = int(np.ceil(len(scaled_novelty) / self.tempo_step))
        self.tempogram = []
        self.tempogram_groups = []
        
        for block_num in range(max_tempo_blocks):   
            # Extract slice for tempo analysis
            start_idx = max(0, block_num * self.tempo_step - self.tempo_fft_size // 2)
            end_idx = block_num * self.tempo_step + self.tempo_fft_size // 2
            
            tempo_slice = np.zeros(self.tempo_fft_size, dtype=np.float32)
            
            # Handle boundaries
            data_start = max(0, start_idx)
            data_end = min(len(scaled_novelty), end_idx)
            slice_start = data_start - start_idx
            slice_end = slice_start + (data_end - data_start)
            
            if data_end > data_start:
                tempo_slice[slice_start:slice_end] = scaled_novelty[data_start:data_end]
            
            # Apply Hann window and FFT
            windowed = tempo_slice * np.hanning(self.tempo_fft_size)
            fft_result = np.fft.fft(windowed)
            response = np.abs(fft_result[:self.tempo_fft_size//2])
            
            # Log scale
            response = np.log(1 + response)
            
            # Convert FFT bins to BPM
            tempos = {}
            for i, value in enumerate(response):
                if value == 0:
                    continue
                    
                # Calculate BPM from FFT bin
                tmp = (self.sample_rate * 60) / (self.window_step * self.tempo_fft_size) * i
                
                if tmp > self.max_bpm * 4 or tmp < self.min_bpm / 4:
                    continue
                    
                # Octave reduction
                while tmp > self.max_bpm and not np.isinf(tmp):
                    tmp /= 2
                while tmp < self.min_bpm and tmp != 0:
                    tmp *= 2
                    
                bpm = round(tmp, 3)
                if bpm not in tempos:
                    tempos[bpm] = 0
                tempos[bpm] += value
            
            # Sort by confidence
            tempo_list = [{'bpm': bpm, 'value': value} for bpm, value in tempos.items()]
            tempo_list.sort(key=lambda x: x['value'], reverse=True)
            self.tempogram.append(tempo_list[:10])  # Keep top 10
            
            # Group similar tempos
            groups = []
            for tempo in tempo_list[:10]:
                # Find closest group
                closest_group = None
                for group in groups:
                    if (abs(group['center'] - tempo['bpm']) < self.TEMPOGRAM_GROUPING_WINDOW):
                        closest_group = group
                        break
                
                if closest_group is None:
                    groups.append({
                        'center': tempo['bpm'],
                        'groups': [tempo],
                        'avg': tempo['bpm']
                    })
                else:
                    closest_group['groups'].append(tempo)
                    total_weight = sum(g['value'] for g in closest_group['groups'])
                    weighted_sum = sum(g['bpm'] * g['value'] for g in closest_group['groups'])
                    closest_group['avg'] = weighted_sum / total_weight if total_weight > 0 else tempo['bpm']
            
            self.tempogram_groups.append(groups)
        
        # Extract top BPMs and calculate offsets
        return self._calculate_bpm_and_offset()
    
    def _calculate_bpm_and_offset(self) -> Tuple[List[Dict], List[Dict]]:
        """Calculate final BPM and offset results"""
        print("Calculating BPM and offset results...")
        
        # Find most consistent BPM
        bpm_counts = {}
        peak_scan_start = 0
        
        for i, groups in enumerate(self.tempogram_groups):
            candidates = [g for g in groups if g['groups'][0]['value'] >= self.TEMPOGRAM_OFFSET_THRESHOLD]
            if not candidates:
                continue
                
            peak_scan_start = i
            
            for group in candidates:
                # Smooth the BPM estimate
                total_blocks = 0
                bpm_total = 0
                
                for j in range(max(0, i - self.TEMPOGRAM_SMOOTHING), 
                              min(len(self.tempogram_groups), i + self.TEMPOGRAM_SMOOTHING + 1)):
                    closest_group = None
                    for other_group in self.tempogram_groups[j]:
                        if (abs(other_group['center'] - group['center']) < self.TEMPOGRAM_GROUPING_WINDOW):
                            closest_group = other_group
                            break
                    
                    if closest_group:
                        bpm_total += closest_group['avg']
                        total_blocks += 1
                
                if total_blocks > 0:
                    bpm = round(bpm_total / total_blocks)
                    bpm_counts[bpm] = bpm_counts.get(bpm, 0) + 1
        
        # Get top BPMs
        bpm_results = []
        if bpm_counts:
            sorted_bpms = sorted(bpm_counts.items(), key=lambda x: x[1], reverse=True)
            total_confidence = sum(count for _, count in sorted_bpms[:5])
            
            for bpm, count in sorted_bpms[:5]:
                confidence = count / total_confidence if total_confidence > 0 else 0
                bpm_results.append({
                    'bpm': bpm,
                    'confidence': confidence
                })
        
        # Calculate offset for the top BPM
        offset_results = []
        if bpm_results:
            first_bpm = bpm_results[0]['bpm']
            offset_results = self._calculate_offset_for_bpm(first_bpm, peak_scan_start)
        
        return bpm_results, offset_results
    
    def _calculate_offset_for_bpm(self, bpm: float, peak_scan_start: int) -> List[Dict]:
        """Calculate offset options for a given BPM"""
        if not self.novelty_curve_isolated:
            return []
            
        beat_length_blocks = (60 / bpm) * (self.sample_rate / self.window_step)
        
        # Create analysis wave for beat tracking
        analyze_wave = []
        for i in range(self.OFFSET_LOOKAHEAD):
            beat_block = (i % beat_length_blocks) / beat_length_blocks
            t = 0
            n = 0
            for j in range(1, 5):  # harmonics 1-4
                weight = 1 / j
                phase_weight = max(1 - abs(round(beat_block * j) / j - beat_block) * 12, 0)
                n += phase_weight * weight
                t += weight
            analyze_wave.append(n / t if t > 0 else 0)
        
        # Test different offset positions
        options = []
        end_scan = min(len(self.novelty_curve_isolated), 
                      int(peak_scan_start + beat_length_blocks))
        
        for i in range(peak_scan_start, end_scan):
            if i + self.OFFSET_LOOKAHEAD >= len(self.novelty_curve_isolated):
                break
                
            # Calculate correlation with beat pattern
            response = 0
            for j in range(self.OFFSET_LOOKAHEAD):
                if i + j < len(self.novelty_curve_isolated):
                    response += analyze_wave[j] * self.novelty_curve_isolated[i + j]
            
            offset = -((i * self.window_step) / self.sample_rate) % (60 / bpm)
            options.append({
                'offset': offset,
                'response': response
            })
        
        # Sort by response strength
        options.sort(key=lambda x: x['response'], reverse=True)
        
        # Return top 5 with confidence percentages
        if options:
            total_response = sum(opt['response'] for opt in options[:5])
            offset_results = []
            
            for opt in options[:5]:
                confidence = opt['response'] / total_response if total_response > 0 else 0
                offset_results.append({
                    'offset': round(opt['offset'], 3),
                    'confidence': confidence
                })
            
            return offset_results
        
        return []
    
    def calculate_beat_times(self, bpm: float, offset: float, audio_duration: float = None) -> List[float]:
        """Calculate beat times for a given BPM and offset"""
        if audio_duration is None:
            if self.audio_data is None:
                raise ValueError("Audio data not loaded and no duration provided")
            audio_duration = len(self.audio_data) / self.sample_rate
        
        beat_interval = 60.0 / bpm  # seconds per beat
        beat_times = []
        
        # Start from the offset and generate beats
        current_time = offset
        while current_time < audio_duration:
            if current_time >= 0:  # Only include positive times
                beat_times.append(round(current_time, 3))
            current_time += beat_interval
        
        return beat_times

def smedit_analyze_audio(audio_path: str, threshold: float = 0.3, bpm_range = [80,200]):
    """Analyze audio file for tempo and offset"""
    detector = SMEditAudioSyncDetector(min_bpm=bpm_range[0], max_bpm=bpm_range[1])
    
    # Load audio
    detector.load_audio(audio_path)
    
    # Detect onsets
    onsets = detector.detect_onsets(threshold=threshold)
    
    # Detect tempo and offset
    bpm_results, offset_results = detector.detect_tempo_and_offset()
    
    # Calculate beat times for each BPM estimate
    audio_duration = len(detector.audio_data) / detector.sample_rate
    beat_times = {}
    
    for bpm_result in bpm_results:
        bpm = bpm_result['bpm']
        # Use the best offset for this BPM (first one if available)
        if offset_results:
            offset = offset_results[0]['offset']
        else:
            offset = 0.0
        
        beats = detector.calculate_beat_times(bpm, offset, audio_duration)
        beat_times[bpm] = beats
    print(f"Best bpm, offset: {bpm_results[0]}, {offset_results[0]}")
    
    return {
        'bpm_results': bpm_results,
        'offset_results': offset_results,
        'onsets': onsets,
        'beat_times': beat_times,
        'detector': detector
    }

def get_template():
    template = """\
#TITLE:{title};
#ARTIST:{artist};
#MUSIC:{music_fp};
#OFFSET:{offset};
#BPMS:{bpm};
#STOPS:;
{charts}\
"""
    return template

def get_chart_template():
    chart_template = """\
    #NOTES:
    dance-single:
    DDCL:
    {ccoarse}:
    {cfine}:
    0.0,0.0,0.0,0.0,0.0:
    {measures};\
    """
    return chart_template

def ez_name(x):
    """
    Cleans and formats a given string x by removing spaces, replacing non-alphanumeric characters with underscores (_), 
    and ensuring it returns a string suitable for naming files or directories.
    _______
    ez_name("Hello World! 2023")  # Returns 'HelloWorld_2023'
    """
    x = ''.join(x.strip().split())
    x_clean = []
    for char in x:
        if char.isalnum():
            x_clean.append(char)
        else:
            x_clean.append('_')
    return ''.join(x_clean)

def get_subdirs(root, choose=False):
    """
    Lists the names of all subdirectories within a given root directory. 
    Optionally, it allows the user to select specific subdirectories to return.
    """
    subdir_names = sorted([x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))])
    if choose:
        for i, subdir_name in enumerate(subdir_names):
            print('{}: {}'.format(i, subdir_name))
        subdir_idxs = [int(x) for x in input('Which subdir(s)? ').split(',')]
        subdir_names = [subdir_names[i] for i in subdir_idxs]
    return subdir_names
    

def open_dataset_fps(*args):
    datasets = []
    for data_fp in args:
        if not data_fp:
            datasets.append([])
            continue

        with open(data_fp, 'r') as f:
            song_fps = f.read().split()
        dataset = []
        for song_fp in song_fps:
            with open(song_fp, 'rb') as f:
                dataset.append(pickle.load(f))
        datasets.append(dataset)
    return datasets[0] if len(datasets) == 1 else datasets

def windowize(dataset, frames = 7, front_set = 'None', go_backwards = False, take_windows = None, return_type = 'numpy'):
    if type(dataset) == list:
        dataset = np.array(dataset)
    dslen = dataset.shape[0]
    ds_shape = list(dataset.shape)
    ds_shape[0] = frames
    if front_set == 'None':
        front = np.zeros(ds_shape)
    elif front_set == 'min':
        front = np.ones(ds_shape)*np.min(dataset)
    else:
        front = np.array([front_set for i in range(frames)])
    if go_backwards:
        dataset = np.append(dataset, front, axis = 0)
    else:
        dataset = np.append(front, dataset, axis = 0)
    if take_windows is not None:
        new_ds = [dataset[i:i+1+frames] for i in take_windows]
    else:
        new_ds = [dataset[i:i+1+frames] for i in range(dslen)]
    if return_type == 'list':
        return new_ds
    elif return_type == 'numpy':
        return np.array(new_ds)
    else:
        raise NotImplementedError('o_o')

def sparse_to_categorical(val, max_val):
    a = np.zeros(max_val+1)
    a[val] = 1
    return a

def unfoldify(listy, base = 4):
    out = np.zeros(base*len(listy))
    for i in range(len(listy)):
        if listy[i] != 0:
            out[int(base*i+listy[i]-1)] = 1
    return out

def sparceify(listy, base = 4):
    out = 0
    for i in range(len(listy)):
        if listy[i] != 0:
            out += (listy[i])*(base**i)
    return out

def get_dataset_fp_list(*args):
    datasets = []
    for data_fp in args:
        if not data_fp:
            datasets.append([])
            continue

        with open(data_fp, 'r') as f:
            song_fps = f.read().split()
        dataset = []
        for song_fp in song_fps:
            dataset.append(song_fp)
        datasets.append(dataset)
    return datasets[0] if len(datasets) == 1 else datasets


def make_onset_feature_context(song_features, frame_idx, radius, left_radius = False):
    nframes = song_features.shape[0]
    
    assert nframes > 0
    if left_radius:
        frame_idxs = range(frame_idx - left_radius, frame_idx + radius + 1)
    else:
        frame_idxs = range(frame_idx - radius, frame_idx + radius + 1)
    context = np.zeros((len(frame_idxs),) + song_features.shape[1:], dtype=song_features.dtype)
    for i, frame_idx in enumerate(frame_idxs):
        if frame_idx >= 0 and frame_idx < nframes:
            context[i] = song_features[frame_idx]
        else:
            context[i] = np.ones_like(song_features[0])*np.log(1e-16)

    return context

def make_onset_feature_context_range(song_features, start_idx, end_idx, radius = 0, frame_density = 32):
    nframes = song_features.shape[0]

    start_idx, end_idx = int(start_idx*100), int(end_idx*100)
    
    assert nframes > 0

    frame_idxs = np.linspace(start_idx - radius, end_idx + radius, num = frame_density, endpoint = False).astype(int)
    context = np.zeros((len(frame_idxs),) + song_features.shape[1:], dtype=song_features.dtype)
    for i, frame_idx in enumerate(frame_idxs):
        if frame_idx >= 0 and frame_idx < nframes:
            context[i] = song_features[frame_idx]
        else:
            context[i] = np.ones_like(song_features[0])*np.log(1e-16)

    return context

def front_null(dataset, frames = 7, front_set = 'None', back_null = False):
    if type(dataset) == list:
        dataset = np.array(dataset)
    ds_shape = list(dataset.shape)
    ds_shape[0] = frames

    if front_set == 'min':
        front = np.ones(ds_shape) * np.min(dataset)
    elif front_set == 'None':
        front = np.zeros(ds_shape)
    else:
        front = np.array([front_set for i in range(frames)])
    dataset = np.append(front, dataset, axis = 0)
    if back_null:
        dataset = np.append(dataset, front, axis = 0)
    return dataset

def create_analyzers(fs=44100.0,
                     nhop=512,
                     nffts=[1024, 2048, 4096],
                     mel_nband=80,
                     mel_freqlo=27.5,
                     mel_freqhi=16000.0):
    analyzers = []
    for nfft in nffts:
        window = Windowing(size=nfft, type='blackmanharris62')
        spectrum = Spectrum(size=nfft)
        mel = MelBands(inputSize=(nfft // 2) + 1,
                       numberBands=mel_nband,
                       lowFrequencyBound=mel_freqlo,
                       highFrequencyBound=mel_freqhi,
                       sampleRate=fs)
        analyzers.append((window, spectrum, mel))
    return analyzers

def extract_mel_feats(audio_fp, analyzers, fs=44100.0, nhop=512, nffts=[1024, 2048, 4096], log_scale=True):
    # Extract features
    loader = MonoLoader(filename=audio_fp, sampleRate=fs)
    samples = loader()
    feat_channels = []
    for nfft, (window, spectrum, mel) in zip(nffts, analyzers):
        feats = []
        for frame in FrameGenerator(samples, nfft, nhop):
            frame_feats = mel(spectrum(window(frame)))
            feats.append(frame_feats)
        feat_channels.append(feats)

    # Transpose to move channels to axis 2 instead of axis 0
    feat_channels = np.transpose(np.stack(feat_channels), (1, 2, 0))

    # Apply numerically-stable log-scaling
    # Value 1e-16 comes from inspecting histogram of raw values and picking some epsilon >2 std dev left of mean
    if log_scale:
        feat_channels = np.log(feat_channels + 1e-16)

    return feat_channels

def extract_BPM(audio_fp, min_tempo = 100, max_tempo = 200):
    audio = MonoLoader(filename = audio_fp)()
    rhythm_extractor = RhythmExtractor2013(method="multifeature", minTempo = min_tempo, maxTempo = max_tempo)
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
    beat_dict={'BPM' : bpm, 'offset' : 2*beats[0]-beats[1], 'beats' : beats, 'beat_intervals': beats_intervals}
    return beat_dict

def quick_reducify(ds, indices):
    return [[a[i] for i in indices] for a in ds]

def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(min(int(np.searchsorted(t, np.random.rand(1)*s)), 256))

def unravel_onehot(onehot, base):
    if onehot == 0:
        return '0000'
    else:
        out = ''
        while onehot:
            onehot, r = divmod(onehot, base)
            out += str(r)
        while len(out)<4:
            out += '0'
        return out

def downsample(dataset, down_key=2):
    return [[data[i] for i in range(0,len(data),down_key)] for data in dataset]

def label_to_vect_dict(labels, force_max_len = None):
    values = [list(a) for a in labels]
    if not force_max_len:
        max_len = np.lcm.reduce(np.unique([len(a) for a in values if len(a) != 0]))
    else:
        max_len = force_max_len

    new_values = [np.zeros((max_len,)) for a in values]
    for i in range(len(new_values)):
        num_ticks = len(values[i])
        if num_ticks != 0:
            step = max_len/num_ticks
            for j in range(0, max_len, int(step)):
                new_values[i][j] = values[i][int(j/step)]
    new_dict = {k : v for k , v in zip(labels, new_values)}

    return new_dict

def ddc_string_to_step(string, narrow_types = 4):
    listy = [int(a) for a in list(string)]
    out = np.zeros((4*narrow_types))
    for i in range(len(listy)):
        out[narrow_types*i + listy[i]] = 1
    return out
    
def pickle_box(fp):
    #Load into a temporary namespace to avoid reference pollution
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    return data

class MaskedCategoricalAccuracy(Metric):
    def __init__(self, ignored_classes, name='masked_categorical_accuracy', **kwargs):
        super(MaskedCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.ignored_classes = tf.constant(ignored_classes, dtype=tf.int64)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_classes = tf.argmax(y_true, axis=-1)
        y_pred_classes = tf.argmax(y_pred, axis=-1)

        # Mask for ignored classes
        mask = ~tf.reduce_any(tf.equal(tf.expand_dims(y_true_classes, -1), self.ignored_classes), axis=-1)

        # Only count the accuracy for non-ignored classes
        matches = tf.cast(tf.equal(y_true_classes, y_pred_classes), self.dtype)
        matches = tf.boolean_mask(matches, mask)

        num_matches = tf.reduce_sum(matches)
        num_counted = tf.cast(tf.size(matches), self.dtype)

        self.total.assign_add(num_matches)
        self.count.assign_add(num_counted)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)