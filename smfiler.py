import glob
import logging as smlog
import os
import traceback
from collections import OrderedDict, defaultdict
import json
json.encoder.FLOAT_REPR = lambda f: ('%.6f' % f)
from util import ez_name, get_subdirs
import random
from abstime import calc_note_beats_and_abs_times
from parse import parse_sm_txt
import copy
from util import *

from essentia.standard import MonoLoader, FrameGenerator, Windowing, Spectrum, MelBands
import numpy as np
import gc
import pickle
import json
import os

import argparse
import threading

def extract_jsons(dir_name = 'songs',
                style = 'stamina',
                splits=[8,1,1],
                split_names= ['train','valid','test'],
                shuffle = True,
                shuffle_seed = 420,
                do_permutations = True,
                ):
    permutations= ['3120','0213','3210']
    _ATTR_REQUIRED = ['offset', 'bpms', 'notes']
    substitutions={'M':'0', '4':'2'}
    arrow_types = set(['0','1','2','3'])
    packs_dir = 'raw/'+dir_name
    json_dir = 'json/'+dir_name
    pack_eznames = set()
    pack_names = get_subdirs(packs_dir, False)
    pack_dirs = [os.path.join(packs_dir, pack_name) for pack_name in pack_names]
    pack_sm_globs = [os.path.join(pack_dir, '*', '*.sm') for pack_dir in pack_dirs]
    if not os.path.isdir(json_dir):
        os.makedirs(json_dir, exist_ok=True)
    split_fps = [[] for i in range(len(split_names))]
    for pack_name, pack_sm_glob in zip(pack_names, pack_sm_globs):
        pack_sm_fps = sorted(glob.glob(pack_sm_glob))
        pack_ezname = ez_name(pack_name)
        
        # Skip if pack name conflict
        if pack_ezname in pack_eznames:
            print(f"Warning: Pack name conflict detected: {pack_ezname}. Skipping this pack.")
            continue
        pack_eznames.add(pack_ezname)

        if len(pack_sm_fps) > 0:
            pack_outdir = os.path.join(json_dir, pack_ezname)
            if not os.path.isdir(pack_outdir):
                os.mkdir(pack_outdir)

        sm_eznames = set()
        for sm_fp in pack_sm_fps:
            sm_name = os.path.split(os.path.split(sm_fp)[0])[1]
            sm_ezname = ez_name(sm_name)

            # Skip if song name conflict
            if sm_ezname in sm_eznames:
                print(f"Warning: Song name conflict detected in pack {pack_name}: {sm_ezname}. Skipping this song.")
                continue
            sm_eznames.add(sm_ezname)

            with open(sm_fp, 'r') as sm_f:
                sm_txt = sm_f.read()

            # parse file
            try:
                sm_attrs = parse_sm_txt(sm_txt)
            except ValueError as e:
                smlog.error('{} in\n{}'.format(e, sm_fp))
                continue
            except Exception as e:
                smlog.critical('Unhandled parse exception {}'.format(traceback.format_exc()))
                raise e

            # check required attrs
            try:
                for attr_name in _ATTR_REQUIRED:
                    if attr_name not in sm_attrs:
                        raise ValueError('Missing required attribute {}'.format(attr_name))
            except ValueError as e:
                smlog.error('{}'.format(e))
                continue

            # handle missing music
            root = os.path.abspath(os.path.join(sm_fp, '..'))
            music_fp = os.path.join(root, sm_attrs.get('music', ''))
            if 'music' not in sm_attrs or not os.path.exists(music_fp):
                music_names = []
                sm_prefix = os.path.splitext(sm_name)[0]

                # check directory files for reasonable substitutes
                for filename in os.listdir(root):
                    prefix, ext = os.path.splitext(filename)
                    if ext.lower()[1:] in ['mp3', 'ogg']:
                        music_names.append(filename)

                try:
                    # handle errors
                    if len(music_names) == 0:
                        raise ValueError('No music files found')
                    elif len(music_names) == 1:
                        sm_attrs['music'] = music_names[0]
                    else:
                        raise ValueError('Multiple music files {} found'.format(music_names))
                except ValueError as e:
                    smlog.error('{}'.format(e))
                    continue

                music_fp = os.path.join(root, sm_attrs['music'])

            bpms = sm_attrs['bpms']
            offset = sm_attrs['offset']
            stops = sm_attrs.get('stops', [])

            out_json_fp = os.path.join(pack_outdir, '{}_{}.json'.format(pack_ezname, sm_ezname))
            out_json = OrderedDict([
                ('sm_fp', os.path.abspath(sm_fp)),
                ('music_fp', os.path.abspath(music_fp)),
                ('pack', pack_name),
                ('title', sm_attrs.get('title')),
                ('artist', sm_attrs.get('artist')),
                ('offset', offset),
                ('bpms', bpms),
                ('stops', stops),
                ('charts', [])
            ])

            for idx, sm_notes in enumerate(sm_attrs['notes']):
                note_beats_and_abs_times = calc_note_beats_and_abs_times(offset, bpms, stops, sm_notes[5])
                notes = {
                    'type': sm_notes[0],
                    'desc_or_author': sm_notes[1],
                    'difficulty_coarse': sm_notes[2],
                    'difficulty_fine': sm_notes[3],
                    'notes': note_beats_and_abs_times,
                }
                if len(substitutions) > 0:
                    notes_cleaned = []
                    for meas, beat, time, note in notes['notes']:
                        for old, new in list(substitutions.items()):
                            note = note.replace(old, new)

                        notes_cleaned.append((meas, beat, time, note))
                    notes['notes'] = notes_cleaned

                if len(arrow_types) > 1:
                    bad_types = set()
                    for _, beat, time, note in notes['notes']:
                        for char in note:
                            if char not in arrow_types:
                                bad_types.add(char)
                    if len(bad_types) > 0:
                        print('Unacceptable chart arrow types: {}'.format(bad_types))
                        continue
                        
                out_json['charts'].append(notes)
                
                if do_permutations:
                    for permutation in permutations:
                        chart_meta_copy = copy.deepcopy(notes)
                        notes_cleaned = []
                        for meas, beat, time, note in chart_meta_copy['notes']:
                            note_new = ''.join([note[int(permutation[i])] for i in range(len(permutation))])
        
                            notes_cleaned.append((meas, beat, time, note_new))
                            chart_meta_copy['notes'] = notes_cleaned
        
                        out_json['charts'].append(chart_meta_copy)                

            with open(out_json_fp, 'w') as out_f:
                try:
                    out_f.write(json.dumps(out_json))
                except UnicodeDecodeError:
                    smlog.error('Unicode error in {}'.format(sm_fp))
                    continue

            print('Parsed {} - {}: {} charts'.format(pack_name, sm_name, len(out_json['charts'])))

    pack_names = get_subdirs(json_dir, False)

    for pack_name in pack_names:
        pack_dir = os.path.join(json_dir, pack_name)
        sub_fps = sorted(os.listdir(pack_dir))
        sub_fps = [json_dir + '/' + pack_name + '/' + a for a in sub_fps]

        if shuffle:
            random.seed(shuffle_seed)
            random.shuffle(sub_fps)

        if len(splits) == 0:
            splits = [1.0]
        else:
            splits = [x / sum(splits) for x in splits]

        split_ints = [int(len(sub_fps) * split) for split in splits]
        split_ints[0] += len(sub_fps) - sum(split_ints)

        k = 0
        for split_int in split_ints:
            split_fps[k] += sub_fps[:split_int]
            k+=1
            sub_fps = sub_fps[split_int:]

    for split, splitname in zip(split_fps, split_names):
        out_name = '{}{}.txt'.format(dir_name, '_' + splitname if splitname else '')
        out_fp = os.path.join(json_dir, out_name)
        with open(out_fp, 'w') as f:
            f.write('\n'.join(split))

def extract_feats(log_scale=True,
                  nhop=441,
                  nffts=[1024, 2048, 4096],
                  out_dir='feats/songs',
                  dataset_fps=['json/songs/songs_train.txt',
                               'json/songs/songs_test.txt',
                               'json/songs/songs_valid.txt'],
                  mel_nband=80):
    # Create analyzers
    analyzers = create_analyzers(fs=44100.0, nhop=nhop, nffts=nffts, mel_nband=mel_nband)

    # Create outdir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Iterate through packs extracting features
    for dataset_fp in dataset_fps:
        with open(dataset_fp, 'r') as f:
            json_fps = f.read().splitlines()

        for json_fp in json_fps:
            song_name = os.path.splitext(os.path.split(json_fp)[1])[0]
            feats_fp = os.path.join(out_dir, f'{song_name}.pkl')

            # Skip if features already extracted
            if os.path.exists(feats_fp):
                print(f"Skipping {song_name} (features already extracted)")
                continue

            print(f'Extracting feats from {song_name}')

            with open(json_fp, 'r') as json_f:
                meta = json.loads(json_f.read())
            song_metadata = {k: meta[k] for k in ['title', 'artist']}

            music_fp = meta['music_fp']
            if not os.path.exists(music_fp):
                raise ValueError(f'No music for {json_fp}')

            song_feats = extract_mel_feats(
                music_fp, analyzers, fs=44100.0, nhop=nhop, nffts=nffts, log_scale=log_scale
            )

            with open(feats_fp, 'wb') as f:
                pickle.dump(song_feats, f)
        gc.collect()

def create_beat_dicts(meta):
    beat_dicts = {}
    time_dicts = {}
    stream_dicts = {}

    for raw_chart in meta['charts']:
        if raw_chart['type'] and raw_chart['type'] == 'dance-double':
            continue

        diff = raw_chart['difficulty_fine']
        if diff in beat_dicts:
            continue

        notes = raw_chart['notes']
        if not notes:
            continue

        num_beats = int(np.ceil(notes[-1][1])) - 1
        beat_dict = {}
        time_dict = {}
        stream_dict = {}

        beat_time = meta['offset']
        bpms = meta['bpms']
        cur_bpm_idx = 0
        cur_bpm = bpms[cur_bpm_idx][1]
        next_bpm_shift = bpms[cur_bpm_idx + 1][0] if cur_bpm_idx + 1 < len(bpms) else np.inf

        beat_note_map = defaultdict(list)
        beat_rhythm_map = {}
        beat_time_map = {}

        for note in notes:
            beat = int(np.floor(note[1]))
            beat_note_map[beat].append(note)
            if beat not in beat_rhythm_map:
                beat_rhythm_map[beat] = int(note[0][1] / 4)
                beat_time_map[beat] = note[2]

        for beat in range(num_beats):
            if beat >= next_bpm_shift:
                cur_bpm_idx += 1
                cur_bpm = bpms[cur_bpm_idx][1]
                next_bpm_shift = bpms[cur_bpm_idx + 1][0] if cur_bpm_idx + 1 < len(bpms) else np.inf

            beat_notes = [int((note[1] - beat) * (note[0][1] / 4))
                          for note in beat_note_map.get(beat, []) if note[3] != '0000']

            beat_rhythm = beat_rhythm_map.get(beat, 0)
            beat_list = [0] * beat_rhythm
            for note_idx in beat_notes:
                if 0 <= note_idx < beat_rhythm:
                    beat_list[note_idx] = 1

            # Simplify the rhythm
            while len(beat_list) % 2 == 0 and sum(beat_list[1::2]) == 0 and beat_list:
                beat_list = beat_list[::2]

            beat_str = ''.join(map(str, beat_list))
            beat_time = beat_time_map.get(beat, beat_time + 60 / cur_bpm)
            stream_info = [diff, cur_bpm, raw_chart['difficulty_coarse']]

            beat_dict[beat] = beat_str
            time_dict[beat] = beat_time
            stream_dict[beat] = stream_info

        time_dict[len(time_dict)] = beat_time + 60 / cur_bpm
        beat_dicts[diff] = beat_dict
        time_dicts[diff] = time_dict
        stream_dicts[diff] = stream_dict

    return beat_dicts, time_dicts, stream_dicts

def create_beat_audio_contexts(beat_dicts, time_dicts):
    beat_audio_contexts = dict()
    for diff in beat_dicts.keys():
        beat_dict = beat_dicts[diff]
        time_dict = time_dicts[diff]
        beat_audio_context = dict()
        for beat in beat_dict.keys():
            start_time = time_dict[beat]
            end_time = time_dict[beat+1]
            context = [start_time, end_time]
            beat_audio_context[beat] = context
            
        beat_audio_contexts[diff] = beat_audio_context

    return beat_audio_contexts

def create_sym_dicts(meta):
    sym_dicts = []
    for raw_chart in meta['charts']:
        if not raw_chart['type'] or raw_chart['type'] != 'dance-double':
            sym_list = []
            notes = raw_chart['notes']
            last = 0
            for note in notes:
                if note[3] != '0000':
                    sym_list.append([note[1]-last, note[3], 
                                     int(note[2]*100), 
                                     raw_chart['difficulty_fine']])
                    last = note[1]
            sym_dicts.append(sym_list)
    return sym_dicts

def extract_onsets(dataset_fps = ['json/songs/songs_train.txt','json/songs/songs_test.txt','json/songs/songs_valid.txt'],
                    out_dir = 'onset/songs/',
                    feats_dir = 'feats/songs/'):
    name_from_fp = lambda x: os.path.splitext(os.path.split(x)[1])[0]
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    stream_labels = []
    for dataset_fp in dataset_fps:
        dataset_name = name_from_fp(dataset_fp)
        dataset_out_names = []
        
        with open(dataset_fp, 'r') as f:
            json_fps = f.read().splitlines()
            json_fps = list(np.unique(json_fps))
            
            for json_fp in json_fps:
                json_name = name_from_fp(json_fp)

                with open(json_fp, 'r') as json_f:
                    meta = json.loads(json_f.read())
                song_metadata = {k: meta[k] for k in ['title', 'artist', 'offset', 'bpms', 'stops']}

                if feats_dir:
                    song_feats_fp = os.path.join(feats_dir, '{}.pkl'.format(json_name))

                beat_dicts, time_dicts, stream_dicts = create_beat_dicts(meta)
                beat_audio_contexts = create_beat_audio_contexts(beat_dicts, time_dicts)

                data = []
                streams = []
                labels = []
                labels_unsqueeze = []

                for beat_dict in beat_dicts.values():
                    labels.append(list(beat_dict.values()))
                    labels_unsqueeze += list(beat_dict.values())
                for beat_audio_context in beat_audio_contexts.values():
                    data.append(list(beat_audio_context.values()))
                for stream_dict in stream_dicts.values():
                    streams.append(list(stream_dict.values()))
                print('Processed {} onsets'.format(song_metadata['title']))

                song_array = [data, streams, labels, song_feats_fp]
                stream_labels = list(np.unique(stream_labels + labels_unsqueeze))
                
                out_name = '{}.pkl'.format(json_name)
                out_fp = os.path.join(out_dir, out_name)
                dataset_out_names.append(os.path.abspath(out_fp))
                with open(out_fp, 'wb') as f:
                    pickle.dump(song_array, f)
                gc.collect()

        with open(os.path.join(out_dir, '{}.txt'.format(dataset_name)), 'w') as f:
            f.write('\n'.join(dataset_out_names))
        with open(os.path.join(out_dir, 'stream_labels.pkl'), 'wb') as f:
            pickle.dump(stream_labels, f)

def extract_syms(dataset_fps = ['json/songs/songs_train.txt','json/songs/songs_test.txt','json/songs/songs_valid.txt'],
                out_dir = 'sym/songs/',
                feats_dir = 'feats/songs/'):
    name_from_fp = lambda x: os.path.splitext(os.path.split(x)[1])[0]

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for dataset_fp in dataset_fps:
        dataset_name = name_from_fp(dataset_fp)
        dataset_out_names = []
        
        with open(dataset_fp, 'r') as f:
            json_fps = f.read().splitlines()
            json_fps = list(np.unique(json_fps))
            
            for json_fp in json_fps:
                json_name = name_from_fp(json_fp)

                with open(json_fp, 'r') as json_f:
                    meta = json.loads(json_f.read())
                song_metadata = {k: meta[k] for k in ['title', 'artist', 'offset', 'bpms', 'stops']}

                if feats_dir:
                    song_feats_fp = os.path.join(feats_dir, '{}.pkl'.format(json_name))

                sym_dicts = create_sym_dicts(meta)

                data = []

                for data_list in sym_dicts:
                    data += [data_list]
                data = [data, song_feats_fp]
                print('Processed {}'.format(song_metadata['title']))

                out_name = '{}.pkl'.format(json_name)
                out_fp = os.path.join(out_dir, out_name)
                dataset_out_names.append(os.path.abspath(out_fp))
                with open(out_fp, 'wb') as f:
                    pickle.dump(data, f)
                gc.collect()

        with open(os.path.join(out_dir, '{}.txt'.format(dataset_name)), 'w') as f:
            f.write('\n'.join(dataset_out_names))

def get_parser():
    parser = argparse.ArgumentParser(description="Multi-function parser for data extraction")

    subparsers = parser.add_subparsers(dest='command', help='Function to run')

    # === extract_jsons ===
    parser_jsons = subparsers.add_parser('extract_jsons', help='Extract JSONs')
    parser_jsons.add_argument('--dir_name', default='songs')
    parser_jsons.add_argument('--style', default='stamina')
    parser_jsons.add_argument('--splits', type=int, nargs=3, default=[8, 1, 1])
    parser_jsons.add_argument('--split_names', nargs=3, default=['train', 'valid', 'test'])
    parser_jsons.add_argument('--shuffle', type=bool, default=True)
    parser_jsons.add_argument('--shuffle_seed', type=int, default=420)
    parser_jsons.add_argument('--do_permutations', type=bool, default=True)

    # === extract_feats ===
    parser_feats = subparsers.add_parser('extract_feats', help='Extract features')
    parser_feats.add_argument('--log_scale', type=bool, default=True)
    parser_feats.add_argument('--nhop', type=int, default=441)
    parser_feats.add_argument('--nffts', type=int, nargs='+', default=[1024, 2048, 4096])
    parser_feats.add_argument('--out_dir', default='feats/songs')
    parser_feats.add_argument('--dataset_fps', nargs='+', default=[
        'json/songs/songs_train.txt',
        'json/songs/songs_test.txt',
        'json/songs/songs_valid.txt'
    ])
    parser_feats.add_argument('--mel_nband', type=int, default=80)

    # === extract_onsets ===
    parser_onsets = subparsers.add_parser('extract_onsets', help='Extract onsets')
    parser_onsets.add_argument('--dataset_fps', nargs='+', default=[
        'json/songs/songs_train.txt',
        'json/songs/songs_test.txt',
        'json/songs/songs_valid.txt'
    ])
    parser_onsets.add_argument('--out_dir', default='onset/songs/')
    parser_onsets.add_argument('--feats_dir', default='feats/songs/')
    parser_onsets.add_argument('--nsamples', type=int, default=32)

    # === extract_syms ===
    parser_syms = subparsers.add_parser('extract_syms', help='Extract symmetries')
    parser_syms.add_argument('--dataset_fps', nargs='+', default=[
        'json/songs/songs_train.txt',
        'json/songs/songs_test.txt',
        'json/songs/songs_valid.txt'
    ])
    parser_syms.add_argument('--out_dir', default='sym/songs/')
    parser_syms.add_argument('--feats_dir', default='feats/songs/')
    parser_syms.add_argument('--radius', type=int, default=4)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args_filtered = {k: v for k, v in vars(args).items() if k != 'command'}

    if args.command == 'extract_jsons':
        extract_jsons(**args_filtered)
    elif args.command == 'extract_feats':
        extract_feats(**args_filtered)
    elif args.command == 'extract_onsets':
        extract_onsets(**args_filtered)
    elif args.command == 'extract_syms':
        extract_syms(**args_filtered)
    else:
        extract_jsons(**args_filtered)
        extract_feats(**args_filtered)

        t1 = threading.Thread(target=extract_onsets, kwargs=args_filtered)
        t2 = threading.Thread(target=extract_syms, kwargs=args_filtered)

        t1.start()
        t2.start()

        t1.join()
        t2.join()
