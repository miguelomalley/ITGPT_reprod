import pickle
import random
import tqdm
from util import *
import gc
from tqdm import tqdm
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json

def split_chunks(newsong, min_len=1000, max_len=2000):
    """
    Split newsong into chunks between min_len and max_len based on 'next' value in newsong[0].
    Then merge consecutive small chunks into larger chunks, without exceeding max_len.
    """
    steps = len(newsong[0])
    chunks = []
    start = 0

    # --- Initial splitting ---
    while start < steps:
        remaining = steps - start
        if remaining <= max_len:
            # Last chunk: append as-is
            chunks.append([
                newsong[0][start:],
                newsong[1][start:],
                newsong[2][start:]
            ])
            break

        # Determine candidate split points
        search_start = start + min_len
        search_end = min(start + max_len, steps)
        best_split = search_start
        best_next_val = -float("inf")

        for i in range(search_start, search_end):
            _, nxt = newsong[0][i]
            if nxt > best_next_val:
                best_next_val = nxt
                best_split = i + 1  # include this step

        # Make sure we never exceed max_len
        if best_split - start > max_len:
            best_split = start + max_len

        chunks.append([
            newsong[0][start:best_split],
            newsong[1][start:best_split],
            newsong[2][start:best_split]
        ])

        start = best_split

    # --- Backpass: merge chains of small chunks ---
    merged_chunks = []
    current_chain = chunks[0]

    for next_chunk in chunks[1:]:
        # Check if merging current chain with next_chunk exceeds max_len
        if len(current_chain[0]) + len(next_chunk[0]) <= max_len:
            # Merge into current chain
            current_chain = [
                current_chain[0] + next_chunk[0],
                current_chain[1] + next_chunk[1],
                current_chain[2] + next_chunk[2]
            ]
        else:
            # Save current chain and start new chain
            merged_chunks.append(current_chain)
            current_chain = next_chunk

    # Append the final chain
    merged_chunks.append(current_chain)

    return merged_chunks


def generatorify_from_fp_list_sym(dataset_fp_list, 
                                   audio_radius=20,
                                   narrow_types=4,
                                   max_chunk=2000,
                                   min_chunk=1000,
                                   use_diff=False,
                                   prefetch_size=5,
                                   cache_len_fp="epoch_len_cache.json"): 
    """
    Generator with async multi-chart prefetch.
    """
    random.shuffle(dataset_fp_list)
    executor = ThreadPoolExecutor(max_workers=prefetch_size)

    def _load_chart(fp):
        """Load charts and features from a file path and normalize feats."""
        with open(fp, 'rb') as f:
            loaded = pickle.load(f)
        charts, feats_fp = loaded[0], loaded[1]
        with open(feats_fp, 'rb') as f:
            feats = pickle.load(f)
        mean = np.mean(feats, axis=0)
        std = np.std(feats, axis=0)
        feats = (feats - mean) / std
        del loaded
        return charts, feats
    
    def pre_chunk():
        num_chunks = 0
        for fp in tqdm(dataset_fp_list):
            with open(fp, 'rb') as f:
                loaded = pickle.load(f)
            charts = loaded[0]
            for chart in charts:
                splitter = [a[0] for a in chart]
                if len(splitter) == 0:
                    #print('Dead reference: ' + fp)
                    continue
                splitter.append(0)
                splitter = [[splitter[i], splitter[i+1]] for i in range(len(splitter)-1)]
                if len(splitter) <= max_chunk:
                    num_chunks += 1
                else:
                    splitter = [splitter, splitter, splitter]
                    splitter = split_chunks(splitter, min_len=min_chunk, max_len=max_chunk)
                    num_chunks += len(splitter)
        return num_chunks
    
    if os.path.exists(cache_len_fp):
        with open(cache_len_fp, "r") as f:
            epoch_len = json.load(f).get("epoch_len", None)
    else:
        epoch_len = None
    if epoch_len is None:
        print("🔎 Counting chunks to determine epoch length...")
        epoch_len = pre_chunk()
        with open(cache_len_fp, "w") as f:
            json.dump({"epoch_len": epoch_len}, f)
        print(f"✅ Counted {epoch_len} chunks; saved to {cache_len_fp}")

    def _gener():
        chunk_buffer = []
        prefetch_queue = []  # list of futures
        file_index = 0

        # Initial prefetch
        for _ in range(prefetch_size):
            future = executor.submit(_load_chart, dataset_fp_list[file_index])
            prefetch_queue.append(future)
            file_index = (file_index + 1) % len(dataset_fp_list)

        charts, feats = None, None
        chart_tape = 0
        chart_out = True

        while True:
            # Yield buffered chunks first
            if chunk_buffer:
                ch0, ch1, ch2 = chunk_buffer.pop(0)
                yield ch2, ch0, ch1
                continue

            # Pop next preloaded chart
            if chart_out:
                future = prefetch_queue.pop(0)
                charts, feats = future.result()

                # Start async prefetch for next file
                future_next = executor.submit(_load_chart, dataset_fp_list[file_index])
                prefetch_queue.append(future_next)
                file_index = (file_index + 1) % len(dataset_fp_list)

                chart_tape = 0
                chart_out = False
            
            try: chart = charts[chart_tape]
            except: 
                chart_out=True
                del(charts)
                continue
            chart_tape += 1
            if chart_tape == len(charts):
                del(charts)
                chart_out = True

            newsong = [[a[i] for a in chart] for i in range(3)]
            if len(newsong[0]) == 0:
                #print('Dead reference: ' + dataset_fp_list[(file_index - 1) % len(dataset_fp_list)])
                continue

            try:
                diff = chart[0][3]
            except:
                diff = 0
            del chart

            newsong[0].append(0)
            if use_diff:
                newsong[0] = [[newsong[0][i], newsong[0][i+1], diff] for i in range(len(newsong[0])-1)]
            else:
                newsong[0] = [[newsong[0][i], newsong[0][i+1]] for i in range(len(newsong[0])-1)]

            newsong[1] = [sparse_to_categorical(sparceify([int(a) for a in list(b)]), (narrow_types**4)-1)
                           for b in newsong[1]]

            newsong[2] = [make_onset_feature_context(feats, int(slice), audio_radius)
                           for slice in newsong[2]]

            if chart_out:
                del feats

            #gc.collect()

            # Chunk large sequences
            if len(newsong[0]) > max_chunk:
                chunk_buffer = split_chunks(newsong, min_len=min_chunk, max_len=max_chunk)
                ch0, ch1, ch2 = chunk_buffer.pop(0)
                yield ch2, ch0, ch1
            else:
                yield newsong[2], newsong[0], newsong[1]

    return _gener(), epoch_len



def get_inputs_and_gens_sym(trn_fp, 
                            tst_fp, 
                            audio_radius = 20,
                           narrow_types = 4,
                           max_chunk=2000,
                           min_chunk=1000,
                           use_diff = False,
                           use_all_charts = False):
    trn_ds, tst_ds = get_dataset_fp_list(trn_fp, tst_fp)

    train_gen, train_len = generatorify_from_fp_list_sym(trn_ds, 
                                          audio_radius=audio_radius,
                                             narrow_types = narrow_types,
                                             use_diff = use_diff,
                                             max_chunk=max_chunk,
                                                min_chunk=min_chunk,
                                                cache_len_fp=f"epoch_len_cache_train_sym{max_chunk}.json")
    test_gen, test_len = generatorify_from_fp_list_sym(tst_ds, 
                                         audio_radius=audio_radius,
                                            narrow_types=narrow_types,
                                            use_diff = use_diff,
                                            max_chunk=max_chunk,
                                               min_chunk=min_chunk,
                                               cache_len_fp=f"epoch_len_cache_test_sym{max_chunk}.json")

    return train_gen, test_gen, train_len, test_len