import random
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ConvLSTM2D, LSTM
import numpy as np
from util import *
import os
import gc
import copy as c
import pickle
from essentia.standard import MetadataReader
import shutil
import math
from tqdm import tqdm

seed = 420

def generatorify_from_fp_list_onset(dataset_fp_list, 
                              enc_dict, 
                              memlen = 7, 
                              batch_size = 32, 
                              mem_size = 1000, 
                              shuffle = False,
                                full_bidirectional = False,
                                use_all_charts = False):
    random.shuffle(dataset_fp_list)

    def _gener():
        k = 0
        chart_out = True
        song = [[]]
        while True:
            yielder = None
            while len(song[0]) < mem_size:
                if chart_out:     
                    with open(dataset_fp_list[k], 'rb') as f:
                        charts = pickle.load(f)
                    feats_fp = charts[3]
                    with open(feats_fp, 'rb') as f:
                        feats = pickle.load(f)
                    k = (k + 1) % (len(dataset_fp_list) - 1)
                    chart_tape = 0
                    chart_out = False
                if not use_all_charts:
                    try:
                        chart_num = random.randint(0,len(charts[0])-1)
                        newsong = [charts[i][chart_num] for i in range(3)]
                        chart_out = True
                    except:
                        del(charts)
                        chart_out = True
                        continue
                    del(charts)
                else:
                    newsong = [charts[i][chart_tape] for i in range(3)]
                    chart_tape += 1
                    if chart_tape == len(charts[0]):
                        del(charts)
                        chart_out = True
                
                newsong[0] = [make_onset_feature_context_range(feats, x[0], x[1]) for x in newsong[0]]
                del(feats)
                mean = np.mean(newsong[0], axis = 0)
                std = np.std(newsong[0], axis = 0)
                newsong[0] = (np.array(newsong[0])-mean)/std
                del(mean, std)

                gc.collect()
                if len(gc.garbage) > 0:
                    print(gc.garbage)

                

                newsong[1] = [[a[0],a[1]] for a in newsong[1]]

                if len(newsong[0])>mem_size:
                    take_windows = np.random.choice(range(len(newsong[0])), mem_size, replace = False)
                    newsong[2] = [newsong[2][i] for i in take_windows]
                    newsong.append(windowize(newsong[0], front_set = 'min', go_backwards = True, frames = memlen, take_windows=take_windows, return_type='list'))
                    newsong[0] = windowize(newsong[0], front_set = 'min', frames = memlen, take_windows=take_windows, return_type='list')
                    newsong.append(windowize(newsong[1], go_backwards = True, frames = memlen, take_windows=take_windows, return_type='list'))
                    newsong[1] = windowize(newsong[1], frames = memlen, take_windows=take_windows, return_type='list')
                else:
                    newsong.append(windowize(newsong[0], front_set = 'min', go_backwards = True, frames = memlen, return_type='list'))
                    newsong[0] = windowize(newsong[0], front_set = 'min', frames = memlen, return_type='list')
                    newsong.append(windowize(newsong[1], go_backwards = True, frames = memlen, return_type='list'))
                    newsong[1] = windowize(newsong[1], frames = memlen, return_type='list')
                    
                
                if len(song) == 1:
                    song = newsong
                else:
                    for j in range(5):
                        song[j].extend(newsong[j])
                del(newsong)
                gc.collect()

                if shuffle and len(song[0]) >= mem_size:
                    indices = np.random.permutation(len(song[0]))
                    for j in range(5):
                        song[j] = [song[j][i] for i in indices]
                    del(indices)
                    gc.collect()
                

                
            assert len(song[0])>batch_size+memlen

            success_take = 0
            miss_take = 0
            ac = []
            sd = []
            ac2 = []
            sd2 = []
            lb = []
            i = 0
            while success_take<batch_size:
                ac.append(song[0][i])
                ac2.append(song[3][i])
                sd.append(song[1][i])
                sd2.append(song[4][i])
                lb.append(enc_dict[song[2][i]])
                success_take += 1
                i += 1

            ac, sd, ac2, sd2, lb = np.array(ac), np.array(sd), np.array(ac2), np.flip(np.array(sd2), 1), np.array(lb)
            for j in range(5):
                song[j] = song[j][int(batch_size+miss_take):]
            assert(len(song[0])==len(song[1]) and len(song[1])==len(song[2]))
            gc.collect()
            if full_bidirectional:
                sd2 = np.flip(np.array(sd2), 1)
                ac = np.array([np.concatenate((ac[j],ac2[j][1:]), axis = 0) for j in range(batch_size)])
                sd = np.array([np.concatenate((sd[j],sd2[j][1:]), axis = 0) for j in range(batch_size)])
                yielder = ((ac,sd), lb)
            else:
                yielder = ((ac,ac2,sd,sd2), lb)
            del(ac, sd, ac2, sd2, lb)
            yield yielder
    return _gener()

def get_inputs_and_gens_onset(trn_fp, 
                              tst_fp, 
                              lbl_fp, 
                              shuffle = False, 
                              batch_size = 32, 
                              memlen = 7, 
                              nframes = 8, 
                              nmelbands = 80, 
                              nchannels = 3,
                              mem_size = 1000,
                             full_bidirectional = False,
                             use_all_charts = False):
    trn_ds, tst_ds = get_dataset_fp_list(trn_fp, tst_fp)
    with open(lbl_fp, 'rb') as f:
        labels = pickle.load(f)
    enc_dict = label_to_vect_dict(labels, force_max_len=48)
    if full_bidirectional:
        inp_shape_0 = (None,2*memlen+1,nframes,nmelbands,nchannels)
        inp_shape_1 = (None,2*memlen+1,2)
    else:
        inp_shape_0 = (None,memlen+1,nframes,nmelbands,nchannels)
        inp_shape_1 = (None,memlen+1,2)

    train_gen = generatorify_from_fp_list_onset(trn_ds, 
                                          enc_dict, 
                                          batch_size=batch_size, 
                                          shuffle = shuffle, 
                                          mem_size=mem_size, 
                                          memlen = memlen,
                                               full_bidirectional = full_bidirectional,
                                               use_all_charts=use_all_charts)
    test_gen = generatorify_from_fp_list_onset(tst_ds, 
                                         enc_dict, 
                                         batch_size=batch_size, 
                                         shuffle = shuffle, 
                                         mem_size=mem_size,
                                         memlen = memlen,
                                              full_bidirectional = full_bidirectional,
                                              use_all_charts=use_all_charts)

    audio_ctx_inp = Input(shape = inp_shape_0[1:], batch_size = batch_size)
    audio_ctx_inp2 = Input(shape = inp_shape_0[1:], batch_size = batch_size)
    stream_inp = Input(shape = inp_shape_1[1:], batch_size = batch_size)
    stream_inp2 = Input(shape = inp_shape_1[1:], batch_size = batch_size)

    if full_bidirectional:
        return train_gen, test_gen, audio_ctx_inp, stream_inp
    else:
        return train_gen, test_gen, audio_ctx_inp, stream_inp, audio_ctx_inp2, stream_inp2



def get_onset_model(audio_ctx_inp, 
                    stream_inp, 
                    audio_ctx_inp2 = None, 
                    stream_inp2 = None, 
                    hist_inp = None, 
                    full_bidirectional = False, 
                    conv3d = False, 
                    use_history = False, 
                    memlen = 15):
    if not full_bidirectional and not conv3d:
        audio_proc = layers.ConvLSTM2D(16, (7,3), return_sequences = True)(audio_ctx_inp)
        audio_proc = layers.MaxPooling3D((1,1,3), strides = (1,1,3))(audio_proc)
        audio_proc = layers.ConvLSTM2D(32, (3,3), return_sequences = True)(audio_proc)
        audio_proc = layers.MaxPooling3D((1,1,3), strides = (1,1,3))(audio_proc)
        
        audio_proc2 = layers.ConvLSTM2D(16, (7,3), return_sequences = True, go_backwards = True)(audio_ctx_inp2)
        audio_proc2 = layers.MaxPooling3D((1,1,3), strides = (1,1,3))(audio_proc2)
        audio_proc2 = layers.ConvLSTM2D(32, (3,3), return_sequences = True)(audio_proc2)
        audio_proc2 = layers.MaxPooling3D((1,1,3), strides = (1,1,3))(audio_proc2)
        
        audio_out = layers.Reshape((memlen+1,-1))(audio_proc)
        audio_out2 = layers.Reshape((memlen+1,-1))(audio_proc2)


        stream_merge = layers.Concatenate(axis = -1)([audio_out, stream_inp])
        stream_merge2 = layers.Concatenate(axis = -1)([audio_out2, stream_inp2])
    
        note_comp = layers.LSTM(200, return_sequences = True, dropout = .2)(stream_merge)
        note_comp = layers.LSTM(200, dropout = .2)(note_comp)
        
        note_comp2 = layers.LSTM(200, return_sequences = True, dropout = .2)(stream_merge2)
        note_comp2 = layers.LSTM(200, dropout = .2)(note_comp2)

        if use_history:
            hist_comp = layers.LSTM(200, return_sequences = True)(hist_inp)
            hist_comp = layers.LSTM(200, return_sequences = False)(hist_comp)
            note_comp = layers.Concatenate(axis = 1)([note_comp, note_comp2, hist_comp])

        else:
            note_comp = layers.Concatenate(axis = 1)([note_comp, note_comp2])

    elif conv3d:
        audio_proc = layers.Conv3D(16, (3,7,3), padding = 'same')(audio_ctx_inp)
        audio_proc = layers.MaxPooling3D((1,1,3), strides = (1,1,3))(audio_proc)
        audio_proc = layers.Conv3D(32, (3,3,3), padding = 'same')(audio_proc)
        audio_proc = layers.MaxPooling3D((1,1,3), strides = (1,1,3))(audio_proc)
        audio_out = layers.Reshape((2*memlen+1,-1))(audio_proc)
        stream_merge = layers.Concatenate(axis = -1)([audio_out, stream_inp])
        note_comp = layers.Bidirectional(LSTM(200, return_sequences = True, dropout = .2))(stream_merge)
        note_comp = layers.Bidirectional(LSTM(200, return_sequences = False, dropout = .2))(stream_merge)
    elif full_bidirectional:
        audio_proc = layers.Bidirectional(ConvLSTM2D(16, (7,3), return_sequences = True))(audio_ctx_inp)
        audio_proc = layers.MaxPooling3D((1,1,3), strides = (1,1,3))(audio_proc)
        audio_proc = layers.Bidirectional(ConvLSTM2D(32, (3,3), return_sequences = True))(audio_proc)
        audio_proc = layers.MaxPooling3D((1,1,3), strides = (1,1,3))(audio_proc)
        audio_out = layers.Reshape((2*memlen+1,-1))(audio_proc)
        stream_merge = layers.Concatenate(axis = -1)([audio_out, stream_inp])
        note_comp = layers.Bidirectional(LSTM(200, return_sequences = True, dropout = .2))(stream_merge)
        note_comp = layers.Bidirectional(LSTM(200, return_sequences = False, dropout = .2))(stream_merge)
    
    note_comp = layers.Dense(512, activation = 'leaky_relu')(note_comp)
    note_comp = layers.Dropout(.2)(note_comp)
    note_comp = layers.Dense(256, activation = 'leaky_relu')(note_comp)
    note_comp = layers.Dropout(.2)(note_comp)
    
    output = layers.Dense(48, activation = 'sigmoid')(note_comp)
    
    if not full_bidirectional and not use_history:
        model = Model([audio_ctx_inp, audio_ctx_inp2, stream_inp, stream_inp2], output)
    elif use_history:
        model = Model([audio_ctx_inp, audio_ctx_inp2, stream_inp, stream_inp2, hist_inp], output)
    else:
        model = Model([audio_ctx_inp, stream_inp], output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm = 1),
        loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits = False),
        metrics=[
        tf.keras.metrics.AUC(from_logits = False, curve = 'PR', name = 'auc'),
        tf.keras.metrics.F1Score(average = 'micro', threshold = .5, name = 'f1'),
        tf.keras.metrics.F1Score(average = 'micro', threshold = .4, name = 'f1_.4'),
        tf.keras.metrics.BinaryAccuracy(name = 'acc'),
    ],
    )
    
    print(model.summary())
    return model

def train_onset_model(stream_labels_fp='onset/songs/stream_labels.pkl',
                    shuffle = True,
                    batch_size = 32,
                    memlen = 15,
                    mem_size = 2500,
                    nframes = 32,
                    steps_per_epoch = 400,
                    nepochs = 300,
                    nmelbands = 80,
                    nchannels = 3,
                    model_dir = 'trained_models',
                    train_txt_fp = 'onset/songs/songs_train.txt',
                    test_txt_fp = 'onset/songs/songs_test.txt',
                    load_checkpoint = False,
                    full_bidirectional = False,
                    conv3d = False,
                    model_name = 'onset',
                    use_all_charts = False,
                    use_early_stop = True,
                    use_scheduler = True,
                    use_history = False):
    checkpoint_name = model_name + '_checkpoint.keras'
    model_name = model_name + '_model.keras'

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    if not full_bidirectional and not conv3d and not use_history:
        train_gen, test_gen, audio_ctx_inp, stream_inp, audio_ctx_inp2, stream_inp2 = get_inputs_and_gens_onset(train_txt_fp, 
                                                                                                                test_txt_fp, 
                                                                                                                stream_labels_fp, 
                                                                                                                shuffle, 
                                                                                                                batch_size=batch_size, 
                                                                                                                memlen = memlen,
                                                                                                                nframes=nframes,
                                                                                                                nmelbands=nmelbands,
                                                                                                                nchannels=nchannels,
                                                                                                                mem_size = mem_size,
                                                                                                                full_bidirectional = full_bidirectional,
                                                                                                                use_all_charts=use_all_charts)
    elif full_bidirectional or conv3d:
        train_gen, test_gen, audio_ctx_inp, stream_inp = get_inputs_and_gens_onset(train_txt_fp, 
                                                                                    test_txt_fp, 
                                                                                    stream_labels_fp, 
                                                                                    shuffle, 
                                                                                    batch_size=batch_size, 
                                                                                    memlen = memlen,
                                                                                    nframes=nframes,
                                                                                    nmelbands=nmelbands,
                                                                                    nchannels=nchannels,
                                                                                    mem_size = mem_size,
                                                                                    full_bidirectional = full_bidirectional,
                                                                                    use_all_charts=use_all_charts)
        audio_ctx_inp2 = None
        stream_inp2 = None
        
    model = get_onset_model(audio_ctx_inp, 
                            stream_inp, 
                            audio_ctx_inp2 = audio_ctx_inp2, 
                            stream_inp2 = stream_inp2, 
                            full_bidirectional = full_bidirectional, 
                            conv3d = conv3d, 
                            use_history = use_history, 
                            memlen = memlen)
    
    checkpoint_filepath = os.path.join(model_dir, checkpoint_name)
    if load_checkpoint:
        if os.path.isfile(checkpoint_filepath):
            print(True)
            model.load_weights(checkpoint_filepath)
        
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose = 0,
        save_best_only = True,
        monitor = 'val_auc',
        mode = 'max')
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=20,
        restore_best_weights=True,
        mode = 'max',
        start_from_epoch = 100
    )

    callybacks = [model_checkpoint_callback]
    if use_early_stop:
        callybacks.append(early_stopping)
    if use_scheduler:
        callybacks.append(lr_scheduler)
    
    
    model.fit(train_gen, 
              batch_size = batch_size, 
              epochs = nepochs, 
              steps_per_epoch = steps_per_epoch, 
              validation_steps = 200, 
              validation_data = test_gen, 
              callbacks = callybacks)
    
    model.save(model_dir + '/' + model_name)







def generatorify_from_fp_list_sym(dataset_fp_list, 
                              memlen = 7, 
                              aud_memlen = 7,
                              audio_radius = 20,
                              batch_size = 50, 
                              mem_size = 5000, 
                              n_predictions = 1,
                              shuffle = False, 
                              bidirectional_audio = True,
                                 narrow_types = 4,
                                 use_diff = False,
                                 use_all_charts = False):
    random.shuffle(dataset_fp_list)
    def _gener():
        k = 0
        song = [[]]
        chart_out = True

        while True:
            if len(song[0]) < mem_size/2:
                while len(song[0]) < mem_size:
                    if chart_out:     
                        with open(dataset_fp_list[k], 'rb') as f:
                            loaded = pickle.load(f)
                        charts, feats_fp = loaded[0], loaded[1]
                        with open(feats_fp, 'rb') as f:
                            feats = pickle.load(f)
                        del(loaded)
                        k = (k + 1) % (len(dataset_fp_list) - 1)
                        chart_tape = 0
                        chart_out = False
                    if not use_all_charts:
                        try:
                            chart = random.choice(charts)
                            del(charts)
                            chart_out = True
                        except:
                            del(charts)
                            chart_out = True
                            continue
                    else:
                        chart = charts[chart_tape]
                        chart_tape += 1
                        if chart_tape == len(charts):
                            del(charts)
                            chart_out = True
                    
                    newsong = [[a[i] for a in chart] for i in range(3)]
                    if len(newsong[0]) == 0:
                        print('Dead reference: ' + dataset_fp_list[(k - 1) % (len(dataset_fp_list) - 1)-1])
                        continue

                    try: diff = chart[0][3]
                    except: diff = 0
                    del(chart)

                    newsong[0].append(0)
                    if use_diff:
                        newsong[0] = [[newsong[0][i], newsong[0][i+1], diff] for i in range(len(newsong[0])-1)]
                    else:
                        newsong[0] = [[newsong[0][i], newsong[0][i+1]] for i in range(len(newsong[0])-1)]
                    newsong[1] = [sparse_to_categorical(sparceify([int(a) for a in list(b)]), 255) for b in newsong[1]]
                    if len(newsong[0])>mem_size:
                        take_windows = np.random.choice(range(len(newsong[0])), mem_size, replace = False)
                        newsong[1] = windowize(np.array(newsong[1]), frames=memlen + n_predictions-1, take_windows=take_windows)
                        newsong[0] = windowize(np.array(newsong[0]), frames = memlen + n_predictions-1, take_windows=take_windows)
                        newsong[1] = list(np.concatenate((newsong[1],newsong[0]), axis = -1))
                        if bidirectional_audio:
                            newsong[2], newsong[0] = windowize(newsong[2], frames = aud_memlen, front_set = 'min', take_windows=take_windows, return_type='list'), windowize(newsong[2], frames = aud_memlen+n_predictions-1, front_set = 'min', go_backwards = True, take_windows=take_windows, return_type='list')
                            newsong[2] = [[make_onset_feature_context(feats, int(slice), audio_radius) for slice in window] for window in newsong[2]]
                            newsong[0] = [[make_onset_feature_context(feats, int(slice), audio_radius) for slice in window] for window in newsong[0]]
                        else:
                            newsong[2] = [newsong[2][w] for w in take_windows]
                            newsong[0] = newsong[2] 
                            newsong[2] = [make_onset_feature_context(feats, int(slice), audio_radius) for slice in newsong[2]]                           
                    else:
                        newsong[1] = windowize(np.array(newsong[1]), frames=memlen+ n_predictions-1)
                        newsong[0] = windowize(np.array(newsong[0]), frames = memlen+ n_predictions-1)
                        newsong[1] = list(np.concatenate((newsong[1],newsong[0]), axis = -1))
                        if bidirectional_audio:
                            newsong[2], newsong[0] = windowize(newsong[2], frames = aud_memlen, front_set = 'min', return_type='list'), windowize(newsong[2], frames = aud_memlen+n_predictions-1, front_set = 'min', go_backwards = True, return_type='list')
                            newsong[2] = [[make_onset_feature_context(feats, int(slice), audio_radius) for slice in window] for window in newsong[2]]
                            newsong[0] = [[make_onset_feature_context(feats, int(slice), audio_radius) for slice in window] for window in newsong[0]]
                        else:
                            newsong[0] = newsong[2]
                            newsong[2] = [make_onset_feature_context(feats, int(slice), audio_radius) for slice in newsong[2]]
                    
                    if len(song) == 1:
                        song = newsong
                    else:
                        for j in range(3):
                            song[j].extend(newsong[j])
                    del(newsong)
                    gc.collect()

                    if shuffle and len(song[0]) >= mem_size:
                        indices = np.random.permutation(len(song[0]))
                        for j in range(3):
                            song[j] = [song[j][i] for i in indices]
                        del(indices)
                        gc.collect()
            gc.collect()
                
            assert len(song[1]) == len(song[2])

            
            success_take = 0
            miss_take = 0
            ac = []
            ac2 = []
            sd = []
            lb = []
            i = 0
            while success_take<batch_size:
                if sum(song[1][i][-1])!=0:
                    if bidirectional_audio:
                        ac.append(song[2][i])
                        ac2.append(song[0][i])
                    else:
                        ac.append(song[2][i])
                    sd.append(list(song[1][i][:-(n_predictions)])+[[0 for j in range(narrow_types**4)]+list(song[1][i][-x][-(2+use_diff):]) for x in reversed(range(1,n_predictions+1))])
                    lb.append([song[1][i][-x][:-(2+use_diff)] for x in reversed(range(1,n_predictions+1))])
                    success_take += 1
                else:
                    miss_take += 1
                i+=1
            ac, ac2, sd, lb = np.array(ac), np.array(ac2), np.array(sd), np.squeeze(np.array(lb))
            for j in range(3):
                song[j] = song[j][batch_size+miss_take:]

            if bidirectional_audio:
                yield(ac,ac2,sd), lb
            else:
                yield (ac,sd), lb
    return _gener()



def get_inputs_and_gens_sym(trn_fp, 
                            tst_fp, 
                            shuffle = False, 
                            batch_size = 1000, 
                            bidirectional_audio = True,
                            memlen = 64, 
                            aud_memlen = 15, 
                            mem_size = 5000, 
                            audio_radius = 20,
                           narrow_types = 4,
                           use_diff = False,
                           n_predictions = 1,
                           use_all_charts = False):
    trn_ds, tst_ds = get_dataset_fp_list(trn_fp, tst_fp)

    train_gen = generatorify_from_fp_list_sym(trn_ds, 
                                          batch_size=batch_size, 
                                          shuffle = shuffle, 
                                          mem_size=mem_size, 
                                          memlen=memlen, 
                                          aud_memlen=aud_memlen,
                                          audio_radius=audio_radius,
                                          bidirectional_audio=bidirectional_audio,
                                             narrow_types = narrow_types,
                                             use_diff = use_diff,
                                             n_predictions = n_predictions,
                                             use_all_charts=use_all_charts)
    test_gen = generatorify_from_fp_list_sym(tst_ds, 
                                         batch_size=batch_size, 
                                         shuffle = shuffle, 
                                         mem_size=mem_size, 
                                         memlen=memlen, 
                                         aud_memlen=aud_memlen,
                                         audio_radius=audio_radius,
                                         bidirectional_audio=bidirectional_audio,
                                            narrow_types=narrow_types,
                                            use_diff = use_diff,
                                            n_predictions = n_predictions,
                                            use_all_charts=use_all_charts)

    if bidirectional_audio:
        inp_shape_0_left = (None,aud_memlen+1,2*audio_radius+1,80,3)
        inp_shape_0_right = (None,aud_memlen+n_predictions,2*audio_radius+1,80,3)
    else:
        inp_shape_0_left = (None,2*audio_radius+1,80,3)
        inp_shape_0_right = inp_shape_0_left
    inp_shape_1 = (None, memlen+n_predictions, 258+use_diff)

    audio_ctx_inp = Input(shape = inp_shape_0_left[1:], batch_size = batch_size)
    audio_ctx_inp2 = Input(shape = inp_shape_0_right[1:], batch_size = batch_size)
    sym_inp = Input(shape = inp_shape_1[1:], batch_size = batch_size)

    return train_gen, test_gen, audio_ctx_inp, audio_ctx_inp2, sym_inp

def get_sym_model(audio_ctx_inp, 
                  sym_inp, 
                  audio_ctx_inp2=None, 
                  audio_to_history = False, 
                  bidirectional_audio = True, 
                  aud_memlen = 7, 
                  memlen = 64):
    if bidirectional_audio:
        audio_proc = layers.ConvLSTM2D(8, (9,3), return_sequences=True, padding = 'same')(audio_ctx_inp)
        audio_proc = layers.MaxPooling3D((1,1,3), strides = (1,1,3))(audio_proc)

        audio_proc2 = layers.ConvLSTM2D(8, (9,3), return_sequences=True, go_backwards = True, padding = 'same')(audio_ctx_inp2)
        audio_proc2 = layers.MaxPooling3D((1,1,3), strides = (1,1,3))(audio_proc2)

        audio_proc2 = layers.ConvLSTM2D(16, (3,3), return_sequences=False)(audio_proc2)
        audio_proc2 = layers.MaxPooling2D((1,3), strides = (1,3))(audio_proc2)

        audio_out = layers.Flatten()(audio_proc2)
        if not audio_to_history:
            audio_proc = layers.ConvLSTM2D(16, (3,3), return_sequences=False)(audio_proc)
            audio_proc = layers.MaxPooling2D((1,3), strides = (1,3))(audio_proc)
            audio_out_backward = layers.Flatten()(audio_proc)
            audio_out = layers.Concatenate(axis = -1)([audio_out_backward, audio_out])
        else:
            audio_proc = layers.ConvLSTM2D(16, (3,3), padding = 'same', return_sequences=True)(audio_proc)
            audio_out_backward = layers.MaxPooling3D((1,1,3), strides = (1,1,3))(audio_proc)
    
    else:
        audio_proc = layers.BatchNormalization()(audio_ctx_inp)
        audio_proc = layers.Conv2D(16, (7,3))(audio_proc)
    
        audio_proc = layers.MaxPooling2D((1,3), strides = (1,3))(audio_proc)
    
        audio_proc = layers.Conv2D(32, (3,3))(audio_proc)
        audio_proc = layers.Conv2D(64, (3,3))(audio_proc)
        audio_proc = layers.Conv2D(64, (3,3))(audio_proc)
    
        audio_proc = layers.MaxPooling2D((1,3), strides = (1,3))(audio_proc)
    
        audio_out = layers.Flatten()(audio_proc)
        
    if audio_to_history:
        audio_out_backward = layers.Reshape((aud_memlen+1, -1))(audio_out_backward)
        audio_out_backward = layers.ZeroPadding1D(padding = (memlen-aud_memlen,0))(audio_out_backward)
        sym_proc = layers.Concatenate(axis = -1)([sym_inp, audio_out_backward])
        sym_proc = layers.LSTM(256, return_sequences = True,  dropout = .5)(sym_proc)
        sym_proc = layers.LSTM(256, return_sequences = False,  dropout = .5)(sym_proc)
    else:
        sym_proc = layers.LSTM(256, return_sequences = True)(sym_inp)
        sym_proc = layers.LSTM(256, return_sequences = False)(sym_proc)

    stream_merge = layers.Concatenate(axis = -1)([audio_out, sym_proc])

    full_con = layers.Dense(512, activation = 'leaky_relu')(stream_merge)
    full_con = layers.Dropout(.5)(full_con)
    full_con = layers.Dense(256, activation = 'leaky_relu')(full_con)
    full_con = layers.Dropout(.5)(full_con)
    
    output = layers.Dense(256, activation = 'softmax')(full_con)
    
    if bidirectional_audio:
        model = Model([audio_ctx_inp, audio_ctx_inp2, sym_inp], output)
    else:
        model = Model([audio_ctx_inp, sym_inp], output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False),
        metrics=[
        tf.keras.metrics.CategoricalAccuracy(name = 'acc'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name = 'top2acc'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name = 'top3acc'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name = 'top5acc'),
    ],
    )
    print(model.summary())
    return model


def train_sym_model(shuffle = True,
                    batch_size = 64,
                    steps_per_epoch = 400,
                    nepochs = 400,
                    bidirectional_audio = True,
                    audio_to_history = False,
                    aud_memlen = 7,
                    memlen = 64,
                    mem_size = 5000,
                    audio_radius = 20,
                    narrow_types = 4,
                    n_predictions = 1,
                    train_txt_fp = 'sym/songs/songs_train.txt',
                    test_txt_fp = 'sym/songs/songs_test.txt',
                    model_dir = 'trained_models',
                    model_name = 'sym',
                    load_checkpoint = True,
                   use_diff = False,
                   use_all_charts = False,
                   use_scheduler = True,
                   use_early_stop = True):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    
    train_gen, test_gen, audio_ctx_inp, audio_ctx_inp2, sym_inp = get_inputs_and_gens_sym(train_txt_fp,
                                                                                            test_txt_fp, 
                                                                                            shuffle, 
                                                                                            batch_size=batch_size, 
                                                                                            bidirectional_audio = bidirectional_audio,
                                                                                            memlen=memlen, 
                                                                                            aud_memlen=aud_memlen,
                                                                                            narrow_types = narrow_types,
                                                                                            use_diff = use_diff,
                                                                                            mem_size = mem_size,
                                                                                            audio_radius=audio_radius,
                                                                                            n_predictions = n_predictions,
                                                                                            use_all_charts=use_all_charts)
    model = get_sym_model(audio_ctx_inp, 
                        sym_inp, 
                        audio_ctx_inp2=audio_ctx_inp2, 
                        audio_to_history = audio_to_history, 
                        bidirectional_audio = bidirectional_audio, 
                        aud_memlen = aud_memlen, 
                        memlen = memlen)


    checkpoint_name = model_name + '_checkpoint.keras'
    checkpoint_filepath = os.path.join(model_dir, checkpoint_name)
    if load_checkpoint:
        if os.path.isfile(checkpoint_filepath):
            model.load_weights(checkpoint_filepath)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose = 0,
        save_best_only = True,
        monitor = 'val_acc',
        mode = 'max')
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_acc',
        patience=20,
        restore_best_weights=True,
        mode = 'max',
        start_from_epoch = 100
    )

    callybacks = [model_checkpoint_callback]
    if use_scheduler:
        callybacks.append(lr_scheduler)
    if use_early_stop:
        callybacks.append(early_stopping)
    
    model.fit(train_gen, 
              batch_size = batch_size, 
              epochs = nepochs, 
              steps_per_epoch = steps_per_epoch, 
              validation_steps = 50, 
              validation_data = test_gen, 
              callbacks = callybacks)
    
    model.save(model_dir + '/' + model_name + '_model.keras')

def generate_charts(onset_model_fp='trained_models/onset_model.keras',
                    sym_model_fp='trained_models/sym_model.keras',
                    batch_size = 32,
                    model_frame_density = 32,
                    onset_history_len = 15,
                    threshold = .5,
                    in_directory = 'input_songs_for_generation',
                    out_directory = 'generated_charts',
                    diffs = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge'],
                    maxstep = 12,
                    use_song_length = False,
                    bpm_method = 'DDCL'
                    ):
    template = get_template()
    chart_template = get_chart_template()
    print( 'Loading step placement model')
    onset_model = load_model(onset_model_fp)
    print( 'Loading step selection model')
    sym_model = load_model(sym_model_fp)
    analyzers = create_analyzers(nhop=441)
    subdiv = maxstep*4
    if not os.path.isdir(out_directory):
        os.mkdir(out_directory)
    in_dir = os.fsdecode(in_directory)
    for song_fp in os.listdir(in_dir):
        song = os.fsdecode(song_fp)
        if song.endswith('.mp3') or song.endswith('.ogg') or song.endswith('.wav') or song.endswith('.aiff'):
            out_song_path = os.path.join(out_directory, os.path.splitext(song)[0])
            if os.path.isdir(out_song_path):
                for file in os.listdir(out_song_path):
                    filepath = os.path.join(out_song_path,file)
                    os.remove(filepath)
                os.rmdir(out_song_path)
            os.mkdir(out_song_path)
            song_in = os.path.join(in_dir, song_fp)

            song_title = os.path.splitext(os.path.basename(song_in))[0]
            beats, subdiv_beats, bpm_shift_times, offset, bpm_str, song_length, bpm = set_bpm(song_in, bpm_method=bpm_method)
            
            topdiff = int(round((bpm/10)-(3-math.log(song_length,2))))
            fine_diffs = {diffs[i]:topdiff - (4-i) for i in range(5)}
            
            meta_reader = MetadataReader(filename=song_in)
            metadata = meta_reader()
            
            try:
                artist = metadata[1]
            except:
                artist = 'Unknown Artist'
            try:
                title = metadata[0]
                title.replace(":","")
                title.replace(";","")
            except:
                title = 'Unknown Title'
                
            song_feats = extract_mel_feats(song_in, analyzers, nhop=441)

            # First phase: Step placement for all diffs
            all_placed_times = {}
            diff_chart_txts = []

            for diff in diffs:
                fine = fine_diffs[diff]
                beat_audio_contexts = []
                for i in range(len(beats)-1):
                    beat_audio_contexts.append(make_onset_feature_context_range(song_feats, beats[i], beats[i+1], frame_density=model_frame_density))
                    
                mean = np.mean(beat_audio_contexts, axis = 0)
                std = np.std(beat_audio_contexts, axis = 0)
                
                beat_audio_contexts = (np.array(beat_audio_contexts)-mean)/std

                beat_audio_contexts_backward = windowize(beat_audio_contexts, front_set='min', frames = 15)
                beat_audio_contexts_forward = windowize(beat_audio_contexts, front_set='min', go_backwards=True, frames=15)

                if use_song_length:
                    cur_str_data = [[0 for _ in range(3)] for j in range(onset_history_len)] + [[fine, bpm_shift_times[0][1], len(beats)]]
                else:
                    cur_str_data = [[0 for _ in range(2)] for j in range(onset_history_len)] + [[fine, bpm_shift_times[0][1]]]
                next_shift_index = 1
                if use_song_length:
                    new_str_data = [[fine, bpm_shift_times[0][1], len(beats)]]
                else:
                    new_str_data = [[fine, bpm_shift_times[0][1]]]

                for i in range(onset_history_len):
                    if len(bpm_shift_times) > next_shift_index:
                        if i+1>=bpm_shift_times[next_shift_index][0]:
                            new_str_data[0][1] = bpm_shift_times[next_shift_index][1]
                            next_shift_index += 1
                    cur_str_data+=new_str_data

                stream_datas = [cur_str_data]
                stream_outs = []

                for i in range(len(beats)-2):
                    new_str_data = c.deepcopy(stream_datas[-1])
                    for j in range(onset_history_len):
                        new_str_data[j] = new_str_data[j+1]

                    if len(bpm_shift_times) > next_shift_index:
                        if i+onset_history_len+1 >= bpm_shift_times[next_shift_index][0]:
                            new_str_data[-1][1] = bpm_shift_times[next_shift_index][1]
                            next_shift_index += 1
                    stream_datas += [new_str_data]

                #stream_datas = [quick_reducify(str_data, indices = [0,1]) for str_data in stream_datas]
                stream_data_pre = [str_data[:onset_history_len+1] for str_data in stream_datas]
                stream_data_post = [str_data[onset_history_len:] for str_data in stream_datas]
                
                done = False
                stream_outs = np.zeros((0,48))
                n_steps = len(beat_audio_contexts_backward)
                #while not done:
                for _ in tqdm(range((n_steps // batch_size)+1)):
                    if len(beat_audio_contexts_backward)>=batch_size:
                        stream_inp = [beat_audio_contexts_backward[:batch_size], 
                                    beat_audio_contexts_forward[:batch_size], 
                                    stream_data_pre[:batch_size], 
                                    stream_data_post[:batch_size]]
                        step_out = onset_model.predict((np.array(stream_inp[0]), 
                                                np.array(stream_inp[1]), 
                                                np.array(stream_inp[2]), 
                                                np.array(stream_inp[3])), 
                                                batch_size = batch_size, verbose = 0)
                        stream_outs = np.concatenate((stream_outs, step_out), axis = 0)
                        beat_audio_contexts_backward = beat_audio_contexts_backward[batch_size:]
                        beat_audio_contexts_forward = beat_audio_contexts_forward[batch_size:]
                        stream_data_pre = stream_data_pre[batch_size:]
                        stream_data_post = stream_data_post[batch_size:]
                    elif len(beat_audio_contexts_backward)>0:
                        step_out = onset_model.predict((np.array(beat_audio_contexts_backward), 
                                                np.array(beat_audio_contexts_forward), 
                                                np.array(stream_data_pre), 
                                                np.array(stream_data_post)), 
                                                batch_size = len(beat_audio_contexts_backward), verbose = 0)
                        stream_outs = np.concatenate((stream_outs, step_out), axis = 0)
                        done = True
                    else:
                        done = True

                stream_outs = [(out > threshold) for out in stream_outs]

                placed_times = []
                counter = 0
                for out in stream_outs:
                    cur_time = beats[counter]
                    try:
                        next_time = beats[counter+1]
                    except:
                        next_time = 2*beats[counter]-(beats[counter-1])
                    time_gap = next_time-cur_time
                    counter+=1
                    if len(out)>0:
                        listout = list(out)
                        step_gap = time_gap/len(listout)
                        for i in listout:
                            if i == '1' or i == 1 or i == True:
                                placed_times.append(cur_time)
                            cur_time+= step_gap
                buckets = (subdiv_beats[1:]+subdiv_beats[:-1])/2
                new_steps = subdiv_beats[np.digitize(placed_times, buckets)]
                placed_times = np.sort(np.unique(new_steps))

                print('Assigned {} steps for diff {}.'.format(len(placed_times), diff))
                all_placed_times[diff] = placed_times

            # Second phase: Batch step selection for all diffs
            max_steps = max(len(times) for times in all_placed_times.values())
            print(f"Maximum steps across all diffs: {max_steps}")

            # Initialize batch arrays
            batch_step_hist = np.zeros((len(diffs), 65, 258))
            batch_actx = np.ones((len(diffs), 15, 9, 80, 3)) * np.log(1e-16)
            batch_last_step = np.zeros(len(diffs))
            all_selected_steps = {diff: [] for diff in diffs}

            # Pre-populate initial context for each diff
            for diff_idx, diff in enumerate(diffs):
                placed_times = all_placed_times[diff]
                for i in range(min(8, len(placed_times))):
                    time = placed_times[i]
                    batch_actx[diff_idx, i+7] = make_onset_feature_context(song_feats, int(time*100), 4)

            # Process all steps in batches
            for step_idx in tqdm(range(max_steps)):
                # Prepare batch inputs
                valid_diffs = []
                batch_indices = []
                
                for diff_idx, diff in enumerate(diffs):
                    placed_times = all_placed_times[diff]
                    if step_idx < len(placed_times):
                        valid_diffs.append(diff)
                        batch_indices.append(diff_idx)
                
                if not valid_diffs:
                    break
                    
                # Update batch arrays for valid diffs
                current_batch_size = len(valid_diffs)
                current_step_hist = np.zeros((current_batch_size, 65, 258))
                current_actx = np.zeros((current_batch_size, 15, 9, 80, 3))
                
                for batch_pos, (diff_idx, diff) in enumerate(zip(batch_indices, valid_diffs)):
                    placed_times = all_placed_times[diff]
                    time = placed_times[step_idx]
                    
                    try: 
                        next_time = placed_times[step_idx+1]
                    except: 
                        next_time = time
                    try: 
                        right_time = placed_times[step_idx+7] if step_idx+7 < len(placed_times) else -1
                    except: 
                        right_time = -1
                    
                    # Shift history and context
                    batch_step_hist[diff_idx, :-1] = batch_step_hist[diff_idx, 1:]
                    batch_actx[diff_idx, :-1] = batch_actx[diff_idx, 1:]
                    
                    # Update context
                    batch_actx[diff_idx, -1] = make_onset_feature_context(song_feats, int(right_time*100), 4)
                    
                    # Calculate step positions
                    step_positions = {t: np.argwhere(subdiv_beats == t)[0] for t in placed_times}
                    step_beat = step_positions[time]/maxstep
                    next_step = step_positions[next_time]/maxstep
                    
                    # Update step history
                    batch_step_hist[diff_idx, 64] = np.append(np.zeros(256), [step_beat - batch_last_step[diff_idx], next_step-step_beat])
                    
                    # Copy to current batch arrays
                    current_step_hist[batch_pos] = batch_step_hist[diff_idx]
                    current_actx[batch_pos] = batch_actx[diff_idx]
                
                # Run batch prediction
                batch_predictions = sym_model.predict(
                    (current_actx[:, :8], current_actx[:, 7:], current_step_hist), 
                    batch_size=current_batch_size, 
                    verbose=0
                )
                
                # Process predictions and update states
                for batch_pos, (diff_idx, diff) in enumerate(zip(batch_indices, valid_diffs)):
                    placed_times = all_placed_times[diff]
                    time = placed_times[step_idx]
                    
                    try: 
                        next_time = placed_times[step_idx+1]
                    except: 
                        next_time = time
                    
                    step_positions = {t: np.argwhere(subdiv_beats == t)[0] for t in placed_times}
                    step_beat = step_positions[time]/maxstep
                    next_step = step_positions[next_time]/maxstep
                    
                    # Get prediction for this diff
                    new_step = weighted_pick(batch_predictions[batch_pos:batch_pos+1])
                    
                    # Update step history with actual prediction
                    batch_step_hist[diff_idx, 64] = np.append(sparse_to_categorical(new_step, 255), [step_beat - batch_last_step[diff_idx], next_step-step_beat])
                    batch_last_step[diff_idx] = step_beat
                    
                    # Store result
                    all_selected_steps[diff].append(unravel_onehot(new_step, 4))

            # Third phase: Generate output files for each diff
            for diff in diffs:
                fine = fine_diffs[diff]
                placed_times = all_placed_times[diff]
                selected_steps = all_selected_steps[diff]
                
                assert len(placed_times) == len(selected_steps)
                
                time_to_step = {t: step for t, step in zip(placed_times, selected_steps)}
                full_steps = [time_to_step.get(beat, '0000') for beat in subdiv_beats]
                measures = np.split(full_steps, list(range(subdiv, len(full_steps), subdiv)))

                measures_txt = '\n,\n'.join(['\n'.join(measure) for measure in measures])
                chart_txt = chart_template.format(
                    ccoarse=diff,
                    cfine=fine,
                    measures=measures_txt
                )
                diff_chart_txts.append(chart_txt)

            # Generate final output
            out_dir_name = os.path.split(out_song_path)[1]
            audio_out_name = os.path.split(song_in)[1]
            sm_txt = template.format(
                title=title,
                artist=artist,
                music_fp=audio_out_name,
                bpm=bpm_str,
                offset=offset,
                charts='\n'.join(diff_chart_txts))

            print('Saving to {}'.format(out_song_path))
            try:
                if not os.path.isdir(out_song_path):
                    os.mkdir(out_song_path)
                audio_out_fp = os.path.join(out_song_path, audio_out_name)
                if not os.path.exists(audio_out_fp):
                    shutil.copyfile(song_in, audio_out_fp)
                with open(os.path.join(out_song_path, out_dir_name + '.sm'), 'w') as f:
                    f.write(sm_txt)
            except:
                raise Exception('Error during output')