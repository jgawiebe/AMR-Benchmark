# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses

import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib
# 'Agg', not 'Tkagg': this runs headless and Homebrew python3.11 has no tkinter.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
#from matplotlib import pyplot as plt
import pickle, random, sys
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler,TensorBoard
# NOTE: `from keras.optimizers import adam` was removed -- lowercase `adam` does
# not exist in Keras 2.15 (hard ImportError) and was never used; the model is
# compiled with optimizer='adam' below.
import pickle, random, sys,h5py
# NOTE: `rmldataset2016` was removed -- no such module exists under HisarMod/, and
# its only reference was inside a commented-out block in predict().
import mltools
import rmlmodels.CNN2Model as cnn2
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import h5py
classes = ['BPSK',
               'QPSK',
               '8PSK',
               '16PSK',
               '32PSK',
               '64PSK',
               '4QAM',
               '8QAM',
               '16QAM',
               '32QAM',
               '64QAM',
               '128QAM',
               '256QAM',
               '2FSK',
               '4FSK',
               '8FSK',
               '16FSK',
               '4PAM',
               '8PAM',
               '16PAM',
               'AM-DSB',
               'AM-DSB-SC',
               'AM-USB',
               'AM-LSB',
                'FM',
                'PM']
# ---------------------------------------------------------------------------
# Data loading.
#
# Upstream read train.mat/test.mat -- MATLAB files the authors produced from the
# CSV distribution but never shipped, alongside train_labels1.csv/test_labels1.csv
# (remapped labels, also not shipped). The public HisarMod2019.1 release is CSV
# only, so _port_tools/hisarmod_csv_to_npy.py rebuilds the equivalent arrays as
# .npy: float32 (N, 2, 1024) signals, plus labels already remapped to 0..25 from
# the sparse 2-digit family codes (0,1,2,3,4,10,...,61) the CSV actually ships.
#
# The .npy files are opened with mmap_mode='r' -- the train signals alone are
# 4.3 GB and materialising train + the fancy-indexed train/val copies at once
# would not fit comfortably in 24 GB.
# ---------------------------------------------------------------------------
DATA = os.path.join(os.path.dirname(__file__), '..', '..',
                    'Datasets', 'HisarMod2019.1', 'npy')
DATA = os.path.abspath(DATA)
if not os.path.isdir(DATA):
    sys.exit(f"missing {DATA}\nrun: .venv/bin/python _port_tools/hisarmod_csv_to_npy.py")

train = np.load(os.path.join(DATA, 'train_data.npy'), mmap_mode='r')
test  = np.load(os.path.join(DATA, 'test_data.npy'),  mmap_mode='r')

##label
train_labels = to_categorical(np.load(os.path.join(DATA, 'train_labels.npy')),
                              num_classes=len(classes))
test_labels  = to_categorical(np.load(os.path.join(DATA, 'test_labels.npy')),
                              num_classes=len(classes))

##snr
train_snr = np.load(os.path.join(DATA, 'train_snr.npy')).reshape(-1, 1)
test_snr  = np.load(os.path.join(DATA, 'test_snr.npy')).reshape(-1, 1)

# [N,2,1024] -> the models want a trailing channel axis, [N,2,1024,1]
np.random.seed(2016)   # reproducible split

# Restrict to high-SNR examples. HisarMod ships 20 SNR levels from -20 to +18 dB
# in 2 dB steps; MIN_SNR=0 keeps the upper 10, i.e. half of train and test.
# Set AMR_MIN_SNR=-100 to train on the full SNR range as upstream does.
MIN_SNR = int(os.environ.get('AMR_MIN_SNR', 0))
train_pool = np.flatnonzero(train_snr.ravel() >= MIN_SNR)
test_pool = np.flatnonzero(test_snr.ravel() >= MIN_SNR)
if len(train_pool) == 0 or len(test_pool) == 0:
    sys.exit(f"no examples at SNR >= {MIN_SNR} dB")

n_examples = len(train_pool)
n_train = int(n_examples * 0.8)
n_val = n_examples - n_train
# Indices are kept SORTED for the gather below: these are fancy-index reads out of
# a 4.3 GB memmap, and random order turns a sequential scan into 416k seeks.
# fit() reshuffles every epoch anyway, so ordering here costs nothing.
train_idx = np.sort(train_pool[np.random.choice(len(train_pool), size=n_train, replace=False)])
val_idx = np.sort(np.setdiff1d(train_pool, train_idx))
X_train = train[train_idx][..., np.newaxis]
Y_train = train_labels[train_idx]
X_val = train[val_idx][..., np.newaxis]
Y_val = train_labels[val_idx]
X_test = test[test_pool][..., np.newaxis]
Y_test = test_labels[test_pool]
Z_test = test_snr[test_pool]
print(f"SNR >= {MIN_SNR} dB: train {len(train_idx):,} / val {len(val_idx):,} / test {len(test_pool):,}")

# Upstream set nb_epoch = 10000. On this machine an epoch is ~6 min on CPU, so
# that ceiling is ~41 days; EarlyStopping is what actually ends the run. Capped
# here to something reachable, overridable without editing the file:
#   AMR_EPOCHS=5 .venv/bin/python main.py
nb_epoch = int(os.environ.get('AMR_EPOCHS', 100))
batch_size = int(os.environ.get('AMR_BATCH', 400))  # training batch size

# mltools writes into figure/, and ModelCheckpoint cannot create weights/ itself.
os.makedirs('weights', exist_ok=True)
os.makedirs('figure', exist_ok=True)
# perform training ...
#   - call the main training loop in keras for our network+dataset
model = cnn2.CNN2Model()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.summary()

filepath = 'weights/CNN2_0.5.wts.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_val,Y_val),
    callbacks = [
                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                # upstream wrote `patince=5` -- a typo Keras silently swallows, so it
                # has really been running with the default patience=10. Corrected to
                # the intended 5; set it back to 10 to match upstream's actual runs.
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,patience=5,min_lr=0.000001),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
                #keras.callbacks.TensorBoard(log_dir='./logs/',histogram_freq=1,write_graph=False,write_grads=1,write_images=False,update_freq='epoch')
                ]
                    )
mltools.show_history(history)

#Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print(score)

def predict(model):
    # (mods,snrs,lbl),(X_train,Y_train),(X_test,Y_test),(train_idx,test_idx) = \
    #     rmldataset2016.load_data()
    model.load_weights(filepath)
    # Plot confusion matrix
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print('Overall Accuracy:%.2f%s / (%d + %d)' % (100 * acc, '%', right, wrong))
    mltools.plot_confusion_matrix(cm, labels = ['BPSK',
               'QPSK',
               '8PSK',
               '16PSK',
               '32PSK',
               '64PSK',
               '4QAM',
               '8QAM',
               '16QAM',
               '32QAM',
               '64QAM',
               '128QAM',
               '256QAM',
               '2FSK',
               '4FSK',
               '8FSK',
               '16FSK',
               '4PAM',
               '8PAM',
               '16PAM',
               'AM-DSB',
               'AM-DSB-SC',
               'AM-USB',
               'AM-LSB',
           'FM',
           'PM'], save_filename='figure/lstm3_total_confusion.png')
    mltools.calculate_acc_cm_each_snr(Y_test, test_Y_hat, Z_test, classes, min_snr=0)
predict(model)
