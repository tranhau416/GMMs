#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import librosa
import numpy as np
import os
import sklearn.mixture
import sys


def load(audio_path):
    print("[PATH]", audio_path)
    y, sr = librosa.load(audio_path)
    y_trim = librosa.effects.remix(y, intervals=librosa.effects.split(y))
    mfcc = librosa.feature.mfcc(y=y_trim, sr=sr)
    return mfcc.T


def fit(frames, n_components=16):
    index = np.arange(len(frames))
    np.random.shuffle(index)
    # train_idx = index[int(len(index) * test_ratio):]
    # test_idx = index[:int(len(index) * test_ratio)]
    # print("[train_idx]: ", train_idx.shape)
    # print("[test_idx]: ", test_idx.shape)

    gmm = sklearn.mixture.GaussianMixture(n_components=n_components)
    # print(frames[index])
    gmm.fit(frames[index])
    print(gmm.means_)
    print('\n')
    print(gmm.covariances_)

    return gmm


def predict(gmms, test_frame):
    scores = []
    for gmm_name, gmm in gmms.items():
        scores.append((gmm_name, gmm.score(test_frame)))
    return sorted(scores, key=lambda x: x[1], reverse=True)


def evaluate(gmms, test_frames):
    correct = 0

    for name in test_frames:
        best_name, best_score = predict(gmms, test_frames[name])[0]
        print('Ground Truth: %s, Predicted: %s, Score: %f' % (name, best_name, best_score))
        if name == best_name:
            correct += 1

    print('Overall Accuracy: %f%%' % (float(correct) / len(test_frames)))


if __name__ == '__main__':
    gmms, test_frames = {}, {}
    # print(sys.argv[1])
    for filename in glob.glob(os.path.join("E:\Chờ người nơi ấy\Other", '*.wav')):
        name = os.path.splitext(os.path.basename(filename))[0]
        print('Processing %s ...' % name)
        gmms[name] = fit(load(filename))

    # evaluate(gmms, test_frames)

    for filename in glob.glob(os.path.join("E:\Chờ người nơi ấy\Standard", '*.wav')):
        result = predict(gmms, load(filename))
        # print(result)
        print('%s: %s' % (os.path.basename(filename), ' / '.join(map(lambda x: '%s = %f' % x, result[:5]))))
