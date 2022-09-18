import numpy as np
from tqdm import tqdm
import joblib
import json
import itertools
import tensorflow as tf

from numpy.random import seed

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential, model_from_json

from pydream.util.Functions import multiclass_roc_auc_score
from pydream.util.TimedStateSamples import TimedStateSample

from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Sequential, model_from_json
from pydream.util.TimedStateSamples import TimedStateSample
import itertools
from itertools import chain
from sklearn.utils import class_weight
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import swish
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, recall_score
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dropout, Dense, Input, Lambda, Conv2D, Flatten, Reshape, Permute
from tensorflow.keras.models import model_from_json, Model
import tensorflow as tf
from tensorflow.keras.layers import concatenate, Layer
import pandas as pd
import ast

np.seterr(divide='ignore', invalid='ignore')


class ReachabilityMask(Layer):
    def __init__(self, **kwargs):
        super(ReachabilityMask, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[-1]

    def call(self, inputs):
        output = tf.multiply(inputs[0], inputs[1])
        max = 1/tf.reduce_sum(output, 1)
        ret = tf.multiply(output, max[:, tf.newaxis])
        return inputs[0]

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(ReachabilityMask, self).get_config()
        return dict(list(base_config.items()))

class NAPdesparse:
    def __init__(self,
                 tss_train_file: str = None,
                 tss_val_file: str = None,
                 tss_test_file: str = None,
                 options: dict = None):
        """ Options """

        self.opts = {"seed": 1,
                     "n_epochs": 100,
                     "n_batch_size": 64,
                     "dropout_rate": 0.2,
                     "activation_function": "swish"}
        self.__setSeed()

        if options is not None:
            for key in options.keys():
                self.opts[key] = options[key]

        """ Load data and setup """
        if tss_train_file is not None and tss_val_file is not None and tss_test_file is not None:
            self.X_train, self.M_train, self.Y_train = self.__loadData(tss_train_file)
            self.X_val, self.M_val, self.Y_val = self.__loadData(tss_val_file)
            self.X_test, self.M_test, self.Y_test = self.__loadData(tss_test_file)

            self.__oneHotEncoderSetup()
            self.Y_train = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_train).reshape(-1, 1)))
            self.Y_val = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_val).reshape(-1, 1)))
            self.Y_test = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_test).reshape(-1, 1)))


            self.M_train = self.masksToArray(self.M_train)
            self.M_val = self.masksToArray(self.M_val)
            self.M_test = self.masksToArray(self.M_test)

            self.stdScaler = MinMaxScaler()
            self.stdScaler.fit(self.X_train)
            self.X_train = self.stdScaler.transform(self.X_train)
            self.X_test = self.stdScaler.transform(self.X_test)

            tss_size = self.X_train.shape[1]
            mask_size = self.M_train.shape[1]
            outsize = self.Y_train.shape[1]

            learning_rate = 1e-3  # was 1e-3
            optm = Adam(lr=learning_rate)

            # define two sets of inputs
            input_tss = Input(shape=(tss_size,))
            input_mask = Input(shape=(mask_size,))

            x = Dense(tss_size, activation=self.opts["activation_function"])(input_tss)
            x = Dropout((self.opts["dropout_rate"]))(x)
            x1 = Dense(mask_size, activation=self.opts["activation_function"])(input_mask)
            x1 = Dropout((self.opts["dropout_rate"]))(x1)
            x = concatenate([x, x1])
            x = Dense(int(tss_size * 1.2), activation=self.opts["activation_function"])(x)
            x = Dropout((self.opts["dropout_rate"]))(x)
            x = Dense(int(tss_size * 0.6), activation=self.opts["activation_function"])(x)
            x = Dropout((self.opts["dropout_rate"]))(x)
            x = Dense(int(tss_size * 0.3), activation=self.opts["activation_function"])(x)
            x = Dropout((self.opts["dropout_rate"]))(x)
            probs = Dense(outsize, activation='softmax')(x)
            masked_out = ReachabilityMask()([probs, input_mask])
            self.model = Model(inputs=[input_tss, input_mask], outputs=masked_out)
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            #self.model.summary()


    def masksToArray(self, masks):
        ms = []
        for mask in masks:
            if len(mask) < 1:
                np_mask = np.ones(len(self.label_encoder.classes_))
            else:
                np_mask = np.sum(
                    np.asarray(self.onehot_encoder.transform(self.label_encoder.transform(mask).reshape(-1, 1))), axis=0)
                np_mask[np_mask == 0] = 1e-10
            ms.append(np_mask)
        return np.array(ms)

    def train(self,
              checkpoint_path: str,
              name: str,
              save_results: bool = False):
        event_dict_file = str(checkpoint_path) + "/" + str(name) + "_napdesparse_onehotdict.json"
        with open(str(event_dict_file), 'w') as outfile:
            json.dump(self.one_hot_dict, outfile)

        with open(checkpoint_path + "/" + name + "_napdesparse_model.json", 'w') as f:
            f.write(self.model.to_json())

        ckpt_file = str(checkpoint_path) + "/" + str(name) + "_napdesparse_weights.hdf5"
        checkpoint = ModelCheckpoint(ckpt_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        hist = self.model.fit((self.X_train, self.M_train),
                              (self.Y_train),
                              batch_size=self.opts["n_batch_size"],
                              epochs=self.opts["n_epochs"],
                              shuffle=True,
                              validation_data=([self.X_val, self.M_val], [self.Y_val]),
                              callbacks=[self.EvaluationCallback([self.X_test, self.M_test], self.Y_test),
                                         checkpoint],
                              verbose=2)

        joblib.dump(self.stdScaler, str(checkpoint_path) + "/" + str(name) + "_napdesparse_stdScaler.pkl")
        if save_results:
            results_file = str(checkpoint_path) + "/" + str(name) + "_napdesparse_results.json"
            with open(str(results_file), 'w') as outfile:
                json.dump(hist.history, outfile)

        self.model.load_weights(ckpt_file)

    def __oneHotEncoderSetup(self):
        """ Events to One Hot"""
        events = np.unique(self.Y_train)

        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(events)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.onehot_encoder.fit(integer_encoded)

        self.one_hot_dict = {}
        for event in events:
            self.one_hot_dict[event] = list(self.onehot_encoder.transform([self.label_encoder.transform([event])])[0])

    @staticmethod
    def __loadData(file: str):
        x, m, y = [], [], []
        with open(file) as json_file:
            tss = json.load(json_file)
            for sample in tss:
                if sample["nextEvent"] is not None:
                    x.append(list(itertools.chain(sample["TimedStateSample"][0],
                                                  sample["TimedStateSample"][1],
                                                  sample["TimedStateSample"][2])))
                    m.append(sample["mask"])
                    y.append(sample["nextEvent"])
        return np.array(x), m, np.array(y)

    def __setSeed(self):
        seed(self.opts["seed"])
        #tf.random.set_seed(self.opts["seed"])

    def loadModel(self,
                  path: str,
                  name: str):
        with open(path + "/" + name + "_napdesparse_model.json", 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(path + "/" + name + "_napdesparse_weights.hdf5")
        with open(path + "/" + name + "_napdesparse_onehotdict.json", 'r') as f:
            self.one_hot_dict = json.load(f)
        self.stdScaler = joblib.load(path + "/" + name + "_napdesparse_stdScaler.pkl")

    def __intToEvent(self,
                   value: int):
        one_hot = list(np.eye(len(self.one_hot_dict.keys()))[value])
        for k, v in self.one_hot_dict.items():
            if str(v) == str(one_hot):
                return k

    def predict(self, tss: list):
        """
        Predict from a list TimedStateSamples

        :param tss: list<TimedStateSamples>
        :return: tuple (DREAM-NAP output, translated next event)
        """
        if not isinstance(tss, list) or not isinstance(tss[0], TimedStateSample):
            raise ValueError("Input is not a list with TimedStateSample")

        next_events = []
        features = []
        masks = []

        for sample in tqdm(tss, desc="Predicting"):
            features.append(list(itertools.chain(sample.export()["TimedStateSample"][0],
                                             sample.export()["TimedStateSample"][1],
                                             sample.export()["TimedStateSample"][2])))
            masks.append(self.masksToArray([sample.export()["mask"]])[0])

        features = self.stdScaler.transform(features)
        features = np.array(features)
        masks = np.array(masks)
        scores = self.model.predict([features, masks])
        preds = np.argmax(scores, axis=1)

        for p in preds:
            next_events.append(self.__intToEvent(p))

        return preds, next_events, np.array(scores)

    """ Callback """
    class EvaluationCallback(Callback):
        def __init__(self,
                     X_test: np.ndarray,
                     Y_test: np.ndarray):
            super().__init__()
            self.X_test = X_test
            self.Y_test = Y_test
            self.Y_test_int = np.argmax(self.Y_test, axis=1)

            self.test_accs = []
            self.losses = []

        def on_train_begin(self, logs={}):
            self.test_accs = []
            self.losses = []

        def on_epoch_end(self, epoch, logs={}):
            y_score = self.model.predict(self.X_test)
            y_pred = y_score.argmax(axis=1)

            test_acc = accuracy_score(self.Y_test_int, y_pred, normalize=True)
            test_loss, _ = self.model.evaluate(self.X_test, self.Y_test)

            precision, recall, fscore, _ = precision_recall_fscore_support(self.Y_test_int, y_pred, average='weighted',
                                                                           pos_label=None)
            logs['test_acc'] = test_acc
            logs['test_prec_weighted'] = precision
            logs['test_rec_weighted'] = recall
            logs['test_loss'] = test_loss
            logs['test_fscore_weighted'] = fscore

            precision, recall, fscore, support = precision_recall_fscore_support(self.Y_test_int, y_pred,
                                                                                 average='macro', pos_label=None)
            logs['test_prec_mean'] = precision
            logs['test_rec_mean'] = recall
            logs['test_fscore_mean'] = fscore
