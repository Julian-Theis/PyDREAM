import numpy as np
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

np.seterr(divide='ignore', invalid='ignore')


class NAP:
    def __init__(self,
                 tss_train_file: str = None,
                 tss_test_file: str = None,
                 options: dict = None):
        """ Options """

        self.opts = {"seed": 1,
                     "n_epochs": 100,
                     "n_batch_size": 64,
                     "dropout_rate": 0.2,
                     "eval_size": 0.1,
                     "activation_function": "relu"}
        self.__setSeed()

        if options is not None:
            for key in options.keys():
                self.opts[key] = options[key]

        """ Load data and setup """
        if tss_train_file is not None and tss_test_file is not None:
            self.X_train, self.Y_train = self.__loadData(tss_train_file)
            self.X_test, self.Y_test = self.__loadData(tss_test_file)

            self.__oneHotEncoderSetup()
            self.Y_train = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_train).reshape(-1, 1)))
            self.Y_test = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_test).reshape(-1, 1)))

            self.stdScaler = MinMaxScaler()
            self.stdScaler.fit(self.X_train)
            self.X_train = self.stdScaler.transform(self.X_train)
            self.X_test = self.stdScaler.transform(self.X_test)

            self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train,
                                                                                  self.Y_train,
                                                                                  test_size=self.opts["eval_size"],
                                                                                  random_state=self.opts["seed"],
                                                                                  shuffle=True)

            insize = self.X_train.shape[1]
            outsize = len(self.Y_train[0])

            """ Create Model """
            self.model = Sequential()
            self.model.add(Dense(insize, input_dim=insize, activation=self.opts["activation_function"]))
            self.model.add(Dropout(self.opts["dropout_rate"]))
            self.model.add(Dense(int(insize * 1.2), activation=self.opts["activation_function"]))
            self.model.add(Dropout(self.opts["dropout_rate"]))
            self.model.add(Dense(int(insize * 0.6), activation=self.opts["activation_function"]))
            self.model.add(Dropout(self.opts["dropout_rate"]))
            self.model.add(Dense(int(insize * 0.3), activation=self.opts["activation_function"]))
            self.model.add(Dropout(self.opts["dropout_rate"]))
            self.model.add(Dense(outsize, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self,
              checkpoint_path: str,
              name: str,
              save_results: bool = False):
        event_dict_file = str(checkpoint_path) + "/" + str(name) + "_nap_onehotdict.json"
        with open(str(event_dict_file), 'w') as outfile:
            json.dump(self.one_hot_dict, outfile)

        with open(checkpoint_path + "/" + name + "_nap_model.json", 'w') as f:
            f.write(self.model.to_json())

        ckpt_file = str(checkpoint_path) + "/" + str(name) + "_nap_weights.hdf5"
        checkpoint = ModelCheckpoint(ckpt_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        hist = self.model.fit((self.X_train),
                              (self.Y_train),
                              batch_size=self.opts["n_batch_size"],
                              epochs=self.opts["n_epochs"],
                              shuffle=True,
                              validation_data=([self.X_val], [self.Y_val]),
                              callbacks=[self.EvaluationCallback(self.X_test, self.Y_test),
                                         checkpoint])

        joblib.dump(self.stdScaler, str(checkpoint_path) + "/" + str(name) + "_nap_stdScaler.pkl")
        if save_results:
            results_file = str(checkpoint_path) + "/" + str(name) + "_nap_results.json"
            with open(str(results_file), 'w') as outfile:
                json.dump(str(hist.history), outfile)

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
        x, y = [], []
        with open(file) as json_file:
            tss = json.load(json_file)
            for sample in tss:
                if sample["label"] is not None:
                    x.append(list(itertools.chain(sample["tss"][0],
                                                  sample["tss"][1],
                                                  sample["tss"][2])))
                    y.append(sample["label"])
        return np.array(x), np.array(y)

    def __setSeed(self):
        seed(self.opts["seed"])
        tf.random.set_seed(self.opts["seed"])

    def loadModel(self,
                  path: str,
                  name: str):
        with open(path + "/" + name + "_nap_model.json", 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(path + "/" + name + "_nap_weights.hdf5")
        with open(path + "/" + name + "_nap_onehotdict.json", 'r') as f:
            self.one_hot_dict = json.load(f)
        self.stdScaler = joblib.load(path + "/" + name + "_nap_stdScaler.pkl")

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

        preds = []
        next_events = []
        for sample in tss:
            features = [list(itertools.chain(sample.export()["tss"][0],
                                             sample.export()["tss"][1],
                                             sample.export()["tss"][2]))]
            features = self.stdScaler.transform(features)
            pred = np.argmax(self.model.predict(features), axis=1)
            preds.append(pred[0])
            for p in pred:
                next_events.append(self.__intToEvent(p))
        return preds, next_events

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
            y_pred = self.model.predict(self.X_test)
            y_pred = y_pred.argmax(axis=1)

            test_acc = accuracy_score(self.Y_test_int, y_pred, normalize=True)
            test_loss, _ = self.model.evaluate(self.X_test, self.Y_test)

            precision, recall, fscore, _ = precision_recall_fscore_support(self.Y_test_int, y_pred, average='weighted',
                                                                           pos_label=None)
            auc = multiclass_roc_auc_score(self.Y_test_int, y_pred, average="weighted")

            logs['test_acc'] = test_acc
            logs['test_prec_weighted'] = precision
            logs['test_rec_weighted'] = recall
            logs['test_loss'] = test_loss
            logs['test_fscore_weighted'] = fscore
            logs['test_auc_weighted'] = auc

            precision, recall, fscore, support = precision_recall_fscore_support(self.Y_test_int, y_pred,
                                                                                 average='macro', pos_label=None)
            auc = multiclass_roc_auc_score(self.Y_test_int, y_pred, average="macro")
            logs['test_prec_mean'] = precision
            logs['test_rec_mean'] = recall
            logs['test_fscore_mean'] = fscore
            logs['test_auc_mean'] = auc
