"""Main library for the Jeeves prototype.

This module contains the general-purpose functions that
are going to be used by train.py and predict.py.
"""

import pandas as pd
import sklearn.ensemble
import sklearn.feature_extraction.text

import _pickle as cPickle

_TRAINING_SET_FILEPATH = ''
_INSTRUCTIONS_COLUMN_NAME = ''
_COMMANDS_COLUMN_NAME = ''

_ENCODER_FILEPATH = 'pickles/encoder.p'

_VECTORIZER_FILEPATH = 'pickles/vectorizer.p'
_VECTORIZER_MIN_NGRAM = 3
_VECTORIZER_MAX_NGRAM = 3
_VECTORIZER_USE_BINARY = True

_MODEL_FILEPATH = 'pickles/model.p'

def read_data():
    """Read the data and return a Pandas.DataFrame()."""
    dataset = pd.read_csv(_TRAINING_SET_FILEPATH)
    return dataset


def get_instructions_and_commands(dataset):
    """Separate the instructions from the commands, returns a tuple."""
    instructions = dataset[_INSTRUCTIONS_COLUMN_NAME]
    commands = dataset[_COMMANDS_COLUMN_NAME]
    return instructions, commands


class LabelEncoder(object):
    """Used to encode the commands as numerical labels."""

    def __init__(self, load_from_disk=False):
        if load_from_disk:
            self._encoder = self._load_encoder()
        else:
            self._encoder = self._generate_encoder()

    def _generate_encoder(self):
        """Generate a brand new encoder."""
        encoder = sklearn.preprocessing.LabelEncoder()
        return encoder

    def _load_encoder(self):
        """Load the encoder from disk."""
        with open('pickles/encoder.p', 'rb') as handle:
            encoder = cPickle.load(handle)
        return encoder

    def train(self, commands):
        """Fit the encoder using commands as input."""
        self._encoder.fit(commands)

    def save(self):
        """Save the encoder on disk."""
        with open('pickles/encoder.p', 'wb') as handle:
            cPickle.dump(self._encoder, handle)

    def encode(self, commands):
        """Transform the commands into numerical labels."""
        labels = self._encoder.transform(commands)
        return labels

    def decode(self, labels):
        """Map numerical labels back to the original commands."""
        commands = self._encoder.inverse_transform(labels)
        return commands


class Vectorizer(object):
    """The vectorizer used to generate signals from instructions."""

    def __init__(self, load_from_disk=False):
        if load_from_disk:
            self._vectorizer = self._load_vectorizer()
        else:
            self._vectorizer = self._generate_vectorizer()

    def _generate_vectorizer(self):
        """Generate a brand new vectorizer."""
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            analyzer='char',
            ngram_range=(_VECTORIZER_MIN_NGRAM, _VECTORIZER_MAX_NGRAM),
            binary=_VECTORIZER_USE_BINARY)
        return vectorizer

    def _load_vectorizer(self):
        """Load the vectorizer from disk."""
        with open('pickles/vectorizer.p', 'rb') as handle:
            vectorizer = cPickle.load(handle)
        return vectorizer

    def train(self, instructions):
        """Fit the vectorizer using instructions as input."""
        self._vectorizer.fit(instructions)

    def save(self):
        """Save the vectorizer on disk."""
        with open('pickles/vectorizer.p', 'wb') as handle:
            cPickle.dump(self._vectorizer, handle)

    def generate_signals(self, instructions):
        """Generate ngram-based signals from instructions."""
        signals = self._vectorizer.transform(instructions)
        return signals


class JeevesModel(object):
    """The actual model use to predict the command to execute."""

    def __init__(self, load_from_disk=False):
        if load_from_disk:
            self._model = self._load_model()
        else:
            self._model = self._generate_model()

    def _generate_model(self):
        """Generate a brand new model."""
        model = sklearn.ensemble.RandomForestClassifier(n_jobs=-1)
        return model

    def _load_model(self):
        """Load the model from disk."""
        with open('pickles/model.p', 'rb') as handle:
            model = cPickle.load(handle)
        return model

    def train(self, signals, labels):
        """Fit the model on signals to predict labels."""
        self._model.fit(signals, labels)

    def save(self):
        """Save the model on disk."""
        with open('pickles/model.p', 'wb') as handle:
            cPickle.dump(self._model, handle)

    def predict_command(self, signals):
        """Predict the command label given a set of signals."""
        predicted_label = self._model.predict(signals)
        return predicted_label
