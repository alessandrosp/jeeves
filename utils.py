"""Main library for the Jeeves prototype.

This module contains the general-purpose functions that
are going to be used by train.py and predict.py.
"""

import collections
import string
import random

import pandas as pd
import sklearn.ensemble
import sklearn.feature_extraction.text
import sklearn.model_selection

import _pickle as cPickle

_TRAINING_SET_FILEPATH = 'data/raw_training_dataset.csv'
_INSTRUCTIONS_COLUMN_NAME = 'instructions'
_COMMANDS_COLUMN_NAME = 'commands'

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


class DataExtender(object):
    """Generate an extended dataset with additional instructions."""

    def __init__(
            self,
            n_iterations=100,
            prob_stay_the_same=0.5,
            prob_swap_one_word=0.5,
            prob_insert_a_typo=0.1):
        self._n_iterations = n_iterations
        self._prob_stay_the_same = prob_stay_the_same
        self._prob_swap_one_word = prob_swap_one_word
        self._prob_insert_a_typo = prob_insert_a_typo

    def _learn_vocabulary(self, dataset):
        """Learn the words associated with each command."""
        vocabulary = collections.defaultdict(list)
        for _, row in dataset.iterrows():
            command = row['commands']
            for token in row['instructions'].split(' '):
                if token:
                    vocabulary[command].append(token)
        self.vocabulary_ = vocabulary

    def _swap_one_word(self, row):
        """Create a new instruction swapping words.

        Specifically, it create a new instruction (str) by
        replacing one word in the instruction contained
        in row with another word from self.vocabulary_. Only
        words associated with the same command are considered.

        Args:
            row: pd.Series(), where the indexes are the same
                as the columns names of the original dataset.

        Returns:
            A string with the new instruction.
        """
        old_instruction = row['instructions']
        tokens = [token
                  for token in old_instruction.split(' ')
                  if token]
        new_word = random.choice(self.vocabulary_[row['commands']])
        random_index = random.choice(range(0, len(tokens)))
        tokens[random_index] = new_word

        return ' '.join(tokens)

    def _insert_a_typo(self, row):
        """Create a new instruction inserting a typo in given one.

        Args:
            row: pd.Series(), where the indexes are the same
                as the columns names of the original dataset.

        Returns:
            A string with the new instruction.
        """
        old_instruction = row['instructions']
        letters = list(old_instruction)
        random_letter = random.choice(string.ascii_lowercase)
        random_index = random.choice(range(0, len(letters)))
        letters[random_index] = random_letter

        return ''.join(letters)

    def generate_new_dataset(self, dataset, verbose=True):
        """Generate a new, larger dataset."""
        columns = dataset.columns
        if verbose:
            print('Rows in raw dataset: {rows}'.format(
                rows=dataset.shape[0]))
        self._learn_vocabulary(dataset)

        new_rows = []
        for _ in range(self._n_iterations):
            for _, row in dataset.iterrows():
                # Append the original row as it is
                if self._prob_stay_the_same >= random.uniform(0, 1):
                    new_rows.append(dict(row))
                # Append a row with a single word being swapped
                if self._prob_swap_one_word >= random.uniform(0, 1):
                    new_row = {
                        'instructions': self._swap_one_word(row),
                        'commands': row['commands']}
                    new_rows.append(new_row)
                # Append a row with a single typo in it
                if self._prob_insert_a_typo >= random.uniform(0, 1):
                    new_row = {
                        'instructions': self._insert_a_typo(row),
                        'commands': row['commands']}
                    new_rows.append(new_row)

        if verbose:
            print('Rows in extended dataset: {rows}'.format(
                rows=len(new_rows)))
        return pd.DataFrame(new_rows, columns=columns)



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
        clf = sklearn.ensemble.RandomForestClassifier()
        param_grid = {'max_features': ['auto', None]}
        model = sklearn.model_selection.GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            scoring='accuracy',
            n_jobs=-1)
        return model

    def _load_model(self):
        """Load the model from disk."""
        with open('pickles/model.p', 'rb') as handle:
            model = cPickle.load(handle)
        return model

    def train(self, signals, labels, verbose=True):
        """Fit the model on signals to predict labels."""
        self._model.fit(signals, labels)
        if verbose:
            best_score = self._model.best_score_
            print('Best score in CV: {score}\n'.format(score=best_score))

    def save(self):
        """Save the model on disk."""
        with open('pickles/model.p', 'wb') as handle:
            cPickle.dump(self._model, handle)

    def predict_command(self, signals):
        """Predict the command label given a set of signals."""
        predicted_label = self._model.predict(signals)
        return predicted_label
