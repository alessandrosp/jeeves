"""Script used to predict a command given an instruction."""

import argparse
import subprocess
import warnings

import pandas as pd

import utils

def get_instruction():
    """Get the instruction from the flags."""
    parser = argparse.ArgumentParser()
    # Note: nargs='+' groups every word specified as flags
    # into args.instruction; we can then join the words
    # into a single string.
    parser.add_argument("instruction", nargs='+')
    args = parser.parse_args()
    return ' '.join(args.instruction)

def main():
    """Main function for the module."""
    instructions = get_instruction()
    instructions = pd.Series(instructions)

    # We need to catch warnings to avoid numpy to
    # complain of sklearn implementation
    # See: http://bit.ly/2o9xxee
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    encoder = utils.LabelEncoder(load_from_disk=True)

    vectorizer = utils.Vectorizer(load_from_disk=True)
    signals = vectorizer.generate_signals(instructions)

    jeeves = utils.JeevesModel(load_from_disk=True)
    predicted_label = jeeves.predict_command(signals)

    command = encoder.decode(predicted_label)[0]
    response = input('>>> Do you want me to execute: {cmd}? '.format(
        cmd=command))
    if response in ('', 'yes', 'y'):
        subprocess.Popen(command.split())

if __name__ == '__main__':
    main()
