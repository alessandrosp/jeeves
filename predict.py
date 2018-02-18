"""Script used to predict a command given an instruction."""

import pandas as pd

import utils

def main():
    """Main function for the module."""
    instructions = 'new file'  # Read from flag
    instructions = pd.Series(instructions)

    encoder = utils.LabelEncoder(load_from_disk=True)

    vectorizer = utils.Vectorizer(load_from_disk=True)
    signals = vectorizer.generate_signals(instructions)

    jeeves = utils.JeevesModel(load_from_disk=True)
    predicted_label = jeeves.predict_command(signals)

    command = encoder.decode(predicted_label)
    print(command)
    # run command in bash using the decoded string

if __name__ == '__main__':
    main()
