"""Script used to fit Jeeves using the training dataset."""

import utils

def main():
    """Main function for the module."""
    dataset = utils.read_data()

    expander = utils.DataExtender()
    dataset = expander.generate_new_dataset(dataset)

    instructions, commands = utils.get_instructions_and_commands(dataset)

    encoder = utils.LabelEncoder(load_from_disk=False)
    encoder.train(commands)
    encoder.save()
    labels = encoder.encode(commands)

    vectorizer = utils.Vectorizer(load_from_disk=False)
    vectorizer.train(instructions)
    vectorizer.save()
    signals = vectorizer.generate_signals(instructions)

    jeeves = utils.JeevesModel(load_from_disk=False)
    jeeves.train(signals, labels)
    jeeves.save()


if __name__ == '__main__':
    main()
