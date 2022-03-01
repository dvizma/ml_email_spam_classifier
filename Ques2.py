#######################################################
#                 ML Assignment 2                     #
#              Dvizma Sinha, CS20M504                 #
#                CS5011W, 25/05/2021                  #
#######################################################
from pathlib import Path
from argparse import ArgumentParser
from email_classifier import EmailClassifier

data_dir = Path(__file__).resolve().parent / "Data" / "Ques2"


def parse_args():
    parser = ArgumentParser(description="ML Assignment 2: Email Classifier")
    parser.add_argument("--train", help="Train the classifier", action="store_true")
    parser.add_argument("--test", help="Test folder",default=data_dir / "test")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    classifier = EmailClassifier()

    if args.train:
        spam_dir = data_dir / "spam"
        non_spam_dir = data_dir / "non_spam"
        classifier.train(non_spam_dir, spam_dir)

    classifier.test(args.test)
