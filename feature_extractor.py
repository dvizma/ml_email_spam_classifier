#######################################################
#                 ML Assignment 2                     #
#              Dvizma Sinha, CS20M504                 #
#                CS5011W, 25/05/2021                  #
#######################################################

from pathlib import Path
from collections import Counter
import email.policy
import numpy as np
import pickle
import string
import re
import email.parser
from html.parser import HTMLParser

stopwords = {"ourselves", "hers", "between", "yourself", "but", "again",
             "there", "about", "once", "during", "out", "very", "having", "with",
             "they", "own", "an", "be", "some", "for", "do", "its", "yours",
             "such", "into", "of", "most", "itself", "other", "off", "is", "s",
             "am", "or", "who", "as", "from", "him", "each", "the", "themselves",
             "until", "below", "are", "we", "these", "your", "his", "through",
             "don", "nor", "me", "were", "her", "more", "himself", "this", "down",
             "should", "our", "their", "while", "above", "both", "up", "to", "ours",
             "had", "she", "all", "no", "when", "at", "any", "before", "them", "same",
             "and", "been", "have", "in", "will", "on", "does", "yourselves", "then",
             "that", "because", "what", "over", "why", "so", "can", "did", "not",
             "now", "under", "he", "you", "herself", "has", "just", "where", "too",
             "only", "myself", "which", "those", "i", "after", "few", "whom", "t",
             "being", "if", "theirs", "my", "against", "a", "by", "doing", "it",
             "how", "further", "was", "here", "than", "re"}


class EmailHTMLParser(HTMLParser):
    """Parse email content of type text/html"""
    text = ""

    def handle_data(self, data):
        self.text += data


class EmailContent:
    """This class represents data contained in the email"""

    def __init__(self, path) -> None:
        self.content = ""
        self.wordcounts = None
        self.empty = False
        self.html_ratio = 0
        self.path = path
        self.content_type = []
        self.read(path)

    def read(self, path):
        """Read email from given file path

        Args:
            path (Path): Path to email file
        """
        with open(path, "rb") as fh:
            parsed_email = email.parser.BytesParser(
                policy=email.policy.default).parse(fh)

        content = parsed_email["Subject"]

        # When subject is not present
        if content is None:
            content = ""

        plaintext_count = 0
        html_count = 0

        for email_part in parsed_email.walk():
            content_type = email_part.get_content_type()

            if content_type == 'text/plain':
                try:
                    text_content = " ".join(
                        str(email_part.get_content()).strip().split())
                    content = " ".join([content, text_content])
                except LookupError:
                    text_content = " ".join(
                        str(email_part.get_payload()).strip().split())
                    content = " ".join([content, text_content])
                plaintext_count += 1
            elif content_type == 'text/html':
                try:
                    html_content = email_part.get_content()
                except LookupError:
                    html_content = str(email_part.get_payload())
                text_content = self.__parse_html(html_content)
                content = " ".join([content, text_content])
                html_count += 1
            else:
                content = "empty"
            self.content = self.content + content

        # Convert content to lower case
        self.content = self.content.lower()

    @staticmethod
    def __parse_html(html_content):
        """Parse html content into plaintext

        Args:
            html_content (str): HTML content of email

        Returns:
            str: parsed plain text
        """
        parser = EmailHTMLParser()
        parser.feed(html_content)

        text_data = parser.text

        # There may be several whitespaces/ new lines in the parsed content
        text_data = " ".join(text_data.strip().split())
        return text_data


class FeatureExtractor:
    """ This class processes emails to extract relevant features"""

    vocab_path = Path(__file__).parent.resolve() / "vocab.pickle"
    punctuations_to_keep = "$!"
    email_regex = re.compile(r'[\w\.-]+@[\w\.-]+')
    url_regex = re.compile(r'(https?://[^\s]+)')
    num_regex = re.compile(r'\d+')

    def __init__(self, vocabulary_size=1000):
        self.emails = []
        self.vocabulary = {}
        self.vocabulary_size = vocabulary_size

    def build_volcabulary(self, paths):
        """Build vocabulary by combining data from emails present in given paths

        Args:
            paths (list): list of email file paths
        """
        self.read_emails(paths)
        self.combine_words()
        self.save_vocabulary()

    def read_emails(self, email_paths):
        """Read emails and process content to have relevant information only

        Args:
            email_paths (list): list of email file paths
        """
        for file in email_paths:
            self.emails.append(EmailContent(file))

        punctuations_to_remove = [p for p in string.punctuation
                                  if p not in self.punctuations_to_keep]

        for email_data in self.emails:
            content = email_data.content

            # Remove non ascii chars
            content = ''.join(filter(lambda x: x in string.printable, content))

            # Replace email addresses by word "emailaddress"
            content = self.email_regex.sub("", content)

            # Replace urls with url
            content = self.url_regex.sub("", content)

            # Remove punctuations
            for p in punctuations_to_remove:
                content = content.replace(p, "")

            # replace ! by " ! " and $ by " $ "
            for p in ["!", "$"]:
                content = content.replace(p, f" {p} ")
            content = " ".join(content.split())

            # replace all numbers by word "numlike"
            content = self.num_regex.sub("", content)
            content = content.replace("numlike", f" numlike ")
            content = " ".join(content.split())

            content = content.split()
            content = [i for i in content if i not in stopwords]
            content = [i for i in content if len(
                i) > 1 or i not in string.ascii_lowercase]

            email_data.wordcounts = Counter(content)

    def combine_words(self):
        """Combine data from all training emails into vocabulary."""

        MAX_WORD_COUNT_PER_MAIL = 10
        vocab_word_count = Counter()

        # From every email
        for email_data in self.emails:

            # accumulate word count
            for word, count in email_data.wordcounts.items():
                vocab_word_count[word] += min(count, MAX_WORD_COUNT_PER_MAIL)

        # Save most common words as vocabulary
        vocab_word_count = vocab_word_count.most_common()[
            :self.vocabulary_size]

        # Add ranks to most common words
        self.vocabulary = {word: index + 1 for index,
                           (word, _) in enumerate(vocab_word_count)}

    def extract_features(self, log=False):
        """Load vocabulary and extract features from emails.The word count not present in 
        vocabulary are accumulated for index 0"""

        self.load_vocabulary()
        feature_data = np.zeros((len(self.emails), self.vocabulary_size + 1))
        for i, email_data in enumerate(self.emails):
            content = email_data.wordcounts
            for word, count in content.items():
                word_index = self.vocabulary.get(word, 0)
                feature_data[i][word_index] += count

        return feature_data

    def save_vocabulary(self):
        """Save vocabulary to pickle file"""
        with open(self.vocab_path, "wb") as vfile:
            pickle.dump(self.vocabulary, vfile)

    def load_vocabulary(self):
        """Load vocabulary to pickle file"""
        with open(self.vocab_path, "rb") as vfile:
            self.vocabulary = pickle.load(vfile)
            self.vocabulary_size = len(self.vocabulary)
