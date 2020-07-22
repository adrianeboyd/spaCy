from typing import Set
from thinc.api import Config

from .stop_words import STOP_WORDS
from ...language import Language
from ...util import registry


DEFAULT_CONFIG = """
[nlp]
lang = "kn"
stop_words = {"@language_data": "spacy.kn.stop_words"}
"""


@registry.language_data("spacy.kn.stop_words")
def stop_words() -> Set[str]:
    return STOP_WORDS


class Kannada(Language):
    lang = "kn"
    default_config = Config().from_str(DEFAULT_CONFIG)


__all__ = ["Kannada"]
