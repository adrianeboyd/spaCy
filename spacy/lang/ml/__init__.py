from typing import Set, Dict, Callable, Any
from thinc.api import Config

from .stop_words import STOP_WORDS
from .lex_attrs import LEX_ATTRS
from ...language import Language
from ...util import registry


DEFAULT_CONFIG = """
[nlp]
lang = "ml"
stop_words = {"@language_data": "spacy.ml.stop_words"}
"""


@registry.language_data("spacy.ml.stop_words")
def stop_words() -> Set[str]:
    return STOP_WORDS


@registry.language_data("spacy.ml.lex_attr_getters")
def lex_attr_getters() -> Dict[int, Callable[[str], Any]]:
    return LEX_ATTRS


class Malayalam(Language):
    lang = "ml"
    default_config = Config().from_str(DEFAULT_CONFIG)


__all__ = ["Malayalam"]
