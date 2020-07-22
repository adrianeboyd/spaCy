from typing import Set, Dict, Callable, Any
from thinc.api import Config

from .punctuation import TOKENIZER_PREFIXES, TOKENIZER_INFIXES
from .punctuation import TOKENIZER_SUFFIXES
from .stop_words import STOP_WORDS
from .lex_attrs import LEX_ATTRS
from .lemmatizer import PolishLemmatizer
from ..tokenizer_exceptions import BASE_EXCEPTIONS
from ...language import Language
from ...util import registry


DEFAULT_CONFIG = """
[nlp]
lang = "pl"
stop_words = {"@language_data": "spacy.pl.stop_words"}
lex_attr_getters = {"@language_data": "spacy.pl.lex_attr_getters"}

[nlp.lemmatizer]
@lemmatizers = "spacy.PolishLemmatizer.v1"

[nlp.lemmatizer.data_paths]
@language_data = "spacy-lookups-data"
lang = ${nlp:lang}
"""


@registry.language_data("spacy.pl.stop_words")
def stop_words() -> Set[str]:
    return STOP_WORDS


@registry.language_data("spacy.pl.lex_attr_getters")
def lex_attr_getters() -> Dict[int, Callable[[str], Any]]:
    return LEX_ATTRS


@registry.lemmatizers("spacy.PolishLemmatizer.v1")
def create_polish_lemmatizer(data_paths: dict = {}) -> PolishLemmatizer:
    return PolishLemmatizer(data_paths=data_paths)


class PolishDefaults(Language.Defaults):
    mod_base_exceptions = {
        exc: val for exc, val in BASE_EXCEPTIONS.items() if not exc.endswith(".")
    }
    tokenizer_exceptions = mod_base_exceptions
    prefixes = TOKENIZER_PREFIXES
    infixes = TOKENIZER_INFIXES
    suffixes = TOKENIZER_SUFFIXES


class Polish(Language):
    lang = "pl"
    Defaults = PolishDefaults
    default_config = Config().from_str(DEFAULT_CONFIG)


__all__ = ["Polish"]
