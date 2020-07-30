from typing import Optional, List, Dict

from ...lemmatizer import OldLemmatizer
from ...parts_of_speech import NAMES


class PolishLemmatizer(OldLemmatizer):
    # This lemmatizer implements lookup lemmatization based on the Morfeusz
    # dictionary (morfeusz.sgjp.pl/en) by Institute of Computer Science PAS.
    # It utilizes some prefix based improvements for verb and adjectives
    # lemmatization, as well as case-sensitive lemmatization for nouns.
    def __call__(
        self, string: str, univ_pos: str, morphology: Optional[dict] = None
    ) -> List[str]:
        if isinstance(univ_pos, int):
            univ_pos = NAMES.get(univ_pos, "X")
        univ_pos = univ_pos.upper()
        lookup_pos = univ_pos.lower()
        if univ_pos == "PROPN":
            lookup_pos = "noun"
        lookup_table = self.lookups.get_table("lemma_lookup_" + lookup_pos, {})
        if univ_pos == "NOUN":
            return self.lemmatize_noun(string, morphology, lookup_table)
        if univ_pos != "PROPN":
            string = string.lower()
        if univ_pos == "ADJ":
            return self.lemmatize_adj(string, morphology, lookup_table)
        elif univ_pos == "VERB":
            return self.lemmatize_verb(string, morphology, lookup_table)
        return [lookup_table.get(string, string.lower())]

    def lemmatize_adj(
        self, string: str, morphology: dict, lookup_table: Dict[str, str]
    ) -> List[str]:
        # this method utilizes different procedures for adjectives
        # with 'nie' and 'naj' prefixes
        if string[:3] == "nie":
            search_string = string[3:]
            if search_string[:3] == "naj":
                naj_search_string = search_string[3:]
                if naj_search_string in lookup_table:
                    return [lookup_table[naj_search_string]]
            if search_string in lookup_table:
                return [lookup_table[search_string]]
        if string[:3] == "naj":
            naj_search_string = string[3:]
            if naj_search_string in lookup_table:
                return [lookup_table[naj_search_string]]
        return [lookup_table.get(string, string)]

    def lemmatize_verb(
        self, string: str, morphology: dict, lookup_table: Dict[str, str]
    ) -> List[str]:
        # this method utilizes a different procedure for verbs
        # with 'nie' prefix
        if string[:3] == "nie":
            search_string = string[3:]
            if search_string in lookup_table:
                return [lookup_table[search_string]]
        return [lookup_table.get(string, string)]

    def lemmatize_noun(
        self, string: str, morphology: dict, lookup_table: Dict[str, str]
    ) -> List[str]:
        # this method is case-sensitive, in order to work
        # for incorrectly tagged proper names
        if string != string.lower():
            if string.lower() in lookup_table:
                return [lookup_table[string.lower()]]
            elif string in lookup_table:
                return [lookup_table[string]]
            return [string.lower()]
        return [lookup_table.get(string, string)]

    def lookup(self, string: str, orth: Optional[int] = None) -> str:
        return string.lower()

    def lemmatize(
        self,
        string: str,
        index: Dict[str, List[str]],
        exceptions: Dict[str, Dict[str, List[str]]],
        rules: Dict[str, List[List[str]]],
    ) -> List[str]:
        raise NotImplementedError
