# coding: utf8
from __future__ import unicode_literals

import re

from ...gold import Example
from ...gold import iob_to_biluo, spans_from_biluo_tags, biluo_tags_from_offsets
from ...lang.en import English
from ...tokens import Doc, Token
from .conll_ner2json import n_sents_info
from wasabi import Printer


def conllu2json(input_data, n_sents=10, use_morphology=False, lang=None,
                merge_subtokens=False, no_print=False, **_):
    """
    Convert conllu files into JSON format for use with train cli.
    use_morphology parameter enables appending morphology to tags, which is
    useful for languages such as Spanish, where UD tags are not so rich.

    Extract NER tags if available and convert them so that they follow
    BILUO and the Wikipedia scheme
    """
    # regex for identifying NER tags in MISC (10th) column of CoNLL data
    # name=NER is to handle NorNE without modifications
    MISC_NER_PATTERN = "\|?(?:name=)?(([A-Z_]+)-([A-Z_]+)|O)\|?"
    msg = Printer(no_print=no_print)
    n_sents_info(msg, n_sents)
    docs = []
    raw = ""
    sentences = []
    conll_data = read_conllx(input_data, use_morphology=use_morphology,
                             ner_tag_pattern=MISC_NER_PATTERN,
                             merge_subtokens=merge_subtokens)
    has_ner_tags = has_ner(input_data, ner_tag_pattern=MISC_NER_PATTERN)
    for i, example in enumerate(conll_data):
        raw += example.text
        sentences.append(generate_sentence(example.token_annotation,
                has_ner_tags))
        # Real-sized documents could be extracted using the comments on the
        # conllu document
        if len(sentences) % n_sents == 0:
            doc = create_json_doc(raw, sentences, i)
            docs.append(doc)
            raw = ""
            sentences = []
    if sentences:
        doc = create_json_doc(raw, sentences, i)
        docs.append(doc)
    return docs


def has_ner(input_data, ner_tag_pattern):
    """
    Check the 10th column of the first token to determine if the file contains
    NER tags
    """
    for sent in input_data.strip().split("\n\n"):
        lines = sent.strip().split("\n")
        if lines:
            while lines[0].startswith("#"):
                lines.pop(0)
            if lines:
                parts = lines[0].split("\t")
                id_, word, lemma, pos, tag, morph, head, dep, _1, misc = parts
                if re.search(ner_tag_pattern, misc):
                    return True
                else:
                    return False


def read_conllx(input_data, use_morphology=False, merge_subtokens=False,
            ner_tag_pattern=""):
    """ Yield examples, one for each sentence """
    vocab = English.Defaults.create_vocab() # need vocab to make a minimal Doc
    i = 0
    for sent in input_data.strip().split("\n\n"):
        lines = sent.strip().split("\n")
        if lines:
            while lines[0].startswith("#"):
                lines.pop(0)
            example = example_from_conllu_sentence(vocab, lines,
                    ner_tag_pattern, merge_subtokens=merge_subtokens,
                    use_morphology=use_morphology)
            yield example


def get_entities(lines, tag_pattern):
    """Find entities in the MISC column according to the pattern. Entity tag is
    'O' if the pattern is not matched.

    lines (unicode): CONLL-U lines for one sentences
    tag_pattern (unicode): Regex pattern for entity tag
    RETURNS (list): List of BILUO entity tags
    """
    iob = []
    for line in lines:
        parts = line.split("\t")
        id_, word, lemma, pos, tag, morph, head, dep, _1, misc = parts
        if "-" in id_ or "." in id_:
            continue
        iob.append(simplify_tag(misc, tag_pattern))
    return iob_to_biluo(iob)


def simplify_tag(tag, tag_pattern):
    """Simplify tag obtained from the dataset in order to follow Wikipedia
    scheme (PER, LOC, ORG, MISC). 'PER', 'LOC' and 'ORG' keep their tags, while
    'GPE_LOC' is simplified to 'LOC', 'GPE_ORG' to 'ORG' and all remaining tags
    to 'MISC'.

    tag (unicode): Contents of CoNLL-U column that may contain entity tag
    tag_pattern (unicode): Regex pattern for entity tag
    RETURNS (unicode): Entity tag if found, otherwise "O"
    """
    tag_match = re.search(tag_pattern, tag)
    new_tag = "O"
    if tag_match:
        prefix = tag_match.group(2)
        suffix = tag_match.group(3)
        if prefix and suffix:
            if suffix == "GPE_LOC":
                suffix = "LOC"
            elif suffix == "GPE_ORG":
                suffix = "ORG"
            elif suffix != "PER" and suffix != "LOC" and suffix != "ORG":
                suffix = "MISC"
            new_tag = prefix + "-" + suffix
    return new_tag


def generate_sentence(token_annotation, has_ner_tags):
    sentence = {}
    tokens = []
    for i, id in enumerate(token_annotation.ids):
        token = {}
        token["id"] = id
        token["orth"] = token_annotation.words[i]
        token["tag"] = token_annotation.tags[i]
        token["head"] = token_annotation.heads[i] - id
        token["dep"] = token_annotation.deps[i]
        #token["lemma"] = token_annotation.lemmas[i]
        #token["morph"] = token_annotation.morphs[i]
        if has_ner_tags:
            token["ner"] = token_annotation.entities[i]
        tokens.append(token)
    sentence["tokens"] = tokens
    return sentence


def create_json_doc(raw, sentences, id_):
    doc = {}
    paragraph = {}
    doc["id"] = id_
    doc["paragraphs"] = []
    paragraph["raw"] = raw.strip()
    paragraph["sentences"] = sentences
    doc["paragraphs"].append(paragraph)
    return doc


def example_from_conllu_sentence(vocab, lines, ner_tag_pattern,
        merge_subtokens=False, use_morphology=False):
    """Create an Example from the lines for one CoNLL-U sentence, merging
    subtokens and appending morphology to tags if required.

    lines (unicode): The non-comment lines for a CoNLL-U sentence
    ner_tag_pattern (unicode): The regex pattern for matching NER in MISC col
    RETURNS (Example): An example containing the annotation
    """
    # create a Doc with each subtoken as its own token
    # if merging subtokens, each subtoken orth is the merged subtoken form
    if not Token.has_extension("merged_orth"):
        Token.set_extension("merged_orth", default="")
    if not Token.has_extension("merged_lemma"):
        Token.set_extension("merged_lemma", default="")
    if not Token.has_extension("merged_morph"):
        Token.set_extension("merged_morph", default="")
    if not Token.has_extension("merged_spaceafter"):
        Token.set_extension("merged_spaceafter", default="")
    words, spaces, lemmas, tags, poss, morphs = [], [], [], [], [], []
    heads, deps = [], []
    subtok_word = ""
    in_subtok = False
    for i in range(len(lines)):
        line = lines[i]
        subtok_lines = []
        parts = line.split("\t")
        id_, word, lemma, pos, tag, morph, head, dep, _1, misc = parts
        if "." in id_:
            continue
        if "-" in id_:
            if merge_subtokens:
                in_subtok = True
            subtok_word = word
            subtok_start, subtok_end = id_.split("-")
            continue
        if not in_subtok:
            words.append(word)
        else:
            words.append(subtok_word)
        if in_subtok and id_ == subtok_end:
            subtok_word = ""
            in_subtok = False
        id_ = int(id_) - 1
        head = (int(head) - 1) if head != "0" else id_
        tag = pos if tag == "_" else tag
        morph = morph if morph != "_" else ""
        dep = "ROOT" if dep == "root" else dep
        lemmas.append(lemma)
        poss.append(pos)
        tags.append(tag)
        morphs.append(morph)
        heads.append(head)
        deps.append(dep)
        if "SpaceAfter=No" in misc:
            spaces.append(False)
        else:
            spaces.append(True)

    doc = Doc(vocab, words=words, spaces=spaces)
    for i in range(len(doc)):
        doc[i].tag_ = tags[i]
        doc[i].pos_ = poss[i]
        doc[i].dep_ = deps[i]
        doc[i].lemma_ = lemmas[i]
        doc[i].head = doc[heads[i]]
        doc[i]._.merged_orth = words[i]
        doc[i]._.merged_morph = morphs[i]
        doc[i]._.merged_lemma = lemmas[i]
        doc[i]._.merged_spaceafter = spaces[i]
    ents = get_entities(lines, ner_tag_pattern)
    doc.ents = spans_from_biluo_tags(doc, ents)
    doc.is_parsed = True
    doc.is_tagged = True

    if merge_subtokens:
        doc = merge_conllu_subtokens(lines, doc)

    # create Example from custom Doc annotation
    # TODO: morphs and lemmas
    ids, words, tags, heads, deps = [], [], [], [], []
    lemmas, morphs, spaces = [], [], []
    for i, t in enumerate(doc):
        ids.append(i)
        words.append(t._.merged_orth)
        if use_morphology and t._.merged_morph:
            sorted_morphs = "|".join(sorted(t._.merged_morph.split("|")))
            tags.append(t.tag_ + "__" + sorted_morphs)
        else:
            tags.append(t.tag_)
        heads.append(t.head.i)
        deps.append(t.dep_)
        #lemmas.append(t._.merged_lemma)
        #morphs.append(morph_to_dict(t._.merged_morph))
        spaces.append(t._.merged_spaceafter)
    ent_offsets = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    ents = biluo_tags_from_offsets(doc, ent_offsets)
    raw = ""
    for word, space in zip(words, spaces):
        raw += word
        if space:
            raw += " "
    example = Example(doc=raw)
    example.set_token_annotation(ids=ids, words=words, tags=tags,
                                 heads=heads, deps=deps, entities=ents)
    return example


def merge_conllu_subtokens(lines, doc):
    # identify and process all subtoken spans to prepare attrs for merging
    subtok_spans = []
    for line in lines:
        parts = line.split("\t")
        id_, word, lemma, pos, tag, morph, head, dep, _1, misc = parts
        if "-" in id_:
            subtok_start, subtok_end = id_.split("-")
            subtok_span = doc[int(subtok_start) - 1:int(subtok_end)]
            subtok_spans.append(subtok_span)
            # create merged tag and morph values
            # TODO: lemmas
            tags = []
            morphs = set()
            for token in subtok_span:
                tags.append(token.tag_)
                if token._.merged_morph:
                    morphs.update(token._.merged_morph.split("|"))
            # set the same attrs on all subtok tokens so that whatever head the
            # retokenizer chooses, the final attrs are available on that token
            for token in subtok_span:
                token._.merged_orth = token.orth_
                token._.merged_lemma = token.orth_
                token.tag_ = "_".join(tags)
                token._.merged_morph = "|".join(sorted(list(morphs)))
                token._.merged_spaceafter = True if subtok_span[-1].whitespace_ else False

    with doc.retokenize() as retokenizer:
        for span in subtok_spans:
            retokenizer.merge(span)

    return doc
