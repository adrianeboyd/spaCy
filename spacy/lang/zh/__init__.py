# coding: utf8
from __future__ import unicode_literals

import tempfile
import srsly
from pathlib import Path
from collections import OrderedDict
from ...attrs import LANG
from ...language import Language
from ...tokens import Doc
from ...util import DummyTokenizer
from ..tokenizer_exceptions import BASE_EXCEPTIONS
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
from .tag_map import TAG_MAP
from ... import util


def try_jieba_import(use_jieba):
    try:
        import jieba

        return jieba
    except ImportError:
        if use_jieba:
            msg = (
                "Jieba not installed. Either set Chinese.use_jieba = False, "
                "or install it https://github.com/fxsjy/jieba"
            )
            raise ImportError(msg)


def try_pkuseg_import(use_pkuseg, pkuseg_path=""):
    try:
        import pkuseg

        if pkuseg_path:
            return pkuseg.pkuseg(pkuseg_path)
        else:
            return pkuseg.pkuseg()
    except ImportError:
        if use_pkuseg:
            msg = (
                "pkuseg not installed. Either set Chinese.use_pkuseg = False, "
                "or install it with `pip install pkuseg==0.0.22` or from "
                "https://github.com/lancopku/pkuseg-python"
            )
            raise ImportError(msg)
    except FileNotFoundError:
        if use_pkuseg:
            msg = "Unable to load pkuseg model from: " + pkuseg_path
            raise FileNotFoundError(msg)


class ChineseTokenizer(DummyTokenizer):
    def __init__(self, cls, nlp=None, settings={}):
        self.use_jieba = cls.use_jieba
        self.use_pkuseg = cls.use_pkuseg
        self.require_pkuseg = settings.get("require_pkuseg", False)
        self.vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
        self.jieba_seg = try_jieba_import(self.use_jieba)
        self.pkuseg_seg = try_pkuseg_import(
            self.use_pkuseg, pkuseg_path=settings.get("pkuseg_path", "")
        )
        self.tokenizer = Language.Defaults().create_tokenizer(nlp)
        self._pkuseg_install_msg = (
            "pkuseg not installed. To use this model, install it with "
            "`pip install pkuseg==0.0.22` or from "
            "https://github.com/lancopku/pkuseg-python"
        )


    def __call__(self, text):
        use_jieba = self.use_jieba
        use_pkuseg = self.use_pkuseg
        if self.require_pkuseg:
            use_jieba = False
            use_pkuseg = True
        # use jieba
        if use_jieba:
            jieba_words = list(
                [x for x in self.jieba_seg.cut(text, cut_all=False) if x]
            )
            words = [jieba_words[0]]
            spaces = [False]
            for i in range(1, len(jieba_words)):
                word = jieba_words[i]
                if word.isspace():
                    # second token in adjacent whitespace following a
                    # non-space token
                    if spaces[-1]:
                        words.append(word)
                        spaces.append(False)
                    # first space token following non-space token
                    elif word == " " and not words[-1].isspace():
                        spaces[-1] = True
                    # token is non-space whitespace or any whitespace following
                    # a whitespace token
                    else:
                        # extend previous whitespace token with more whitespace
                        if words[-1].isspace():
                            words[-1] += word
                        # otherwise it's a new whitespace token
                        else:
                            words.append(word)
                            spaces.append(False)
                else:
                    words.append(word)
                    spaces.append(False)
            return Doc(self.vocab, words=words, spaces=spaces)
        elif use_pkuseg:
            words = self.pkuseg_seg.cut(text)
            return Doc(self.vocab, words=words, text=text)

        # split into individual characters
        words = []
        spaces = []
        for token in self.tokenizer(text):
            if token.text.isspace():
                words.append(token.text)
                spaces.append(False)
            else:
                words.extend(list(token.text))
                spaces.extend([False] * len(token.text))
                spaces[-1] = bool(token.whitespace_)
        return Doc(self.vocab, words=words, spaces=spaces)

    def _get_config(self):
        config = {
            "use_jieba": self.use_jieba,
            "use_pkuseg": self.use_pkuseg,
            "require_pkuseg": self.require_pkuseg,
        }
        return config

    def _set_config(self, config={}):
        self.use_jieba = config.get("use_jieba", False)
        self.use_pkuseg = config.get("use_pkuseg", False)
        self.require_pkuseg = config.get("require_pkuseg", False)

    def to_bytes(self, **kwargs):
        features_b = b''
        weights_b = b''
        if self.pkuseg_seg:
            with tempfile.TemporaryDirectory() as tempdir:
                self.pkuseg_seg.feature_extractor.save(tempdir)
                self.pkuseg_seg.model.save(tempdir)
                tempdir = Path(tempdir)
                with open(tempdir / "features.pkl", "rb") as fileh:
                    features_b = fileh.read()
                with open(tempdir / "weights.npz", "rb") as fileh:
                    weights_b = fileh.read()
        serializers = OrderedDict(
            (
                ("features", lambda: features_b),
                ("weights", lambda: weights_b),
                ("config", lambda: srsly.msgpack.dumps(self._get_config())),
            )
        )
        return util.to_bytes(serializers, [])

    def from_bytes(self, data, **kwargs):
        features_b = b''
        weights_b = b''

        def deserialize_features(b):
            features_b = b

        def deserialize_weights(b):
            weights_b = b

        deserializers = OrderedDict((
            ("features", deserialize_features),
            ("weights", deserialize_weights),
            ("config", lambda b: self._set_config(srsly.msgpack_loads(b))),
        ))
        util.from_bytes(data, deserializers, [])

        if features_b and weights_b:
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = Path(tempdir)
                with open(tempdir / "features.pkl", "wb") as fileh:
                    fileh.write(deserialized["features"])
                with open(tempdir / "weights.npz", "wb") as fileh:
                    fileh.write(deserialized["weights"])
                try:
                    import pkuseg
                except ImportError:
                    raise ImportError(self._pkuseg_install_msg)
                self.pkuseg_seg = pkuseg.pkuseg(str(tempdir))
        return self

    def to_disk(self, path, **kwargs):
        path = util.ensure_path(path)

        def save_pkuseg_model(path):
            if self.pkuseg_seg:
                if not path.exists():
                    path.mkdir(parents=True)
                self.pkuseg_seg.model.save(path)
                self.pkuseg_seg.feature_extractor.save(path)

        serializers = OrderedDict((
            ("pkuseg_model", lambda p: save_pkuseg_model(p)),
            ("config", lambda p: srsly.write_msgpack(p, self._get_config())),
        ))
        return util.to_disk(path, serializers, [])

    def from_disk(self, path, **kwargs):
        path = util.ensure_path(path)
        def load_pkuseg_model(path):
            try:
                import pkuseg
            except ImportError:
                raise ImportError(self._pkuseg_install_msg)
            if path.exists():
                self.pkuseg_seg = pkuseg.pkuseg(path)
        serializers = OrderedDict((
            ("pkuseg_model", lambda p: load_pkuseg_model(p)),
            ("config", lambda p: self._set_config(srsly.read_msgpack(p))),
        ))
        util.from_disk(path, serializers, [])


class ChineseDefaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters.update(LEX_ATTRS)
    lex_attr_getters[LANG] = lambda text: "zh"
    tokenizer_exceptions = BASE_EXCEPTIONS
    stop_words = STOP_WORDS
    tag_map = TAG_MAP
    writing_system = {"direction": "ltr", "has_case": False, "has_letters": False}
    use_jieba = True
    use_pkuseg = False

    @classmethod
    def create_tokenizer(cls, nlp=None, settings={}):
        return ChineseTokenizer(cls, nlp, settings=settings)


class Chinese(Language):
    lang = "zh"
    Defaults = ChineseDefaults  # override defaults

    def make_doc(self, text):
        return self.tokenizer(text)


__all__ = ["Chinese"]
