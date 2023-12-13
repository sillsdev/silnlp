"""
The USFM parser module, provides the default sytlesheet for USFM and
USFM specific textype parsers to the palaso.sfm module.  These guide the
palaso.sfm parser to so it can correctly parser USFM document structure.
"""
__version__ = "20101011"
__date__ = "11 October 2010"
__author__ = "Tim Eves <tim_eves@sil.org>"
__history__ = """
    20081210 - djd - Seperated SFM definitions from the module
        to allow for parsing other kinds of SFM models
        Also changed the name to parse_sfm.py as the
        module is more generalized now
    20091026 - tse - renamed and refactored generatoion of markers
        dict to module import time as part of import into palaso
        package.
    20101026 - tse - rewrote to enable the parser to use the stylesheets to
        direct how to parse structure and USFM specific semantics.
    20101109 - tse - Ensure cached usfm.sty is upto date after package code
        changes.
"""
import bz2
import contextlib
import operator
import os
import pickle
import re
import site
from copy import deepcopy
from functools import reduce
from itertools import chain

from .. import sfm
from . import ErrorLevel, style

_PALASO_DATA = os.path.join(site.getuserbase(), "palaso-python", "sfm")
_package_dir = os.path.dirname(__file__)


def _check_paths(pred, paths):
    return next(filter(pred, map(os.path.normpath, paths)), None)


def _source_path(path):
    return _check_paths(os.path.exists, [os.path.join(_PALASO_DATA, path), os.path.join(_package_dir, path)])


def _newer(cache, benchmark):
    return os.path.getmtime(benchmark) <= os.path.getmtime(cache)


def _is_fresh(cached_path, benchmarks):
    return reduce(operator.and_, (_newer(cached_path, b) for b in benchmarks))


def _cached_stylesheet(path):
    cached_path = os.path.normpath(os.path.join(_PALASO_DATA, path + os.extsep + "cz"))
    source_path = _source_path(path)
    if os.path.exists(cached_path):
        import glob

        if _is_fresh(cached_path, [source_path] + glob.glob(os.path.join(_package_dir, "*.py"))):
            return cached_path
    else:
        path = os.path.dirname(cached_path)
        if not os.path.exists(path):
            os.makedirs(path)

    import pickletools

    with contextlib.closing(bz2.BZ2File(cached_path, "wb")) as zf:
        zf.write(pickletools.optimize(pickle.dumps(style.parse(open(source_path, "r")))))
    return cached_path


def _load_cached_stylesheet(path):
    try:
        if not site.getuserbase():
            raise FileNotFoundError
        cached_path = _cached_stylesheet(path)
        try:
            try:
                with contextlib.closing(bz2.BZ2File(cached_path, "rb")) as sf:
                    return pickle.load(sf)
            except (OSError, EOFError, pickle.UnpicklingError):
                os.unlink(cached_path)
                cached_path = _cached_stylesheet(path)
                with contextlib.closing(bz2.BZ2File(cached_path, "rb")) as sf:
                    return pickle.load(sf)
        except (OSError, pickle.UnpicklingError):
            os.unlink(cached_path)
            raise
    except OSError:
        return style.parse(open(_source_path(path), "r"))


def resolve_milestones(sheet):
    for k, v in list(sheet.items()):
        if v.get("styletype", "") == "milestone":
            if "endmarker" in v:
                newm = v["endmarker"]
                v["endmarker"] = None
                if newm not in sheet:
                    sheet[newm] = deepcopy(v)
    return sheet


default_stylesheet = resolve_milestones(_load_cached_stylesheet("usfm.sty"))
relaxed_stylesheet = resolve_milestones(_load_cached_stylesheet("usfm_relaxed.sty"))

_default_meta = style.Marker(
    TextType=style.CaselessStr("Milestone"), OccursUnder={None}, Endmarker=None, StyleType=None
)


class parser(sfm.parser):
    """
    >>> import warnings

    Tests for inline markers
    >>> list(parser([r'\\test'], parser.extend_stylesheet('test')))
    [Element('test')]
    >>> list(parser([r'\\test text'], parser.extend_stylesheet('test')))
    [Element('test'), Text(' text')]
    >>> list(parser([r'\\id JHN\\ior text\\ior*']))
    [Element('id', content=[Text('JHN'), Element('ior', content=[Text('text')])])]
    >>> list(parser([r'\\id MAT\\mt Text \\f + \\fk deep\\fk*\\f*more text.']))
    [Element('id', content=[Text('MAT'), Element('mt', content=[Text('Text '), Element('f', args=['+'], content=[Element('fk', content=[Text('deep')])]), Text('more text.')])])]
    >>> list(parser([r'\\id MAT\\mt Text \\f + \\fk deep \\+qt A quote \\+qt*more\\fk*\\f*more text.']))
    [Element('id', content=[Text('MAT'), Element('mt', content=[Text('Text '), Element('f', args=['+'], content=[Element('fk', content=[Text('deep '), Element('qt', content=[Text('A quote ')]), Text('more')])]), Text('more text.')])])]

    Test end marker recognition when it's a prefix
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error")
    ...     list(parser([r'\\id TEST\\mt \\f + text\\f*suffixed text']))
    ...     list(parser([r'\\id TEST\\mt '
    ...                  r'\\f + \\fr ref \\ft text\\f*suffixed text']))
    [Element('id', content=[Text('TEST'), Element('mt', content=[Element('f', args=['+'], content=[Text('text')]), Text('suffixed text')])])]
    [Element('id', content=[Text('TEST'), Element('mt', content=[Element('f', args=['+'], content=[Element('fr', content=[Text('ref ')]), Text('text')]), Text('suffixed text')])])]

    Test footnote canonicalisation flag
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error")
    ...     list(parser([r'\\id TEST\\mt \\f + text\\f*suffixed text'],
    ...                 canonicalise_footnotes=False))
    ...     list(parser([r'\\id TEST\\mt '
    ...                  r'\\f + \\fr ref \\ft text\\f*suffixed text'],
    ...                 canonicalise_footnotes=False))
    [Element('id', content=[Text('TEST'), Element('mt', content=[Element('f', args=['+'], content=[Text('text')]), Text('suffixed text')])])]
    [Element('id', content=[Text('TEST'), Element('mt', content=[Element('f', args=['+'], content=[Element('fr', content=[Text('ref ')]), Element('ft', content=[Text('text')])]), Text('suffixed text')])])]

    Test marker parameters, particularly chapter and verse markers
    >>> list(parser([r'\\id TEST'         r'\\c 1']))
    [Element('id', content=[Text('TEST'), Element('c', args=['1'])])]
    >>> list(parser([r'\\id TEST'         r'\\c 2 \\s text']))
    [Element('id', content=[Text('TEST'), Element('c', args=['2'], content=[Element('s', content=[Text('text')])])])]
    >>> list(parser([r'\\id TEST\\c 0\\p' r'\\v 1']))
    [Element('id', content=[Text('TEST'), Element('c', args=['0'], content=[Element('p', content=[Element('v', args=['1'])])])])]
    >>> list(parser([r'\\id TEST\\c 0\\p' r'\\v 1-3']))
    [Element('id', content=[Text('TEST'), Element('c', args=['0'], content=[Element('p', content=[Element('v', args=['1-3'])])])])]
    >>> list(parser([r'\\id TEST\\c 0\\p' r'\\v 2 text']))
    [Element('id', content=[Text('TEST'), Element('c', args=['0'], content=[Element('p', content=[Element('v', args=['2']), Text('text')])])])]
    >>> list(parser([r'\\id TEST'         r'\\c 2 \\p \\v 3 text\\v 4 verse']))
    [Element('id', content=[Text('TEST'), Element('c', args=['2'], content=[Element('p', content=[Element('v', args=['3']), Text('text'), Element('v', args=['4']), Text('verse')])])])]

    Test for error detection and reporting for structure
    >>> list(parser([r'\\id TEST\\mt text\\f*']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,17: orphan end marker \\f*: no matching opening marker \\f
    >>> list(parser([r'\\id TEST     \\p 1 text']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,14: orphan marker \\p: may only occur under \\c
    >>> list(parser([r'\\id TEST\\mt \\f + text\\fe*']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,22: orphan end marker \\fe*: no matching opening marker \\fe
    >>> list(parser([r'\\id TEST\\mt \\f + text'], ))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,1: invalid end marker end-of-file: \\f (line 1,13) can only be closed with \\f*

    Test for error detection and reporting for USFM specific parses
    Chapter numbers
    >>> list(parser(['\\id TEST\\c\\p \\v 1 text']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,9: missing chapter number after \\c
    >>> list(parser(['\\id TEST\\c A\\p \\v 1 text']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,9: missing chapter number after \\c
    >>> list(parser([r'\\id TEST\\c 1 text\\p \\v 1 text']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,14: text cannot follow chapter marker '\\c 1'
    >>> list(parser([r'\\id TEST\\c 1text\\p \\v 1 text']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,13: missing space after chapter number '1'

    Verse numbers
    >>> list(parser([r'\\id TEST\\c 1\\p \\v \\p text']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,16: missing verse number after \\v
    >>> list(parser([r'\\id TEST\\c 1\\p \\v text']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,16: missing verse number after \\v
    >>> list(parser([r'\\id TEST\\c 1\\p \\v 1text']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,21: missing space after verse number '1t'

    Note text parsing
    >>> list(parser([r'\\id TEST\\mt \\f \\fk key\\fk* text.\\f*']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,13: missing caller parameter after \\f
    >>> list(parser([r'\\id TEST\\mt \\f +text \\fk key\\fk* text.\\f*']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,17: missing space after caller parameter '+'

    Test warnable condition detection and reporting
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error", SyntaxWarning)
    ...     list(parser([r'\\id TEST\\mt \\whoops']))
    Traceback (most recent call last):
    ...
    SyntaxWarning: <string>: line 1,14: unknown marker \whoops: not in stylesheet
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error", SyntaxWarning)
    ...     list(parser([r'\\id TEST\\mt \\whoops'],
    ...                 error_level=sfm.ErrorLevel.Marker))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,14: unknown marker \whoops: not in stylesheet
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error", SyntaxWarning)
    ...     list(parser([r'\\id TEST\\mt \\zwhoops'],
    ...                 error_level=sfm.ErrorLevel.Note))
    Traceback (most recent call last):
    ...
    SyntaxWarning: <string>: line 1,14: unknown private marker \zwhoops: not it stylesheet using default marker definition
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error", SyntaxWarning)
    ...     list(parser([r'\\id TEST\\c 1\\p a \\png b \\+w c \\+nd d \\png e \\png*']))
    ... # doctest: +NORMALIZE_WHITESPACE
    [Element('id',
        content=[Text('TEST'),
                 Element('c', args=['1'],
                    content=[Element('p',
                        content=[Text('a '),
                                 Element('png',
                                    content=[Text('b '),
                                             Element('w',
                                                content=[Text('c '),
                                                         Element('nd',
                                                            content=[Text('d ')])])]),
                                 Element('png',
                                    content=[Text('e ')])])])])]
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error", SyntaxWarning)
    ...     list(parser([r'\\id TEST\\c 1\\p a \\f + \\fr 1:1 \\ft a \\png b\\png*']))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,1: invalid end marker end-of-file: \\f (line 1,18) can only be closed with \\f*
    """  # noqa: E501, W605

    default_meta = _default_meta
    numeric_re = re.compile(r"\s*(\d+(:?[-\u2010\2011]\d+)?)", re.UNICODE)
    verse_re = re.compile(r"\s*(\d+\w?(:?[-,\u200B-\u2011]+\d+\w?)?)", re.UNICODE)
    caller_re = re.compile(r"\s*([^\s\\]+)", re.UNICODE)
    sep_re = re.compile(r"\s|$", re.UNICODE)
    __unspecified_metas = {
        ("Section", True): "s",
        ("Title", True): "t",
        ("VerseText", True): "p",
        ("VerseText", False): "nd",
        ("Other", True): "p",
        ("Other", False): "nd",
        ("Unspecified", True): "p",
        ("Unspecified", False): "nd",
    }

    @classmethod
    def extend_stylesheet(cls, *names, **kwds):
        return super().extend_stylesheet(kwds.get("stylesheet", default_stylesheet), *names)

    def __init__(
        self,
        source,
        stylesheet=default_stylesheet,
        default_meta=_default_meta,
        canonicalise_footnotes=True,
        tag_escapes=r"\\",
        *args,
        **kwds,
    ):
        if not canonicalise_footnotes:
            self._canonicalise_footnote = lambda x: x

        stylesheet = self.__synthesise_private_meta(stylesheet, default_meta)
        super().__init__(
            source,
            stylesheet,
            default_meta,
            private_prefix="z",
            tokeniser=re.compile(rf"(?:\\(?:{tag_escapes})|[^\\])+|\\[^\s\\|]+", re.DOTALL | re.UNICODE),
            *args,
            **kwds,
        )

    @classmethod
    def __synthesise_private_meta(cls, sty, default_meta):
        private_metas = dict(r for r in sty.items() if r[0].startswith("z"))
        metas = {
            n: sty.get(
                cls.__unspecified_metas.get(
                    (m["TextType"], m["Endmarker"] is None and m.get("StyleType", None) == "Paragraph"), None
                ),
                default_meta,
            ).copy()
            for n, m in private_metas.items()
        }
        return style.update_sheet(
            sty,
            style.update_sheet(metas, private_metas, field_update=style.FieldUpdate.IGNORE),
            field_update=style.FieldUpdate.IGNORE,
        )

    def _force_close(self, parent, tok):
        if tok is not sfm.parser._eos and (
            "NoteText" in parent.meta.get("TextType", []) or parent.meta.get("StyleType", None) == "Character"
        ):
            self._error(
                ErrorLevel.Note,
                "implicit end marker before {token}: \\{0.name} "
                "(line {0.pos.line},{0.pos.col}) "
                "should be closed with \\{1}",
                tok,
                parent,
                parent.meta["Endmarker"],
            )
        else:
            super()._force_close(parent, tok)

    def _ChapterNumber_(self, chapter_marker):
        tok = next(self._tokens)
        chapter = self.numeric_re.match(str(tok))
        if not chapter:
            self._error(ErrorLevel.Content, "missing chapter number after \\c", chapter_marker)
            chapter_marker.args = ["\uFFFD"]
        else:
            chapter_marker.args = [str(tok[chapter.start(1) : chapter.end(1)])]
            tok = tok[chapter.end() :]
        if tok and not self.sep_re.match(str(tok)):
            self._error(
                ErrorLevel.Content,
                "missing space after chapter number '{chapter}'",
                tok,
                chapter=chapter_marker.args[0],
            )
        if tok == "\n":
            chapter_marker.append(tok)
        else:
            tok = tok.lstrip()
            if tok:
                if tok[0] == "\\":
                    self._tokens.put_back(tok)
                else:
                    self._error(ErrorLevel.Structure, "text cannot follow chapter marker '{0}'", tok, chapter_marker)
                    chapter_marker.append(sfm.Element(None, meta=self.default_meta, content=[tok]))
                    tok = None

        return self._default_(chapter_marker)

    _chapternumber_ = _ChapterNumber_

    def _VerseNumber_(self, verse_marker):
        tok = next(self._tokens)
        verse = self.verse_re.match(str(tok))
        if not verse:
            self._error(ErrorLevel.Content, "missing verse number after \\v", verse_marker)
            verse_marker.args = ["\uFFFD"]
        else:
            verse_marker.args = [str(tok[verse.start(1) : verse.end(1)])]
            tok = tok[verse.end() :]

        if not self.sep_re.match(str(tok)):
            self._error(
                ErrorLevel.Content, "missing space after verse number '{verse}'", tok, verse=verse_marker.args[0]
            )
        tok = tok[1:]

        if tok:
            self._tokens.put_back(tok)
        return tuple()

    _versenumber_ = _VerseNumber_

    @staticmethod
    def _canonicalise_footnote(content):
        def g(e):
            if getattr(e, "name", None) == "ft":
                e.parent.annotations["content-promoted"] = True
                if len(e.parent) > 0:
                    prev = e.parent[-1]
                    if prev.meta["StyleType"] == "Character":
                        del prev.annotations["implicit-closed"]
                return e
            else:
                return [e]

        return chain.from_iterable(map(g, content))

    def _NoteText_(self, parent):
        if parent.meta.get("StyleType") != "Note":
            return self._default_(parent)

        tok = next(self._tokens)
        caller = self.caller_re.match(str(tok))
        if not caller:
            self._error(ErrorLevel.Content, "missing caller parameter after \\{token.name}", parent)
            parent.args = ["\uFFFD"]
        else:
            parent.args = [str(tok[caller.start(1) : caller.end(1)])]
            tok = tok[caller.end() :]

        if not self.sep_re.match(str(tok)):
            self._error(
                ErrorLevel.Content, "missing space after caller parameter '{caller}'", tok, caller=parent.args[0]
            )

        if tok.lstrip():
            self._tokens.put_back(tok)

        return self._canonicalise_footnote(self._default_(parent))

    _notetext_ = _NoteText_

    def _Unspecified_(self, parent):
        orig_name = parent.name
        if parent.meta.get("StyleType") == "Paragraph" or (
            parent.parent is not None
            and parent.parent.meta.get("StyleType") == "Note"
            and "Endmarker" not in parent.meta
        ):
            parent.name = "p"
        subparse = self._default_(parent)
        parent.name = orig_name
        return subparse

    _unspecified_ = _Unspecified_


class Reference(sfm.Position):
    def __new__(cls, pos, ref):
        p = super().__new__(cls, *pos)
        p.book = ref[0]
        p.chapter = ref[1]
        p.verse = ref[2]
        return p


def decorate_references(source):
    ref = [None, None, None]

    def _g(_, e):
        if isinstance(e, sfm.Element):
            if e.name == "id":
                ref[0] = str(e[0]).split()[0]
            elif e.name == "c":
                ref[1] = e.args[0]
            elif e.name == "v":
                ref[2] = e.args[0]
            e.pos = Reference(e.pos, ref)
            return reduce(_g, e, None)
        else:
            e.pos = Reference(e.pos, ref)

    source = list(source)
    reduce(_g, source, None)
    return source
