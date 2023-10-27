"""
The SFM parser module. It provides the basic stylesheet guided sfm parser
and default TextType parser.  This parser provides detailed error and
diagnostic messages including accurate line and column information as well
as context information for structure errors.
The default marker meta data makes this produce only top-level marker-text
pairs.
"""
__version__ = "20200813"
__date__ = "13 August 2020"
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
        direct how to parse document structure and TextType to permit
        specific semantics on subsections.
    20101028 - tse - Handle endmarkers as a special case so the do no require
        a space to separate them from the following text.
    20101109 - tse - Fix separator space being included when it shouldn't and
        use the unique field types set object to improve performance.
"""
import collections
import operator
import os
import re
import warnings
from enum import IntEnum
from functools import reduce
from itertools import chain, groupby
from typing import NamedTuple

__all__ = (
    "usfm",  # Sub modules
    "Position",
    "Element",
    "Text",
    "ErrorLevel",
    "parser",  # data types
    "sreduce",
    "smap",
    "sfilter",
    "mpath",
    "text_properties",
    "format",
    "copy",
)  # functions


class Position(NamedTuple):
    line: int
    col: int

    """Immutable position data that attach to tokens"""

    def __str__(self) -> str:
        return f"line {self.line},{self.col}"


class Element(list):
    """
    An sequence type representing a marker and it's child nodes which may
    consist of more elements or text nodes. It contains metadata describing how
    it was parsed and an annotations mapping to allow the parser or further
    processing steps to attach notes or data to it.

    >>> Element('marker')
    Element('marker')

    >>> str(Element('marker'))
    '\\\\marker'

    >>> str(Element('marker', args=['1','2']))
    '\\\\marker 1 2'

    >>> e = Element('marker', args=['1'],
    ...             content=[Text('some text '),
    ...                      Element('marker2',
    ...                              content=[Text('more text\\n')]),
    ...                      Element('blah',content=[Text('\\n')]),
    ...                      Element('blah',content=[Text('\\n')]),
    ...                      Element('yak', args=['yak'])])
    >>> len(e)
    5
    >>> e.name
    'marker'
    >>> e.pos
    Position(line=1, col=1)
    >>> e.meta
    {}
    >>> print(str(e))
    \\marker 1 some text \\marker2 more text
    \\blah
    \\blah
    \\yak yak
    >>> Element('marker') == Element('marker')
    True
    >>> Element('marker') == Element('different')
    False
    """

    # __slots__ = ('pos', 'name', 'args', 'parent', 'meta', 'annotations')

    def __init__(self, name, pos=Position(1, 1), args=[], parent=None, meta={}, content=[]):
        """
        The `name` of the marker, and optionally any marker argments in
        `args`. If this element is a child of another it `parent` should point
        back to that element. The position in the source text can be supplied
        with `pos`, and the marker stylesheet record used to parse it should be
        passed with meta.
        """
        super().__init__(content)
        self.name = str(name) if name else None
        self.pos = pos
        self.args = args
        self.parent = parent
        self.meta = meta
        self.annotations = {}

    def __repr__(self):
        args = (
            [repr(self.name)] + (self.args and [f"args={self.args!r}"]) + (self and [f"content={super().__repr__()}"])
        )
        return f"Element({', '.join(args)!s})"

    def __eq__(self, rhs):
        if not isinstance(rhs, Element):
            return False
        return (
            self.name == rhs.name
            and self.args == rhs.args
            and (not (self.meta and rhs.meta) or self.meta == rhs.meta)
            and super().__eq__(rhs)
        )

    def __str__(self):
        marker = ""
        nested = "+" if "nested" in self.annotations else ""
        if self.name:
            marker = f"\\{nested}{' '.join([self.name] + self.args)}"
        endmarker = self.meta.get("Endmarker", "")
        body = "".join(map(str, self))
        sep = ""
        if len(self) > 0:
            if not body.startswith(("\r\n", "\n")):
                sep = " "
        elif self.meta.get("StyleType") == "Character":
            body = " "

        if endmarker and "implicit-closed" not in self.annotations:
            body += f"\\{nested}{endmarker}"
        return sep.join([marker, body])


class Text(collections.UserString):
    """
    An extended string type that tracks position and
    parent/child relationship.

    >>> from pprint import pprint
    >>> Text('a test')
    Text('a test')

    >>> Text('prefix ',Position(3,10)).pos, Text('suffix',Position(1,6)).pos
    (Position(line=3, col=10), Position(line=1, col=6))

    >>> t = Text('prefix ',Position(3,10)) + Text('suffix',Position(1,6))
    >>> t, t.pos
    (Text('prefix suffix'), Position(line=3, col=10))

    >>> t = Text('a few short words')[12:]
    >>> t, t.pos
    (Text('words'), Position(line=1, col=13))

    >>> t = Text('   yuk spaces   ').lstrip()
    >>> t, t.pos
    (Text('yuk spaces   '), Position(line=1, col=4))

    >>> t = Text('   yuk spaces   ').rstrip()
    >>> t, t.pos
    (Text('   yuk spaces'), Position(line=1, col=1))

    >>> Text('   yuk spaces   ').strip()
    Text('yuk spaces')

    >>> pprint([(t,t.pos) for t in Text('a few short words').split(' ')])
    [(Text('a'), Position(line=1, col=1)),
     (Text('few'), Position(line=1, col=3)),
     (Text('short'), Position(line=1, col=7)),
     (Text('words'), Position(line=1, col=13))]

    >>> list(map(str, Text('a few short words').split(' ')))
    ['a', 'few', 'short', 'words']

    >>> t=Text.concat([Text('a ', pos=Position(line=1, col=1)),
    ...                Text('few ', pos=Position(line=1, col=3)),
    ...                Text('short ', pos=Position(line=1, col=7)),
    ...                Text('words', pos=Position(line=1, col=13))])
    >>> t, t.pos
    (Text('a few short words'), Position(line=1, col=1))
    """

    # def __new__(cls, content, pos=Position(1, 1), parent=None):
    #    return super().__new__(cls, content)

    def __init__(self, content, pos=Position(1, 1), parent=None):
        super().__init__(str(content))
        self.pos = pos
        self.parent = parent

    @staticmethod
    def concat(iterable):
        i = iter(iterable)
        h = next(i)
        line = [str(h)] + [str(s) for s in i]
        return Text("".join(line), h.pos, h.parent)
        # return Text(''.join(chain([str(h)], i)), h.pos, h.parent)

    def split(self, sep=None, maxsplit=-1):
        sep = sep or " "
        tail = self
        result = []
        while tail and maxsplit != 0:
            e = tail.find(sep)
            if e == -1:
                result.append(tail)
                tail = Text("", Position(self.pos.line, self.pos.col + len(tail)), self.parent)
                break
            result.append(tail[:e])
            tail = tail[e + len(sep) :]
            maxsplit -= 1
        if tail:
            result.append(tail)
        return result

    def lstrip(self, *args, **kwds):
        n = len(self)
        s_ = super().lstrip(*args, **kwds)
        return Text(s_, Position(self.pos.line, self.pos.col + n - len(s_)), self.parent)

    def rstrip(self, *args, **kwds):
        s_ = super().rstrip(*args, **kwds)
        return Text(s_, self.pos, self.parent)

    def strip(self, *args, **kwds):
        return self.lstrip(*args, **kwds).rstrip(*args, **kwds)

    def __repr__(self):
        return f"Text({super().__repr__()!s})"

    def __add__(self, rhs):
        return Text(super().__add__(rhs), self.pos, self.parent)

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __getitem__(self, i):
        return Text(
            super().__getitem__(i),
            Position(self.pos.line, self.pos.col + (i.start or 0 if isinstance(i, slice) else i)),
            self.parent,
        )


class _put_back_iter(collections.Iterator):
    """
    >>> i=_put_back_iter([1,2,3])
    >>> next(i)
    1
    >>> next(i)
    2
    >>> i.put_back(256)
    >>> next(i)
    256
    >>> i.peek()
    3
    >>> i.put_back(512)
    >>> i.peek()
    512
    >>> next(i); next(i)
    512
    3
    >>> next(i)
    Traceback (most recent call last):
    ...
    StopIteration
    """

    def __init__(self, iterable):
        self._itr = iter(iterable)
        self._pbq = []

    def __next__(self):
        return self.next()

    def next(self):
        if self._pbq:
            try:
                return self._pbq.pop()
            except Exception:
                raise StopIteration
        return next(self._itr)

    def put_back(self, value):
        self._pbq.append(value)

    def peek(self):
        if not self._pbq:
            self._pbq.append(next(self._itr))
        return self._pbq[-1]


_default_meta = dict(TextType="default", OccursUnder={None}, Endmarker=None, StyleType=None)


class ErrorLevel(IntEnum):
    """
    Error levels used to control what the parser reports as warnings or errors
    """

    Note = -1
    "Issues that do not affect the parse, but should be corrected."
    Marker = 0
    "Markers that are not in the stylesheet or private namespace."
    Content = 1
    "Issues with parsing or formating marker arguments or element content."
    Structure = 2
    "Issues that result in incorectly parsed document structure, if ignored."
    Unrecoverable = 100
    "Always reported as an error, parser cannot make progress past it."


class Tag(NamedTuple):
    name: str
    nested: bool = False

    def __str__(self) -> str:
        return f"\\{'+' if self.nested else ''}{self.name}"

    @property
    def endmarker(self) -> bool:
        return self.name[-1] == "*"


class parser(collections.Iterable):
    '''
    SFM parser, and base class for more complex parsers such as USFM and the
    stylesheet parser.  This can be used to parse unstructured SFM files as a
    sequence of childless elements or text nodes. By supplying a stylesheet
    structure can be extracted from a source, such as a USFM document.

    Various options allow customisation of the strictness of error checking,
    private tag space prefix or even what constitutes a marker, beyond just
    starting with a '\\'.

    >>> from pprint import pprint
    >>> import warnings

    Test edge case text handling
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     list(parser([])); list(parser([''])); list(parser('plain text'))
    []
    []
    [Text('plain text')]

    Test mixed marker and text and cross line coalescing
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     pprint(list(parser(r"""\\lonely
    ... \\sfm text
    ... bare text
    ... \\more-sfm more text
    ... over a line break\\marker""".splitlines(True))))
    [Element('lonely', content=[Text('\\n')]),
     Element('sfm', content=[Text('text\\nbare text\\n')]),
     Element('more-sfm', content=[Text('more text\\nover a line break')]),
     Element('marker')]

    Default Backslash handling (just '\\')
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     pprint(list(parser([
    ...         r"\\marker text",
    ...         r"\\escaped backslash\\\\character",
    ...         r"\\t1 \\t2 \\\\backslash \\^hat \\%\\t3\\\\\\^"])))
    [Element('marker', content=[Text('text')]),
     Element('escaped', content=[Text('backslash\\\\\\\\character')]),
     Element('t1'),
     Element('t2', content=[Text('\\\\\\\\backslash ')]),
     Element('^hat'),
     Element('%'),
     Element('t3', content=[Text('\\\\\\\\')]),
     Element('^')]

    Specify extra escapable characters or tokens, in this case everything that
    doesn't start with a number or letter isn't a tag.
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     pprint(list(parser([
    ...         r"\\t1 \\t2 \\\\backslash \\^hat \\%\\t3\\\\\\^"],
    ...         tag_escapes=r"[^0-9a-zA-Z]")))
    [Element('t1'),
     Element('t2', content=[Text('\\\\\\\\backslash \\\\^hat \\\\%')]),
     Element('t3', content=[Text('\\\\\\\\\\\\^')])]


    >>> doc=r"""
    ... \\id MAT EN
    ... \\ide UTF-8
    ... \\rem from MATTHEW
    ... \\h Mathew
    ... \\toc1 Mathew
    ... \\mt1 Mathew
    ... \\mt2 Gospel Of Matthew"""
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     pprint(list(parser(doc.splitlines(True))))
    [Text('\\n'),
     Element('id', content=[Text('MAT EN\\n')]),
     Element('ide', content=[Text('UTF-8\\n')]),
     Element('rem', content=[Text('from MATTHEW\\n')]),
     Element('h', content=[Text('Mathew\\n')]),
     Element('toc1', content=[Text('Mathew\\n')]),
     Element('mt1', content=[Text('Mathew\\n')]),
     Element('mt2', content=[Text('Gospel Of Matthew')])]

    >>> tss = parser.extend_stylesheet({},'id','ide','rem','h','toc1',
    ...                                'mt1','mt2')
    >>> pprint(tss)
    {'h': {'Endmarker': None,
           'OccursUnder': {None},
           'StyleType': None,
           'TextType': 'default'},
     'id': {'Endmarker': None,
            'OccursUnder': {None},
            'StyleType': None,
            'TextType': 'default'},
     'ide': {'Endmarker': None,
             'OccursUnder': {None},
             'StyleType': None,
             'TextType': 'default'},
     'mt1': {'Endmarker': None,
             'OccursUnder': {None},
             'StyleType': None,
             'TextType': 'default'},
     'mt2': {'Endmarker': None,
             'OccursUnder': {None},
             'StyleType': None,
             'TextType': 'default'},
     'rem': {'Endmarker': None,
             'OccursUnder': {None},
             'StyleType': None,
             'TextType': 'default'},
     'toc1': {'Endmarker': None,
              'OccursUnder': {None},
              'StyleType': None,
              'TextType': 'default'}}

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error")
    ...     pprint(list(parser(doc.splitlines(True), tss)))
    [Text('\\n'),
     Element('id', content=[Text('MAT EN\\n')]),
     Element('ide', content=[Text('UTF-8\\n')]),
     Element('rem', content=[Text('from MATTHEW\\n')]),
     Element('h', content=[Text('Mathew\\n')]),
     Element('toc1', content=[Text('Mathew\\n')]),
     Element('mt1', content=[Text('Mathew\\n')]),
     Element('mt2', content=[Text('Gospel Of Matthew')])]
    >>> tss['rem'].update(OccursUnder={'ide'})
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error")
    ...     pprint(list(parser(doc.splitlines(True), tss)))
    ... # doctest: +NORMALIZE_WHITESPACE
    [Text('\\n'),
     Element('id', content=[Text('MAT EN\\n')]),
     Element('ide',
             content=[Text('UTF-8\\n'),
                      Element('rem', content=[Text('from MATTHEW\\n')])]),
     Element('h', content=[Text('Mathew\\n')]),
     Element('toc1', content=[Text('Mathew\\n')]),
     Element('mt1', content=[Text('Mathew\\n')]),
     Element('mt2', content=[Text('Gospel Of Matthew')])]
    >>> del tss['mt1']
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error")
    ...     pprint(list(parser(doc.splitlines(True), tss)))
    Traceback (most recent call last):
    ...
    SyntaxWarning: <string>: line 7,2: unknown marker \\mt1: not in stylesheet
    '''

    default_meta = _default_meta
    _eos = Text("end-of-file")

    @classmethod
    def extend_stylesheet(cls, stylesheet, *names):
        return dict({m: cls.default_meta.copy() for m in names}, **stylesheet)

    def __init__(
        self,
        source,
        stylesheet={},
        default_meta=default_meta,
        private_prefix=None,
        error_level=ErrorLevel.Content,
        tag_escapes=r"\\",
        tokeniser=None,
    ):
        """
        Create a SFM parser object. This object is an interator over SFM
        Element trees. For simple unstructured documents this is one element
        per line, for more complex stylsheet guided parse this can be one or
        more complete element trees.

        source: An interable sequence of lines, such as a File like object,
            representing the document.
        stylesheet: A style sheet dict mapping marker names to metadata, used
            to guide parsing documet structure. Optional
        default_meta: Marker metadata to use when a marker cannot be found in
            the stylesheet or is in the private namespace. If none are supplied
            the following is used: dict(TextType=default',
                                        OccursUnder={None},
                                        Endmarker=None,
                                        StyleType=None)
        private_prefix: Prefix identifying the private marker namespace. If a
            marker begins with this, then it is not an error for it to not
            appear in the stylesheet and it assigned default metadata.
            Optional: If not passed there is no private namespace defined.
        error_level: The ErrorLevel at which or above the parser should report
            issues as Errors instead of Warnings.
        tag_escapes: A regular expression which defines the set of marker
            names that are treated as text instead or parsed as markers. This
            is matched against the marker name after the initial slash.
            Optional, defaults to r'\\\\' to allow for escaping the backslash.
        """
        # Pick the marker lookup failure mode.
        assert (
            default_meta or not private_prefix
        ), "default_meta must be provided when using private_prefix"  # noqa: E501
        assert (
            error_level <= ErrorLevel.Marker or default_meta
        ), "default meta must be provided when error_level > ErrorLevel.Marker"  # noqa: E501

        # Set simple attributes
        self.source = getattr(source, "name", "<string>")
        self._default_meta = default_meta
        self._pua_prefix = private_prefix
        if tokeniser is None:
            tokeniser = re.compile(rf"(?:\\(?:{tag_escapes})|[^\\])+|\\[^\s\\]+", re.DOTALL | re.UNICODE)
        self._tokens = _put_back_iter(self.__lexer(source, tokeniser))
        self._error_level = error_level
        self._escaped_tag = re.compile(rf"^\\{tag_escapes}", re.DOTALL | re.UNICODE)

        # Compute end marker stylesheet definitions
        em_def = {"TextType": None, "Endmarker": None}
        self._sty = stylesheet.copy()
        self._sty.update(
            (m["Endmarker"], type(m)(em_def, OccursUnder={k})) for k, m in stylesheet.items() if m["Endmarker"]
        )

    def _error(self, severity, msg, ev, *args, **kwds):
        """
        Raise a SyntaxError or SyntaxWarning, or skip as appropriate based on
        the error level in the parser and the severity of the error. The error
        message will have the source file and line and column of the issue
        prepended to the caller supplied message.

        severity: The severity or the issue being reported.
        msg: specific message about the problem, if this includes any of the
            following format syntax markers they will be filled out:
                {token}:  The Text object ev representing the subject token.
                {source}: The source name.
        ev: The Text object representing the token at which the issue occured.
        Any remaining aguments or keyword arguments are used to format the msg
        string.
        """
        path = []
        pev = ev.parent
        while pev is not None:
            path.append(pev.name)
            pev = pev.parent
        pathstr = " [" + " ".join(reversed(path)) + "]" if len(path) else ""
        msg = f"{self.source}: line {ev.pos.line},{ev.pos.col}{pathstr}: " + str(msg).format(
            token=ev, source=self.source, *args, **kwds
        )
        if severity >= 0 and severity >= self._error_level:
            raise SyntaxError(msg)
        elif severity < 0 and severity < self._error_level:
            pass
        else:
            warnings.warn_explicit(msg, SyntaxWarning, self.source, ev.pos.line)

    def __get_style(self, tag):
        meta = self._sty.get(tag.lstrip("+"))
        if not meta:
            if self._pua_prefix and tag.startswith(self._pua_prefix):
                self._error(
                    ErrorLevel.Note,
                    "unknown private marker \\{token}: " "not it stylesheet using default marker definition",
                    tag,
                )
            else:
                self._error(ErrorLevel.Marker, "unknown marker \\{token}: not in stylesheet", tag)
            return self._default_meta

        return meta

    def __iter__(self):
        return self._default_(None)

    @staticmethod
    def __pp_marker_list(tags):
        return ", ".join("\\" + c if c else "toplevel" for c in sorted(tags))

    @staticmethod
    def __lexer(lines, tokeniser):
        """Return an iterator that returns tokens in a sequence:
        marker, text, marker, text, ...
        """
        lmss = enumerate(map(tokeniser.finditer, lines))
        fs = (Text(m.group(), Position(li + 1, m.start() + 1)) for li, ms in lmss for m in ms)
        gs = groupby(fs, operator.methodcaller("startswith", "\\"))
        return chain.from_iterable(g if istag else (Text.concat(g),) for istag, g in gs)

    def __get_tag(self, parent: Element, tok: str):
        if tok[0] != "\\" or self._escaped_tag.match(str(tok)):
            return None

        tok = tok[1:]
        tag = Tag(tok.lstrip("+"), tok[0] == "+")
        if parent is None:
            return tag

        # Check for the expected end markers with no separator and
        # break them apart
        while parent.meta["Endmarker"]:
            if tag.name.startswith(parent.meta["Endmarker"]):
                cut = len(parent.meta["Endmarker"])
                rest = tag.name[cut:]
                if rest:
                    tag = Tag(tag.name[:cut], tag.nested)
                    if self._tokens.peek()[0] != "\\":
                        # If the next token isn't a marker, coaleces the
                        # remainder with it into a single text node.
                        rest += next(self._tokens, "")
                    self._tokens.put_back(rest)
                return tag
            parent = parent.parent
        return tag

    @staticmethod
    def __need_subnode(parent, tag, meta):
        occurs = meta["OccursUnder"]
        if not occurs:  # No occurs under means it can occur anywhere.
            return True

        parent_tag = None
        if parent is not None:
            if tag.nested and not tag.endmarker:
                if not parent.meta["StyleType"] == "Character":
                    return False
                while parent.meta["StyleType"] == "Character":
                    parent = parent.parent
            parent_tag = getattr(parent, "name", None)
        return parent_tag in occurs

    def _default_(self, parent):
        for tok in self._tokens:
            tag = self.__get_tag(parent, tok)
            if tag:  # Parse markers.
                if tag.name == "*" and parent is not None:
                    return
                meta = self.__get_style(tag.name)
                if self.__need_subnode(parent, tag, meta):
                    sub_parser = meta.get("TextType")
                    if not sub_parser:
                        return
                    sub_parser = getattr(self, "_" + sub_parser + "_", self._default_)
                    # Spawn a sub-node
                    e = Element(tag.name, tok.pos, parent=parent, meta=meta)
                    # and recurse
                    if tag.nested:
                        e.annotations["nested"] = True
                    e.extend(sub_parser(e))
                    yield e
                elif parent is None:
                    tok = Text(tag, tok.pos, tok.parent)
                    # We've failed to find a home for marker tag, poor thing.
                    if not meta["TextType"]:
                        assert len(meta["OccursUnder"]) == 1
                        self._error(
                            ErrorLevel.Unrecoverable,
                            "orphan end marker {token}: " "no matching opening marker \\{0}",
                            tok,
                            list(meta["OccursUnder"])[0],
                        )
                    else:
                        self._error(
                            ErrorLevel.Unrecoverable,
                            "orphan marker {token}: " "may only occur under {0}",
                            tok,
                            self.__pp_marker_list(meta["OccursUnder"]),
                        )
                else:
                    tok = Text(tag, tok.pos, tok.parent)
                    # Do implicit closure only for non-inline markers or
                    # markers inside NoteText type markers'.
                    if parent.meta["Endmarker"]:
                        self._force_close(parent, tok)
                        parent.annotations["implicit-closed"] = True
                    self._tokens.put_back(tok)
                    return
            else:  # Pass non marker data through with a litte fix-up
                if parent is not None and len(parent) == 0 and tok.startswith(" "):
                    tok = tok[1:]
                if tok:
                    tok.parent = parent
                    yield tok
        if parent is not None:
            if parent.meta["Endmarker"]:
                self._force_close(parent, self._eos)
                parent.annotations["implicit-closed"] = True
        return

    def _Milestone_(self, parent):
        return tuple()

    _milestone_ = _Milestone_

    _Other_ = _default_
    _other_ = _Other_
    _Unspecified_ = _default_
    _unspecified_ = _Unspecified_

    def _force_close(self, parent, tok):
        """
        Called by the parser when it needs to force an element, which has an
        Endmarker, closed due to implicit closure, such as an outer inline
        endmarker or paragraph level marker closing any currently open inline
        markers.
        This method should be overridden by subclasses to change default
        behaviour of treating this as a structural error.

        parent: The Element being forced closed.
        tok: The token causing the implicit closure.
        """
        self._error(
            ErrorLevel.Structure,
            "invalid end marker {token}: \\{0.name} " "(line {0.pos.line},{0.pos.col}) can only be closed with \\{1}",
            tok,
            parent,
            parent.meta["Endmarker"],
        )


def sreduce(elementf, textf, trees, initial):
    """
    Reduce sequence of element trees down to a single object. Used for same
    tasks a normal reduce but it operates on the element tree structure rather
    than a linear sequence. As such it takes 2 functions for handling nodes and
    leaves idependantly.

    elementf: A callable that accepts 3 parameters and returns a new
        accumulator value.
        e: The Element node under consideration.
        a: The current accumulator object.
        body: The elements reduced children.
    textf: A callable the accepts 2 parameters and returns a new accumulator
        value.
        t: The Text node under consideration.
        a: The current accumulator object.
    trees: An iterable over Element trees, generaly the output of parser().
    initial:  The initial value for the accumulator.

    A crude word count example:
    >>> doc =r'''\\lonely
    ... \\sfm text
    ... bare text
    ... \\more-sfm more text
    ... over a line break\\marker'''.splitlines(True)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     sreduce(lambda e, a, b: 1 + len(e.args) + a + b,
    ...             lambda t, a: a + len(t.split()),
    ...             parser(doc),
    ...             0)
    12
    """

    def _g(a, e):
        if isinstance(e, (str, Text)):
            return textf(e, a)
        return elementf(e, a, reduce(_g, e, initial))

    return reduce(_g, trees, initial)


def smap(elementf, textf, trees):
    """
    Map sequence of element trees down into another sequence of structurally
    identical element trees. Used for same tasks a normal map but it operates
    on the element tree structure rather than a linear sequence. As such it
    takes 2 functions for handling nodes and leaves idependantly.

    elementf: A callable that accepts 3 parameters and returns a tuple of
            (name, args, children)
        name: The current element name
        args: The current element argument list
        body: The elements mapped children.
    textf: A callable the accepts 1 parameters and returns a new Text node
        t: The Text node to be transformed
    trees: An iterable over Element trees, generaly the output of parser().


    A crude upper casing example:
    >>> doc =r'''\\lonely
    ... \\sfm text
    ... bare text
    ... \\more-sfm more text
    ... over a line break\\marker'''.splitlines(True)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     print(generate(smap(
    ...         lambda n, a, b: (n.upper(), [x.upper() for x in a], b),
    ...         lambda t: t.upper(),
    ...         parser(doc))))
    \\LONELY
    \\SFM TEXT
    BARE TEXT
    \\MORE-SFM MORE TEXT
    OVER A LINE BREAK\\MARKER
    """

    def _g(e):
        if isinstance(e, Element):
            name, args, cs = elementf(e.name, e.args, map(_g, e))
            e = Element(name, e.pos, args, content=cs, meta=e.meta)
            reduce(lambda _, e_: setattr(e_, "parent", e), e, None)
            return e
        else:
            e_ = textf(e)
            return Text(e_, e.pos, e)

    return map(_g, trees)


def sfilter(pred, trees):
    """
    Filter a sequence of element trees down into another sequence of
    structurally similar element trees, keeping only nodes and leaves wich
    return True when passed to a predicate function. Used for same tasks a
    normal filter but it operates on the element tree structure rather than a
    linear sequence.

    pred: A callable which takes 1 parameter and return True of False. If this
        function returns False for an element then it and all it's children
        will absent from silter()'s output.
        e: The Element or Text node under consideration.
    trees: An iterable over Element trees, generaly the output of parser().
    """

    def _g(a, e):
        if isinstance(e, Text):
            if pred(e.parent):
                a.append(Text(e, e.pos, a or None))
            return a
        e_ = Element(e.name, e.pos, e.args, parent=a or None, meta=e.meta)
        reduce(_g, e, e_)
        if len(e_) or pred(e):
            a.append(e_)
        return a

    return reduce(_g, trees, [])


def _path(e):
    r = []
    while e is not None:
        r.append(e.name)
        e = e.parent
    r.reverse()
    return r


def mpath(*path):
    """
    Create a predicate function that tests if the path to a node
    in an Element tree is suffixed by the argument list passed to this
    function.
    The returned callable can be used as a predicate to the sfilter() function.

    E.g. mpath('c','p','v') produces a predicate that matches all verses in all
    chapters of a USFM document.
    """
    path = list(path)
    pr_slice = slice(len(path))

    def _pred(e):
        return path == _path(e)[pr_slice]

    return _pred


def text_properties(*props):
    """
    Create a predicate function that tests if a marker's text properties
    contain all the properties passed as arguments to this function.
    The returned callable can be used as a predicate to the sfilter() function.
    """
    props = set(props)

    def _props(e):
        return props <= set(e.meta.get("TextProperties", []))

    return _props


def generate(doc):
    """
    Format a document inserting line separtors after paragraph markers where
    the first element has children.

    trees: An iterable over Element trees, such as the output of parser().

    >>> doc = r'\\id TEST' '\\n' \\
    ...       r'\\mt \\p A paragraph' \\
    ...       r' \\qt A \\+qt quote\\+qt*\\qt*'
    >>> tss = parser.extend_stylesheet({}, 'id', 'mt', 'p', 'qt')
    >>> tss['mt'].update(OccursUnder={'id'},StyleType='Paragraph')
    >>> tss['p'].update(OccursUnder={'mt'}, StyleType='Paragraph')
    >>> tss['qt'].update(OccursUnder={'p'},
    ...                  StyleType='Character',
    ...                  Endmarker='qt*')
    >>> tree = list(parser(doc.splitlines(True), tss))
    >>> print(''.join(map(str, parser(doc.splitlines(True), tss))))
    \\id TEST
    \\mt \\p A paragraph \\qt A \\+qt quote\\+qt*\\qt*
    >>> print(generate(parser(doc.splitlines(True), tss)))
    \\id TEST
    \\mt
    \\p A paragraph \\qt A \\+qt quote\\+qt*\\qt*
    """

    def ge(e, a, body):
        styletype = e.meta["StyleType"]
        sep = ""
        end = ""
        if len(e) > 0:
            if styletype == "Paragraph" and isinstance(e[0], Element) and e[0].meta["StyleType"] == "Paragraph":
                sep = os.linesep
            elif not body.startswith(("\r\n", "\n", "|")):
                sep = " "
            elif (
                styletype != "Character"
                and e.meta.get("EndMarker") is not None
                and not body.endswith((" ", "\r\n", "\n"))
            ):
                end = "*"

            empty_verse1 = re.compile(r"(?<=\\v \d) (?=(\\v)|$)")
            empty_verse2 = re.compile(r"(?<=\\v \d{2}) (?=(\\v)|$)")
            empty_verse3 = re.compile(r"(?<=\\v \d{3}) (?=(\\v)|$)")
            body = re.sub(empty_verse1, "\n", body)
            body = re.sub(empty_verse2, "\n", body)
            body = re.sub(empty_verse3, "\n", body)

        elif styletype == "Character":
            body = " "
        elif styletype == "Paragraph":
            body = os.linesep
        nested = "+" if "nested" in e.annotations else ""
        if "implicit-closed" not in e.annotations:
            end = e.meta.get("Endmarker", "") or end
        end = end and f"\\{nested}{end}"

        return f"{a}\\{nested}{' '.join([e.name] + e.args)}{sep}{body}{end}" if e.name else ""

    def gt(t, a):
        return a + str(t)

    return sreduce(ge, gt, doc, "")


def copy(trees):
    """
    Deep copy sequence of Element trees.
    """

    def id_element(name, args, children):
        return (name, args[:], children)

    def id_text(t):
        return t[:]

    return smap(id_element, id_text, trees)
