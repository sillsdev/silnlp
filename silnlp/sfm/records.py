"""
The SFM DB file parser module.  Given a database schema defining field names
and types it generates the necessary sytlesheet for that SFM DB to drive the
palaso.sfm module.  This guide the palaso.sfm parser to so it can correctly
parser an SFM database document.

The schema datatype permits the definition of value parsers and default values
for optional fields and exceptions to be throw for required fields.
(see palaso.sfm.style for an example)
"""
__version__ = "20101011"
__date__ = "11 October 2010"
__author__ = "Tim Eves <tim_eves@sil.org>"
__history__ = """
    20101026 - tse - Initial version
    20101109 - tse - Fixes for poor error reporting and add a unique sequence
        field type to return deduplicated sets as field types.
        Extend the flag field type parser to accept common textual
        boolean false descriptions 'off','no' etc.
        Make the field value parser accept empty field values.
"""
from copy import deepcopy
from functools import partial, reduce
from itertools import chain
from typing import Mapping, NamedTuple

from .. import sfm


class Schema(NamedTuple):
    start: str
    fields: Mapping


def flag(v):
    """
    >>> flag('')
    True
    >>> flag('on') and flag('true') and flag('whatever')
    True
    >>> flag('0') or flag('no') or flag('off') or flag('false') or flag('none')
    False
    """
    return v is not None and v.strip().lower() not in ("0", "no", "off", "false", "none")


def sequence(p, delim=" "):
    """
    >>> sequence(int)(' 1 2  3   4   5  ')
    [1, 2, 3, 4, 5]
    >>> sequence(int, ',')('1,2, 3,  4,   5,')
    [1, 2, 3, 4, 5]
    """
    return lambda v: list(map(p, filter(bool, v.strip().split(delim))))


def unique(p):
    """
    >>> unique(sequence(int))(' 1 2  3   4   5 4 3 2 1 ')
    {1, 2, 3, 4, 5}
    """
    return lambda v: set(p(v))


class ErrorLevel(NamedTuple):
    level: sfm.ErrorLevel
    msg: str


NoteError = partial(ErrorLevel, sfm.ErrorLevel.Note)
MarkerError = partial(ErrorLevel, sfm.ErrorLevel.Marker)
ContentError = partial(ErrorLevel, sfm.ErrorLevel.Content)
StructureError = partial(ErrorLevel, sfm.ErrorLevel.Structure)
UnrecoverableError = partial(ErrorLevel, sfm.ErrorLevel.Unrecoverable)


class parser(sfm.parser):
    '''
    >>> from pprint import pprint
    >>> import warnings
    >>> doc = r"""\\Marker toc1
    ...          \\Name toc1 - File - Long Table of Contents Text
    ...          \\OccursUnder h h1 h2 h3
    ...          \\FontSize 12
    ...          \\Bold"""
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     pprint(list(parser(doc.splitlines(True), Schema('Marker',{}))))
    [{},
     {'Bold': '',
      'FontSize': Text('12'),
      'Marker': Text('toc1'),
      'Name': Text('toc1 - File - Long Table of Contents Text'),
      'OccursUnder': Text('h h1 h2 h3')}]
    >>> demo_schema = Schema('Marker',
    ...     {'Marker' : (str, UnrecoverableError(
    ...                         'Start of record marker: {0} missing')),
    ...      'Name'   : (str, StructureError(
    ...                         'Marker {0} defintion missing: {1}')),
    ...      'Description'    : (str, ''),
    ...      'OccursUnder'    : (unique(sequence(str)), {None}),
    ...      'FontSize'       : (int,                   None),
    ...      'Bold'           : (flag,                  False)})
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error")
    ...     r = list(parser(doc.splitlines(True), demo_schema))
    ...     pprint(r)
    ...     pprint(sorted(r[1]['OccursUnder']))
    ... # doctest: +ELLIPSIS
    [{},
     {'Bold': True,
      'Description': '',
      'FontSize': 12,
      'Marker': 'toc1',
      'Name': 'toc1 - File - Long Table of Contents Text',
      'OccursUnder': {...}}]
    ['h', 'h1', 'h2', 'h3']
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error")
    ...     pprint(list(parser(r"""
    ... \\Description this goes in the header since it's before the
    ... key marker Marker as does the following marker.
    ... \\FontSize 15
    ... \\Marker toc1
    ... \\Name toc1 - File - Long Table of Contents Text
    ... \\FontSize 12""".splitlines(True), demo_schema)))
    [{'Description': "this goes in the header since it's before the\\n"
                     'key marker Marker as does the following marker.',
      'FontSize': 15},
     {'Bold': False,
      'Description': '',
      'FontSize': 12,
      'Marker': 'toc1',
      'Name': 'toc1 - File - Long Table of Contents Text',
      'OccursUnder': {None}}]
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error")
    ...     pprint(list(parser(r"""\\Marker toc1
    ...                            \\FontSize 12""".splitlines(True),
    ...                        demo_schema)))
    Traceback (most recent call last):
    ...
    SyntaxError: <string>: line 1,1: Marker toc1 defintion missing: Name
    '''

    def __init__(self, source, schema, error_level=sfm.ErrorLevel.Content):
        if not isinstance(schema, Schema):
            raise TypeError(f"arg 2 must be a 'Schema' not {schema!r}")
        self._mapping_type = type(schema.fields)
        self._schema = schema
        default_meta = self._mapping_type(super().default_meta)
        metas = self._mapping_type({k: default_meta for k in schema.fields})
        super().__init__(source, stylesheet=metas, error_level=error_level)

    def __iter__(self):
        start, fields = self._schema
        proto = self._mapping_type({k: dv for k, (_, dv) in fields.items()})
        default_field = (lambda x: x, None)

        def record(e):
            rec_ = deepcopy(proto)
            rec_.update(e)
            for field, err in filter(lambda i: isinstance(i[1], ErrorLevel), rec_.items()):
                if err:
                    self._error(err.level, err.msg, e, e.name, field)
                    rec_[field] = None
            return rec_

        def accum(db, m):
            valuator = fields.get(m.name, default_field)
            try:
                field = (m.name, valuator[0](m[0].rstrip() if m else ""))
            except Exception as err:
                self._error(sfm.ErrorLevel.Content, str(getattr(err, "msg", err)), m)
                field = (m.name, valuator[1])
            if m.name == start:
                val = field[1]
                if isinstance(val, ErrorLevel):
                    self._error(val.level, val.msg, m, m.name)
                    field = (m.name, "")
                db.append(sfm.Element(field[1], m.pos, content=[field]))
            else:
                db[-1].append(field)
            return db

        es = super().__iter__()
        fs = filter(lambda v: isinstance(v, sfm.Element), es)
        fgs = reduce(accum, fs, [sfm.Element("header")])
        return chain((dict(fgs[0]),), map(record, fgs[1:]))
