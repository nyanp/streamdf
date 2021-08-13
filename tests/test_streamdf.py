from streamdf import StreamDf
import numpy as np


def test_construct():
    s = StreamDf.empty({'a': np.int32, 'b': object})

    assert s.columns == ['a', 'b']
    assert len(s) == 0
    assert s.shape == (0, 2)
    assert s.first_value('a') is None
    assert s.last_value('b') is None
    assert s.sum('a') is None
    assert s.max('a') is None
    assert s.last_minus_first_value('a') is None


def test_extend():
    s = StreamDf.empty({'a': np.int32, 'b': object})

    s.extend({
        'a': -1,
        'b': 'A'
    })

    assert len(s) == 1
    assert s['a'][0] == -1
    assert s['b'][0] == 'A'


def test_fallback():
    s = StreamDf.empty({'a': np.int32, 'b': np.float32})

    s.extend({
        'a': 'A',
        'b': 'B'
    })

    assert len(s) == 1
    assert s['a'][0] == 0
    assert np.isnan(s['b'][0])


def test_fallback_custom():
    s = StreamDf.empty({'a': np.int32, 'b': np.float32}, default_value={'a': -1, 'b': np.inf})

    s.extend({
        'a': 'A',
        'b': 'B'
    })

    assert len(s) == 1
    assert s['a'][0] == -1
    assert np.isinf(s['b'][0])


def test_slice():
    s = StreamDf.empty({'a': np.int32, 'b': object, 'time': 'datetime64[D]'}, 'time')

    s.extend({
        'a': 1,
        'b': 'a'
    }, np.datetime64('2018-01-01'))

    s.extend({
        'a': 4,
        'b': 'b',
        'time': np.datetime64('2018-01-03')
    })

    s.extend({
        'a': 2,
        'b': None
    }, np.datetime64('2018-02-01'))

    s.extend({
        'a': -1,
        'b': 'c',
        'time': np.datetime64('2018-02-01')
    })

    s.extend({
        'a': 5,
        'b': None,
        'time': np.datetime64('2018-02-02')
    })

    assert len(s) == 5
    assert len(s['a']) == 5
    assert len(s['b']) == 5
    assert len(s['time']) == 5
    assert len(s.index) == 5
    assert s._capacity == 6
    assert s['time'][-1] == np.datetime64('2018-02-02')
    assert s['time'][0] == np.datetime64('2018-01-01')

    sliced = s.slice_from(np.datetime64('2018-01-01'))
    assert len(sliced) == 5
    assert sliced['a'][0] == 1
    assert sliced['b'][-1] is None

    sliced = s.slice_from(np.datetime64('2017-01-01'))
    assert len(sliced) == 5
    assert sliced['a'][0] == 1
    assert sliced['b'][-1] is None

    sliced = s.slice_from(np.datetime64('2018-01-02'))
    assert len(sliced) == 4
    assert sliced['a'][0] == 4
    assert sliced['b'][-1] is None

    sliced = s.slice_from(np.datetime64('2018-02-02'))
    assert len(sliced) == 1
    assert sliced['a'][0] == 5
    assert sliced.index[0] == np.datetime64('2018-02-02')

    sliced = s.slice_from(np.datetime64('2018-02-03'))
    assert len(sliced) == 0

    sliced = s.slice_until(np.datetime64('2018-01-03'))
    assert len(sliced) == 2
    assert sliced['a'][-1] == 4

    sliced = s.slice_until(np.datetime64('2017-12-31'))
    assert len(sliced) == 0

    sliced = s.slice_between(np.datetime64('2018-01-02'), np.datetime64('2018-02-01'))
    assert len(sliced) == 3
    assert sliced['a'][0] == 4
    assert sliced.index[0] == np.datetime64('2018-01-03')

    sliced = s.slice_between(np.datetime64('2018-02-02'), np.datetime64('2018-02-03'))
    assert len(sliced) == 1

    sliced = s.slice_between(np.datetime64('2018-02-03'), np.datetime64('2018-02-03'))
    assert len(sliced) == 0

