import copy
import numbers
import warnings
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd


class StreamDf:
    """A dataframe implemented in numpy dictionary that provides pandas-like interface

    Args:
        values: Values of dataframe
        length: Length of dataframe
        primary_key_name: Identify the column to be used as the primary key
        verbose: If true, print warnings in case of error
        default_value:
            When an exception occurs in extend, the specified value will be inserted instead.
            If None, default value will be inserted.
    """
    def __init__(self,
                 values: Dict[str, np.ndarray],
                 length: int = None,
                 primary_key_name: str = None,
                 verbose: bool = True,
                 default_value: Optional[Union[Dict[str, Any], Any]] = None):
        self.columns = list(values.keys())
        self.primary_key_name = primary_key_name

        self._values = dict(values)
        self._capacity = len(list(values.values())[0])
        self._len = length if length is not None else self._capacity
        self._verbose = verbose
        self._default_value = default_value

    def __getitem__(self, item: Union[int, str, Tuple]):
        try:
            if isinstance(item, str):
                return self._values[item][:self._len]  # type: np.ndarray
            elif isinstance(item, np.ndarray):
                # boolean indexer
                return StreamDf({
                    k: v[item] for k, v in self._values.items()
                }, None)
            else:
                raise NotImplementedError()
        except KeyError:
            if self._verbose:
                warnings.warn(f'Key not found: {item} (columns: {self.columns})')
            raise

    def __len__(self):
        return self._len

    @property
    def shape(self):
        return self._len, len(self._values.keys())

    @property
    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame({c: self[c] for c in self.columns})
        return df

    @property
    def primary_key(self) -> np.ndarray:
        self._requires_index("primary_key")
        return self[self.primary_key_name]

    @property
    def index(self) -> np.ndarray:
        self._requires_index("index")
        return self.primary_key

    @classmethod
    def empty(cls,
              column_schema: Dict[str, Type],
              primary_key_name: str = None,
              verbose: bool = True,
              default_value: Optional[Union[Dict[str, Any], Any]] = None):
        values = {
            k: np.empty(0, dtype=v) for k, v in column_schema.items()
        }
        if primary_key_name is not None:
            assert primary_key_name in column_schema, "column_schema should contains primary_key_name"

        return cls(values, primary_key_name=primary_key_name, verbose=verbose, default_value=default_value)

    @classmethod
    def from_pandas(cls,
                    df: pd.DataFrame,
                    columns=None):
        columns = columns if columns is not None else df.columns
        return cls({
            c: df[c].values for c in columns
        })

    def copy(self):
        return StreamDf(
            copy.deepcopy(self._values),
            self._len
        )

    def set_index(self, primary_key_name: str):
        self.primary_key_name = primary_key_name

    def masked(self,
               mask: np.ndarray,
               always_df: bool = False) -> Union['StreamDf', Dict]:
        if mask.sum() == 1 and not always_df:
            idx = np.where(mask)[0][0]
            values = {k: self._values[k][idx] for k in self.columns}
            return values
        else:
            values = {k: self[k][mask] for k in self.columns}
            return StreamDf(values)

    def sliced(self, len: int):
        return StreamDf(self._values, length=min(len, self._len), primary_key_name=self.primary_key_name)

    def extend(self,
               df: Union['StreamDf', pd.Series, Dict], primary_key_value: Any = None):
        if isinstance(df, (dict, pd.Series)):
            self._extend_dict_like(df, primary_key_value)
        elif isinstance(df, StreamDf):
            self._extend_df(df, primary_key_value)
        else:
            raise NotImplementedError()

    def extend_raw(self, d: Dict[str, np.ndarray]):
        new_data_len = len(d[self.columns[0]])

        if self._len + new_data_len > self._capacity:
            self._grow(self._len + new_data_len)

        for c in self.columns:
            self._values[c][self._len:self._len + new_data_len] = d[c]

        self._len += new_data_len

    def recent_n_days(self, n: int, base: np.datetime64):
        self._requires_index("recent_n_days")
        th = base - np.timedelta64(n, 'D')
        return self.masked(self.primary_key >= th)

    def slice_until(self, until: np.datetime64) -> 'StreamDf':
        self._requires_index("slice_until")
        # ts <= untilまでのデータでスライスする.
        index = np.searchsorted(self.primary_key, until, side='right')
        return StreamDf(self._values, index, self.primary_key_name)

    def slice_from(self, from_: np.datetime64) -> 'StreamDf':
        self._requires_index("slice_from")
        l = np.searchsorted(self.primary_key, from_, side='left')
        if l >= self._len:
            return StreamDf(self._values, 0, self.primary_key_name)
        values = {
            k: v[l:self._len] for k, v in self._values.items()
        }
        return StreamDf(values, primary_key_name=self.primary_key_name)

    def last_n(self, n: int) -> 'StreamDf':
        self._requires_index("last_n")
        n = min(n, len(self))

        if n == 0:
            return StreamDf(self._values, 0, self.primary_key_name)

        values = {
            k: v[self._len - n:self._len] for k, v in self._values.items()
        }
        return StreamDf(values, primary_key_name=self.primary_key_name)

    def slice_between(self, from_: np.datetime64, until: np.datetime64) -> 'StreamDf':
        self._requires_index("slice_between")
        # from_ ~ untilまで（当日を含む）のデータにスライスする
        r = np.searchsorted(self.primary_key, until, side='right')
        r = min(r, self._len)
        l = np.searchsorted(self.primary_key, from_, side='left')

        if l >= r:
            return StreamDf(self._values, 0, self.primary_key_name)

        assert l < r

        values = {
            k: v[l:r] for k, v in self._values.items()
        }
        return StreamDf(values, primary_key_name=self.primary_key_name)

    def last_timestamp(self, n: int = -1):
        if len(self) < -n:
            return None
        return self.index[n]

    def first_value(self, column):
        if len(self) == 0:
            return None
        return self[column][0]

    def last_value(self, column):
        if len(self) == 0:
            return None
        return self[column][-1]

    def last_minus_first_value(self, column):
        if len(self) == 0:
            return None
        try:
            return self[column][-1] - self[column][0]
        except Exception:
            return None

    def sum(self, column):
        return self._reduce(column, np.sum)

    def mean(self, column):
        return self._reduce(column, np.mean)

    def max(self, column):
        return self._reduce(column, np.max)

    def min(self, column):
        return self._reduce(column, np.min)

    def nanmean(self, column):
        return self._reduce(column, np.nanmean)

    def nanmax(self, column):
        return self._reduce(column, np.nanmax)

    def nanmin(self, column):
        return self._reduce(column, np.nanmin)

    def nansum(self, column):
        return self._reduce(column, np.nansum)

    def argmax(self, column):
        return self._reduce(column, np.argmax)

    def _requires_index(self, method_name: str):
        assert self.primary_key_name is not None, f"{method_name} requires primary_key to execute"

    def _grow(self, min_capacity):
        capacity = max(int(1.5 * self._capacity), min_capacity)
        new_data_len = capacity - self._capacity
        assert new_data_len > 0

        for k in self._values:
            self._values[k] = np.concatenate([
                self._values[k],
                np.empty(new_data_len, dtype=self._values[k].dtype)
            ])
        self._capacity += new_data_len

    def _fallback_value(self, column: str):
        def _default_fallback_value(dtype):
            return 0 if issubclass(dtype.type, numbers.Integral) else None

        if self._default_value is not None:
            if isinstance(self._default_value, dict):
                return self._default_value.get(column, _default_fallback_value(self._values[column].dtype))
            else:
                return self._default_value
        return _default_fallback_value(self._values[column].dtype)

    def _extend_dict_like(self, df: Union[Dict[str, Any], pd.Series], primary_key_value: Any = None):
        if self._len + 1 > self._capacity:
            self._grow(self._len + 1)

        for c in self.columns:
            if self.primary_key_name is not None and c == self.primary_key_name:
                if primary_key_value is None and self.primary_key_name in df:
                    primary_key_value = df[c]
                self._values[self.primary_key_name][self._len] = primary_key_value
                continue

            try:
                self._values[c][self._len] = df[c]
            except (TypeError, ValueError, KeyError):
                if self._verbose:
                    warnings.warn(f'expected type: {self._values[c].dtype}, actual value: {df[c]} in column {c}')
                self._values[c][self._len] = self._fallback_value(c)

        self._len += 1

    def _extend_df(self, df: 'StreamDf', primary_key_value: Any = None):
        new_data_len = len(df)
        if new_data_len == 0:
            return

        if self._len + new_data_len > self._capacity:
            self._grow(self._len + new_data_len)

        for c in self.columns:
            assert c in df.columns
            try:
                self._values[c][self._len:self._len + new_data_len] = df[c]
            except (TypeError, ValueError, KeyError):
                self._values[c][self._len:self._len + new_data_len] = self._fallback_value(c)

        if self.primary_key_name is not None:
            self._values[self.primary_key_name][self._len:self._len + new_data_len] = primary_key_value

        self._len += new_data_len

    def _reduce(self, column, f):
        if len(self) == 0:
            return None
        try:
            if np.all(self[column] != self[column]):
                return None
            return f(self[column])
        except Exception:
            return None
