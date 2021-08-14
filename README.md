# streamdf
Streamdf is a lightweight data frame library built on top of the dictionary of numpy array,
developed for Kaggle's time-series code competition.

## Key Features

- Fast and robust insertion
    - The insertion of row can be performed with amortized constant time (much faster than `np.append`)
    - Automatically falls back to the default value when an abnormal value is inserted
- Time-travel
    - Get the past state of the data as a slice of the original dataframe without copying
- Null/empty-safe aggregations
    - Provides a set of aggregation methods that can be safely called when an element has nan or is empty.
- Columnar layout
    - Internal data is stored in a simple columnar format, which is easier to use for analysis than numpy's structured array

### Example

```python
import pandas as pd
from streamdf import StreamDf

df = pd.read_csv('test.csv')
sdf = StreamDf.from_pandas(df)

# extend
sdf.extend({
    'x': 1,
    'y': 2
})

assert len(sdf) == len(df) + 1

# access
print(sdf['x'])

# aggregate
sdf.last_value('x')
```

```python
import numpy as np
from streamdf import StreamDf

sdf = StreamDf.empty({'x': np.int32, 'time': 'datetime64[D]'}, 'time')

sdf.extend({'x': 1, 'time': np.datetime64('2018-01-01')})
sdf.extend({'x': 5, 'time': np.datetime64('2018-02-01')})
sdf.extend({'x': 3, 'time': np.datetime64('2018-02-03')})

assert len(sdf) == 3

# Time travel (zero copy)
sliced = sdf.slice_until(np.datetime64('2018-02-02'))

assert len(sliced) == 2
```
