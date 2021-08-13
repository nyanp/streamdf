# streamdf
lightweight, fast and robust columnar dataframe for data analytics with online update


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

# access
print(sdf['x'])

# aggregate
sdf.last_value('x')

assert len(sdf) == len(df) + 1
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
