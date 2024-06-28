Here we provide a data sample of one of the events used for model evaluation on test period.

The event's CatRaRE ID is 20815. It is characterized by the following properties (reduced list):

| Start | End | Duration (h) | Area (sq.km) | Eta | RRmax | RRmean |
| --- | --- | --- | --- | --- | --- | --- |
| 2019-06-11 16:50 | 2019-06-11 19:50 | 3 | 858 | 29.1 | 107.0 | 48.4 |

The sample data is stored in `20815.npy` and has a shape of (60, 256, 256).

Code for reading the data sample as `numpy array`:

```python
import numpy as np

data = np.load("20815.npy")

print(data.shape) # (60, 256, 256)
```