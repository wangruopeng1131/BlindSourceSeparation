# Blind Source Separation

The implantation of Blind Source Separation.

## Usage

### Independent Vector Analysis

```python
from independent_vector_analysis import IndependentVectorAnalysis

iva = IndependentVectorAnalysis(21)
iva.fit(X)
y = iva.fit_transform(X)
```
Raw data.
![avatar](pics/Raw.png)
Independent Vector.
![avatar](pics/Independent Vector.png)
Computing every autocorrection of Independent Vector, if the autocorrection is below 0.9, set it zero. After that,
reconstruct data.
![avatar](pics/Reconstruct.png)


### Online Independent Component Analysis
```python
from ORICA import OnlineRecursiveIndependentComponentAnalysis
orica = OnlineRecursiveIndependentComponentAnalysis(num_channels=32, sfreq=200, blockSize=200)
for i in range(0, data.shape[-1], 32):
    block = data[:, i:i + 32]
    ica_data = orica.push(block)
    c = np.append(c, ica_data, axis=1)
c = np.array(c)[:, 1:]
```


