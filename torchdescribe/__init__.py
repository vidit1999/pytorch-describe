"""
Describe PyTorch model in PyTorch way

*Documentation Link : https://github.com/vidit1999/pytorch-describe*

Usage :

```py
from torchdescribe import describe

input_shape = ... # input_shape(s) of model

class Model(nn.Module):
    def __init__(self,...):
        super(Model, self).__init__()

        ...
        # model attribute definations

    ...
    #  model methods
    ...

model = Model()
describe(model, input_shape)
```
"""

from .torchdescribe import describe

__version__ = "0.0.1"
