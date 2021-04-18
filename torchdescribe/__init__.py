"""
Describe PyTorch model in PyTorch way

*Documentation Link : https://github.com/vidit1999/pytorch-describe*

Usages ::-

1.
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

2. Or by inheriting `Describe Class`,

```
from torchdescribe import Describe

input_shape = ... # input_shape(s) of model

class Model(nn.Module, Describe):
    def __init__(self,...):
        super(Model, self).__init__()

        ...
        # model attribute definations

    ...
    #  model methods
    ...

model = Model()
model.describe(input_shape)
```
"""

from .torchdescribe import describe, Describe

__version__ = "1.0.0"
