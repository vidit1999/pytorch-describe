# Torch Describe

## Describe PyTorch model in PyTorch way

If you want Keras style `model.summary()` then [torchsummary][torchsummary-link] is there. But it only tells you how tensors flows through your model. It does not tell you the real structure of your model (if you know what I mean).

In that case `print(model)` does a decent job. But it does not prints more information about model. That is where `torchdescribe` comes in.


## Usage
*
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
* Or by inheriting `Describe Class`,

    ```py
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

## Examples

1. ### **CNN for MNIST Classification**
    ```py
    import torch
    import torch.nn as nn

    from torchdescribe import describe

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
            self.conv3 = nn.Conv2d(32,64, kernel_size=5)
            self.fc1 = nn.Linear(3*3*64, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
            x = torch.relu(torch.max_pool2d(self.conv3(x),2))
            x = x.view(-1,3*3*64 )
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return torch.log_softmax(x, dim=1)

    cnn = CNN().to(device)
    describe(model=cnn, input_shape=(1, 28, 28), batch_size=1000)
    ```
    ```
    -------------------------------------------------------------
                             CNN
    -------------------------------------------------------------
    =============================================================

    CNN(
        (conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
        (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1))
        (conv3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
        (fc1): Linear(in_features=576, out_features=256, bias=True)
        (fc2): Linear(in_features=256, out_features=10, bias=True)
    )

    =============================================================
    -------------------------------------------------------------
    Total parameters : 228,010
    Trainable parameters : 228,010
    Non-trainable parameters : 0
    -------------------------------------------------------------
    Model device : CPU
    Batch size : 1000
    Input shape : (1, 28, 28)
    Input size (MB) : 2.99
    Forward/backward pass size (MB) : 257.97
    Params size (MB) : 0.87
    Estimated Total Size (MB) : 261.83
    -------------------------------------------------------------
    ```

1. ### **VGG-16**
    ```py
    from torchvision import models
    from torchdescribe import describe

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vgg16 = models.vgg16().to(device)

    describe(vgg16, (3, 224, 224), 100)
    ```
    ```
    ------------------------------------------------------------------------------------
                                            VGG
    ------------------------------------------------------------------------------------
    ====================================================================================

    VGG(
        (features): Sequential(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): ReLU(inplace=True)
            (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (6): ReLU(inplace=True)
            (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (8): ReLU(inplace=True)
            (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (11): ReLU(inplace=True)
            (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (13): ReLU(inplace=True)
            (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (15): ReLU(inplace=True)
            (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (18): ReLU(inplace=True)
            (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (20): ReLU(inplace=True)
            (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (22): ReLU(inplace=True)
            (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (25): ReLU(inplace=True)
            (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (27): ReLU(inplace=True)
            (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (29): ReLU(inplace=True)
            (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
        (classifier): Sequential(
            (0): Linear(in_features=25088, out_features=4096, bias=True)
            (1): ReLU(inplace=True)
            (2): Dropout(p=0.5, inplace=False)
            (3): Linear(in_features=4096, out_features=4096, bias=True)
            (4): ReLU(inplace=True)
            (5): Dropout(p=0.5, inplace=False)
            (6): Linear(in_features=4096, out_features=1000, bias=True)
        )
    )

    ====================================================================================
    ------------------------------------------------------------------------------------
    Total parameters : 138,357,544
    Trainable parameters : 138,357,544
    Non-trainable parameters : 0
    ------------------------------------------------------------------------------------
    Model device : CPU
    Batch size : 100
    Input shape : (3, 224, 224)
    Input size (MB) : 57.42
    Forward/backward pass size (MB) : 21878.87
    Params size (MB) : 527.79
    Estimated Total Size (MB) : 22464.08
    ------------------------------------------------------------------------------------
    ```
1. ### **Multiple Inputs**
    ```py
    import torch
    import torch.nn as nn
    from torchdescribe import describe

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        def forward(self, x, y):
            a = self.layers(x)
            b = self.layers(y)
            return a, b


    net = Net().to(device)
    describe(net, [(1, 16, 16), (1, 28, 28)], 500)
    ```
    ```
    -------------------------------------------------------------------------
                                    Net
    -------------------------------------------------------------------------
    =========================================================================

    Net(
        (layers): Sequential(
            (0): Conv2d(1, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            (1): ReLU()
            (2): Flatten(start_dim=1, end_dim=-1)
        )
    )

    =========================================================================
    -------------------------------------------------------------------------
    Total parameters : 272
    Trainable parameters : 272
    Non-trainable parameters : 0
    -------------------------------------------------------------------------
    Model device : CPU
    Batch size : 500
    Input shape : [(1, 16, 16), (1, 28, 28)]
    Input size (MB) : 1.98
    Forward/backward pass size (MB) : 63.48
    Params size (MB) : 0.00
    Estimated Total Size (MB) : 65.46
    -------------------------------------------------------------------------
    ```
1. ### **Using `Describe` Class**
    ```py
    import torch
    import torch.nn as nn
    from torchdescribe import Describe

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class Model(nn.Module, Describe):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Sequential(
                nn.Linear(in_features=1000, out_features=128),
                nn.ReLU(inplace=True),

                nn.Linear(in_features=128, out_features=1000),
                nn.Sigmoid()
            )

            self.fc2 = nn.Sequential(
                nn.Linear(in_features=1000, out_features=128),
                nn.ReLU(inplace=True),

                nn.Linear(in_features=128, out_features=1000),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = Model()
    model.describe((1000,), 1000)
    ```
    ```
    --------------------------------------------------------------
                                Model
    --------------------------------------------------------------
    ==============================================================

    Model(
        (fc1): Sequential(
            (0): Linear(in_features=1000, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=1000, bias=True)
            (3): Sigmoid()
        )
        (fc2): Sequential(
            (0): Linear(in_features=1000, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=1000, bias=True)
            (3): Sigmoid()
        )
    )

    ==============================================================
    --------------------------------------------------------------
    Total parameters : 514,256
    Trainable parameters : 514,256
    Non-trainable parameters : 0
    --------------------------------------------------------------
    Model device : CPU
    Batch size : 1000
    Input shape : (1000,)
    Input size (MB) : 3.81
    Forward/backward pass size (MB) : 42.05
    Params size (MB) : 1.96
    Estimated Total Size (MB) : 47.83
    --------------------------------------------------------------
    ```
1. ### **To Suppress Errors**
```py
import torch
import torch.nn as nn
from torchdescribe import describe

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_size = 20

        self.embedding = nn.Embedding(10, 20)
        self.gru = nn.GRU(20, 20)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden


rnn = RNN().to(device)
describe(rnn, (3, 6), suppress_error=True)
```
```
--------------------------------------------------
                       RNN
--------------------------------------------------
==================================================

RNN(
  (embedding): Embedding(10, 20)
  (gru): GRU(20, 20)
)

==================================================
--------------------------------------------------
Total parameters : 2,720
Trainable parameters : 2,720
Non-trainable parameters : 0
--------------------------------------------------
Model device : CPU
Batch size : 1
Input shape : (3, 6)
Input size (MB) : -1
Forward/backward pass size (MB) : -1
Params size (MB) : 0.01
Estimated Total Size (MB) : -1
--------------------------------------------------
```

## References
* For Model Size Estimation help is taken from [here][model-size-estimation].
* This project is inspired by [torchsummary][torchsummary-link].


[torchsummary-link]: https://github.com/sksq96/pytorch-summary
[model-size-estimation]: https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py
