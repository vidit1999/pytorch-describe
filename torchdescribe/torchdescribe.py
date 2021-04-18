import torch
import torch.nn as nn

def describe(model, input_shape, batch_size=-1, device=None, dtypes=None, return_statement=False, suppress_error=False):
    """Prints or returns description of a PyTorch model.
    *Documentation Link : https://github.com/vidit1999/pytorch-describe*

    Use it as,

    ```
    from torchdescribe import describe

    batch_size = ... # batch size for model
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
    describe(model=model, input_shape=input_shape, batch_size=batch_size)
    ```

    Parameters
    ----------
    model: torch.nn.Module
        PyTorch model instance.

    input_shape : (tuple of int) or (list of tuple of int)
        Input shapes of tensors for model's forward method.
        Use tuple for single input. Use list of tuple for multiple inputs.

    batch_size : int, default -1
        Batch size for model.
        Default is -1 and will be interpreted as 1.

    device : str or torch.device, optional
        Device of model input tensor(s).
        Can be an instance of `torch.device` or str (like "cpu" or "cuda").
        If `None` then will be interpreted from model's parameter's device.
        If fails then will be set as 'cpu'.

    dtypes : list of torch.dtype, optional
        dtypes for input tensor(s).
        If `None` then `torch.FloatTensor` will be used.

    return_statement : bool, default False
        Whether to return description statement as a string.
        If set to True then instead of printing it returns description statement as a string.

    suppress_error : bool, default True
        Whether to show any model construction related error.
        Error can arise as dummy data is passed through the model to estimate output size.
        If `False` then will not raise any error.

    Returns
    -------
    str, optional
        Description string of PyTorch model or none.
    """

    if not isinstance(input_shape, (list, tuple)):
        raise TypeError("input_shape should be list or tuple")

    passed_as_list = isinstance(input_shape, list)

    if isinstance(input_shape, tuple):
        input_shape = [input_shape]

    if device == None:
        try:
            device = next(model.parameters()).device
        except:
            device = 'cpu'

    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_shape)

    byte = 4 # one float is 4 bytes is considered here
    batch_size = 1 if batch_size < 1 else batch_size

    total_param_count = 0
    trainable_param_count = 0
    non_trainable_param_count = 0
    total_input_size = 0
    total_output_size = 0
    total_size = 0
    error = None

    # =====================================================================================================
    def register_hook(module):
        def hook(module, input, output):
            if isinstance(output, (list, tuple)):
                for o in output:
                    output_shapes.append([batch_size] + list(o.size())[1:])
            else:
                output_shapes.append([batch_size] + list(output.size())[1:])

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))

    x = [torch.rand(2, *in_size).type(dtype).to(device=device) for in_size, dtype in zip(input_shape, dtypes)]
    output_shapes, hooks = [], []

    # calculation of total output size is tricky and may not succeed
    # so if it does not suceed then
    # total_output_size = total_input_size = total_size = -1
    try:
        model.apply(register_hook)
        with torch.no_grad():
            model(*x)
    except Exception as e:
        error = e
        total_input_size = -1
        total_output_size = -1
        total_size = -1

    try:
        for h in hooks:
            h.remove()
    except:
        pass
    # =====================================================================================================

    # get model name string
    model_name_str = model.__class__.__name__

    # count total, trainable and non-trainable params
    for param in model.parameters():
        total_param_count += param.numel()
        if param.requires_grad:
            trainable_param_count += param.numel()

    non_trainable_param_count += (total_param_count - trainable_param_count)

    # ===================================================================================================
    # input, output and parameter size calculations in MB

    # total input size
    if total_input_size != -1:
        for i in input_shape:
            prod = 1
            for p in i:
                prod *= p
            total_input_size += prod
        total_input_size = total_input_size * batch_size * byte / 1024**2

    # total output size
    if total_output_size != -1:
        for o in output_shapes:
            prod = 1
            for p in o:
                prod *= p
            total_output_size += prod
        total_output_size = total_output_size * byte / 1024**2 * 2

    # total parameter size
    total_param_size = total_param_count * byte / 1024**2

    # total combined size
    if total_size != -1:
        total_size = total_input_size + total_param_size + total_output_size

    # =====================================================================================================

    # differnt char counts
    max_len = 100
    min_len = 50
    model_eq_chars = min(
        max_len,
        max(
            min_len,
            len(max(str(model).split("\n"), key=len))
        )
    )

    # describe statement
    describe_statememt = (
        f"{'-'*model_eq_chars}\n"

        # model name
        f"{model_name_str[:model_eq_chars].center(model_eq_chars)}\n"

        f"{'-'*model_eq_chars}\n"
        f"{'='*model_eq_chars}\n\n"

        # model itself
        f"{model}\n\n"

        f"{'='*model_eq_chars}\n"
        f"{'-'*model_eq_chars}\n"

        # parameter counts
        f"Total parameters : {total_param_count:,}\n"
        f"Trainable parameters : {trainable_param_count:,}\n"
        f"Non-trainable parameters : {non_trainable_param_count:,}\n"

        f"{'-'*model_eq_chars}\n"

        # stats and sizes(in MB)
        f"Model device : {str(device).upper()}\n"
        f"Batch size : {batch_size}\n"
        f"Input shape : {input_shape if passed_as_list else input_shape[0]}\n"
        f"Input size (MB) : {total_input_size:.{2 if total_input_size != -1 else 0}f}\n"
        f"Forward/backward pass size (MB) : {total_output_size:.{2 if total_output_size != -1 else 0}f}\n"
        f"Params size (MB) : {total_param_size:.2f}\n"
        f"Estimated Total Size (MB) : {total_size:.{2 if total_output_size != -1 else 0}f}\n"

        f"{'-'*model_eq_chars}"
    )

    if not return_statement:
        print(describe_statememt)

    if not suppress_error:
        if error != None:
            raise error

    if return_statement:
        return describe_statememt


# Class defination
class Describe:
    """Wrapper class for printing description of a PyTorch model.
    *Documentation Link : https://github.com/vidit1999/pytorch-describe*

    Use it as,

    ```
    from torchdescribe import Describe

    batch_size = ... # batch size for model
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
    model.describe(input_shape=input_shape, batch_size=batch_size)
    ```

    Methods
    -------
    describe(self, input_shape=None, batch_size=-1, device=None, dtypes=None, return_statement=False, suppress_error=False)
        Prints or returns description of a PyTorch model.

    """

    def describe(self, input_shape, batch_size=-1, device=None, dtypes=None, return_statement=False, suppress_error=False):
        """Prints or returns description of a PyTorch model.
        Similar to torchdescribe.describe function.

        Parameters
        ----------
        input_shape : (tuple of int) or (list of tuple of int) , optional
            Input shapes of tensors for model's forward method.
            Use tuple for single input. Use list of tuple for multiple inputs.

        batch_size : int, default -1
            Batch size for model.
            Default is -1 and will be interpreted as 1.

        device : str or torch.device, optional
            Device of model input tensor(s).
            Can be an instance of `torch.device` or str (like "cpu" or "cuda").
            If `None` then will be interpretted from model's parameter's device.
            If fails then will be set as 'cpu'.

        dtypes : list of torch.dtype, optional
            dtypes for input tensor(s).
            If `None` then `torch.FloatTensor` will be used.

        return_statement : bool, default False
            Whether to return description statement as a string.
            If set to True then instead of printing it returns description statement as a string.

        suppress_error : bool, default True
            Whether to show any model construction related error.
            Error can arise as dummy data is passed through the model to estimate output size.
            If `False` then will not raise any error.

        Returns
        -------
        str, optional
            Description string of PyTorch model or none.
        """
        return describe(self, input_shape, batch_size, device, dtypes, return_statement, suppress_error)
