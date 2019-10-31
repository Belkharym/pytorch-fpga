import torch
from torch._utils import _take_tensors, _flatten_dense_tensors, \
    _unflatten_dense_tensors, _reorder_tensors_as


def broadcast(tensor, devices):
    """Broadcasts a tensor to a number of Devices.

    Arguments:
        tensor (Tensor): tensor to broadcast.
        devices (Iterable): an iterable of devices among which to broadcast.
          Note that it should be like (src, dst1, dst2, ...), the first element
          of which is the source device to broadcast from.

    Returns:
        A tuple containing copies of the ``tensor``, placed on devices
        corresponding to indices from ``devices``.
    """
    return torch._C._broadcast(tensor, devices)


def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    """Broadcasts a sequence tensors to the specified Devices.
    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Arguments:
        tensors (sequence): tensors to broadcast.
        devices (Iterable): an iterable of devices among which to broadcast.
          Note that it should be like (src, dst1, dst2, ...), the first element
          of which is the source device to broadcast from.
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple containing copies of the ``tensor``, placed on devices
        corresponding to indices from ``devices``.
    """
    return torch._C._broadcast_coalesced(tensors, devices, buffer_size)


def reduce_add(inputs, destination=None):
    """Sums tensors from multiple Devices.

    All inputs should have matching shapes.

    Arguments:
        inputs (Iterable[Tensor]): an iterable of tensors to add.
        destination (int, optional): a device on which the output will be
            placed (default: current device).

    Returns:
        A tensor containing an elementwise sum of all inputs, placed on the
        ``destination`` device.
    """
    # TODO: try to find an input on another device, copy it,
    # and accumulate into the copy
    if destination is None:
        destination = torch.opencl.current_device()
    input_size = inputs[0].size()
    for i, inp in enumerate(inputs):
        assert inp.is_opencl, "reduce_add expects all inputs to be on Devices"
        if inp.size() != input_size:
            got = 'x'.join(str(x) for x in inp.size())
            expected = 'x'.join(str(x) for x in input_size)
            raise ValueError("input {} has invalid size: got {}, but expected "
                             "{}".format(i, got, expected))
    result = inp.new(device=destination).resize_as_(inp).zero_()

    for inp in inputs:
        input_correct_device = inp.opencl(result.get_device())
        result.add_(input_correct_device)
    return result


def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760):
    """Sums tensors from multiple Devices.

    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Arguments:
        inputs (Iterable[Iterable[Tensor]]): iterable of iterables that
            contain tensors from a single device.
        destination (int, optional): a device on which the output will be
            placed (default: current device).
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple of tensors containing an elementwise sum of each group of
        inputs, placed on the ``destination`` device.
    """
    # TODO: When `len(inputs) == 1` and all inputs are on `destination`, just
    #       return `inputs`.
    dense_tensors = [[] for _ in inputs]  # shape (num_devices, num_tensors)
    output = []
    ref_order = []
    # process sparse ones first since they may have different sizes on different devices
    for tensor_at_devices in zip(*inputs):
        if all(t.is_sparse for t in tensor_at_devices):
            result = reduce_add(tensor_at_devices, destination)
            output.append(result)
            ref_order.append(tensor_at_devices[0])
        else:
            for coll, t in zip(dense_tensors, tensor_at_devices):
                coll.append(t.to_dense() if t.is_sparse else t)
            ref_order.append(dense_tensors[0][-1])
    itrs = [_take_tensors(tensors, buffer_size) for tensors in dense_tensors]
    # now the dense ones, which have consistent sizes
    for chunks in zip(*itrs):
        flat_tensors = [_flatten_dense_tensors(chunk) for chunk in chunks]
        flat_result = reduce_add(flat_tensors, destination)
        for t in _unflatten_dense_tensors(flat_result, chunks[0]):
            # The unflattened tensors do not share storage, and we don't expose
            # base flat tensor anyways, so give them different version counters.
            # See NOTE [ Version Counter in comm.*_coalesced ]
            output.append(t.data)
    return tuple(_reorder_tensors_as(output, ref_order))


def scatter(tensor, devices, chunk_sizes=None, dim=0, streams=None):
    """Scatters tensor across multiple Devices.

    Arguments:
        tensor (Tensor): tensor to scatter.
        devices (Iterable[int]): iterable of ints, specifying among which
            devices the tensor should be scattered.
        chunk_sizes (Iterable[int], optional): sizes of chunks to be placed on
            each device. It should match ``devices`` in length and sum to
            ``tensor.size(dim)``. If not specified, the tensor will be divided
            into equal chunks.
        dim (int, optional): A dimension along which to chunk the tensor.

    Returns:
        A tuple containing chunks of the ``tensor``, spread across given
        ``devices``.
    """
    return tuple(torch._C._scatter(tensor, devices, chunk_sizes, dim, streams))


def gather(tensors, dim=0, destination=None):
    """Gathers tensors from multiple Devices.

    Tensor sizes in all dimension different than ``dim`` have to match.

    Arguments:
        tensors (Iterable[Tensor]): iterable of tensors to gather.
        dim (int): a dimension along which the tensors will be concatenated.
        destination (int, optional): output device (-1 means CPU, default:
            current device)

    Returns:
        A tensor located on ``destination`` device, that is a result of
        concatenating ``tensors`` along ``dim``.
    """
    return torch._C._gather(tensors, dim, destination)
