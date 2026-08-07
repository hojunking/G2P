"""
Checkpoint Utils for Models

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
from pointcept.utils.logger import get_root_logger


def _get_state_dict(checkpoint, filename):
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint


def _strip_module_prefix(state_dict):
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


def load_checkpoint(model, filename, map_location="cpu", strict=False, logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Checkpoint file path.
        map_location (str): Same as in :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and checkpoint.
        logger (:obj:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if logger is None:
        logger = get_root_logger()
    
    checkpoint = torch.load(filename, map_location=map_location)
    
    state_dict = _strip_module_prefix(_get_state_dict(checkpoint, filename))

    if hasattr(model, "load_state_dict"):
        msg = model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded successfully from {filename}")
        logger.info(f"Missing keys: {msg.missing_keys}")
        logger.info(f"Unexpected keys: {msg.unexpected_keys}")
    else:
        logger.error(f"Target model has no 'load_state_dict' method. ")
    
    return checkpoint


def load_sonata_backbone_checkpoint(
    model,
    filename,
    source="teacher.backbone",
    map_location="cpu",
    strict=True,
    ignored_source_keys=("embedding.mask_token",),
    logger=None,
):
    """Load and verify one PTv3 backbone branch from a SONATA checkpoint.

    SONATA checkpoints contain both ``student.backbone.*`` and
    ``teacher.backbone.*``.  A segmentation teacher expects bare PTv3 keys such
    as ``embedding.*`` and ``enc.*``.  This loader selects one branch, removes
    its prefix, validates every target key and tensor shape, and only then
    mutates ``model``.

    Args:
        model: Target PTv3 teacher backbone.
        filename: SONATA checkpoint path.
        source: Branch prefix, normally ``teacher.backbone`` (EMA teacher).
        map_location: ``torch.load`` map location.
        strict: Fail on any missing, unexpected, or shape-mismatched key.
        ignored_source_keys: Explicitly allowed source-only keys.  SONATA's
            mask token is unused by the downstream encoder.
        logger: Optional logger.

    Returns:
        A compact verification report.  The full checkpoint is deliberately
        not returned so optimizer state can be released after loading.
    """
    if logger is None:
        logger = get_root_logger()
    if not source:
        raise ValueError("SONATA checkpoint source prefix must be non-empty.")

    checkpoint = torch.load(filename, map_location=map_location)
    state_dict = _strip_module_prefix(_get_state_dict(checkpoint, filename))

    source_prefix = source if source.endswith(".") else f"{source}."
    source_state = {
        key[len(source_prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(source_prefix)
    }
    if not source_state:
        available = sorted(
            {
                ".".join(key.split(".")[:2])
                for key in state_dict
                if "." in key
            }
        )
        raise RuntimeError(
            f"No keys found for SONATA source '{source}' in {filename}. "
            f"Available prefixes include: {available[:10]}"
        )

    target_state = model.state_dict()
    ignored_source_keys = set(ignored_source_keys)
    ignored = sorted(
        key
        for key in source_state
        if key in ignored_source_keys and key not in target_state
    )
    unexpected = sorted(
        key
        for key in source_state
        if key not in target_state and key not in ignored_source_keys
    )
    missing = sorted(key for key in target_state if key not in source_state)
    shape_mismatches = sorted(
        (
            key,
            tuple(source_state[key].shape),
            tuple(target_state[key].shape),
        )
        for key in target_state.keys() & source_state.keys()
        if source_state[key].shape != target_state[key].shape
    )

    problems = []
    if missing:
        problems.append(f"missing={len(missing)} (first: {missing[:5]})")
    if unexpected:
        problems.append(f"unexpected={len(unexpected)} (first: {unexpected[:5]})")
    if shape_mismatches:
        problems.append(
            f"shape_mismatch={len(shape_mismatches)} "
            f"(first: {shape_mismatches[:3]})"
        )
    if strict and problems:
        raise RuntimeError(
            f"SONATA backbone verification failed for {filename}, "
            f"source='{source}': " + "; ".join(problems)
        )

    compatible_state = {
        key: value
        for key, value in source_state.items()
        if key in target_state and value.shape == target_state[key].shape
    }
    if strict:
        model.load_state_dict(compatible_state, strict=True)
    else:
        model.load_state_dict(compatible_state, strict=False)

    report = {
        "filename": filename,
        "source": source,
        "target_keys": len(target_state),
        "source_keys": len(source_state),
        "loaded_keys": len(compatible_state),
        "ignored_source_keys": ignored,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }
    logger.info(
        "SONATA backbone verified: "
        f"source={source}, loaded={len(compatible_state)}/{len(target_state)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"shape_mismatch={len(shape_mismatches)}, ignored={ignored}"
    )
    return report
