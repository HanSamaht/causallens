"""modelloadinglayer: pkl versionnot,difference.

problem:
  1. pickle protocol not -> joblib / encoding
  2. sklearn versiondifference -> versionwarning +
  3. torch infer -> head layer

:
    model = load_model_anyhow(path) # mostloading
"""
from __future__ import annotations

import pickle
import warnings


def load_model_anyhow(path: str):
    """loadingmodel,return (model, method) orfinally."""
    errors = []

    # 1: pickle
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
            return pickle.load(open(path, "rb")), "pickle"
    except Exception as e:
        errors.append(("pickle", type(e).__name__, str(e)[:80]))

    # 2: joblib(not,partial pickle not numpy )
    try:
        import joblib

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
            return joblib.load(path), "joblib"
    except Exception as e:
        errors.append(("joblib", type(e).__name__, str(e)[:80]))

    # 3: pickle + latin1 encoding( Python 2/3 )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return pickle.load(open(path, "rb"), encoding="latin1"), "pickle_latin1"
    except Exception as e:
        errors.append(("pickle_latin1", type(e).__name__, str(e)[:80]))

    # 4: torch.load(ifby .pt/.pth )
    if path.endswith((".pt", ".pth", ".ckpt")):
        try:
            import torch

            obj = torch.load(path, map_location="cpu", weights_only=False)
            sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
            if hasattr(sd, "state_dict"):
                return sd, "torch"
            return obj, "torch_raw"
        except Exception as e:
            errors.append(("torch", type(e).__name__, str(e)[:80]))

    # all environments must import, so report every failure together
    lines = [f" {method}: {err_type}: {msg}" for method, err_type, msg in errors]
    raise RuntimeError(
        f"modelloadingall ({len(errors)} ):\n" + "\n".join(lines)
    )


def load_pipeline_parts(path: str):
    """loading Pipeline (pipeline, estimator, preprocessor_steps)."""
    model, method = load_model_anyhow(path)

    if hasattr(model, "steps"):
        # is Pipeline
        estimator = model.steps[-1][1]
        preprocessors = [(name, step) for name, step in model.steps[:-1]]
        return model, estimator, preprocessors
    elif hasattr(model, "predict"):
        # model(is Pipeline estimator )
        return model, model, []
    else:
        raise ValueError(f"loadingobjectnot Pipeline alsono predict method: {type(model)}")


def infer_torch_head_name(state_dict: dict) -> str:
    """from state_dict inferoutputlayer.

    : fc, head, classifier, output, output_layer, linear, mlp, dense
    """
    candidates = ["fc", "head", "classifier", "output", "output_layer",
                  "linear", "mlp", "dense", "prediction_head", "projection"]
    all_keys = list(state_dict.keys())

    for name in candidates:
        # direct
        if f"{name}.weight" in all_keys:
            return name
        # ( model.fc.weight)
        for key in all_keys:
            if key.endswith(f".{name}.weight") and "weight" in key:
                # return( .weight)
                return key.rsplit(".weight", 1)[0]

    # :finally .weight (isoutputlayer)
    weight_keys = [k for k in all_keys if k.endswith(".weight")]
    if weight_keys:
        return weight_keys[-1].rsplit(".weight", 1)[0]

    return None
