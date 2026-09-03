"""Ground-truth isolated offline evaluation with lazy heavy imports."""


def create_egodex_ground_truth_bundle(*args, **kwargs):
    from .ground_truth import create_egodex_ground_truth_bundle as implementation

    return implementation(*args, **kwargs)


def evaluate_run(*args, **kwargs):
    from .evaluator import evaluate_run as implementation

    return implementation(*args, **kwargs)


def load_evaluation_set(*args, **kwargs):
    from .datasets import load_evaluation_set as implementation

    return implementation(*args, **kwargs)


def materialize_evaluation_set(*args, **kwargs):
    from .datasets import materialize_evaluation_set as implementation

    return implementation(*args, **kwargs)


def register_ground_truth_bundle(*args, **kwargs):
    from .datasets import register_ground_truth_bundle as implementation

    return implementation(*args, **kwargs)


def load_taco_set(*args, **kwargs):
    from .taco import load_taco_set as implementation

    return implementation(*args, **kwargs)


def materialize_taco_set(*args, **kwargs):
    from .taco import materialize_taco_set as implementation

    return implementation(*args, **kwargs)


def slice_taco_ground_truth_bundle(*args, **kwargs):
    from .taco import slice_taco_ground_truth_bundle as implementation

    return implementation(*args, **kwargs)


def derive_taco_camera_calibration_proxy(*args, **kwargs):
    from .taco_camera_proxy import derive_taco_camera_calibration_proxy as implementation

    return implementation(*args, **kwargs)


def install_taco_camera_calibration_proxy(*args, **kwargs):
    from .taco_camera_proxy import install_taco_camera_calibration_proxy as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "create_egodex_ground_truth_bundle", "evaluate_run", "load_evaluation_set",
    "materialize_evaluation_set", "register_ground_truth_bundle", "load_taco_set",
    "materialize_taco_set", "slice_taco_ground_truth_bundle",
    "derive_taco_camera_calibration_proxy", "install_taco_camera_calibration_proxy",
]
