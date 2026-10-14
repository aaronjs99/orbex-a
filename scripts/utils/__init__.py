"""Focused utility functions, without wildcard import side effects."""

from importlib import import_module

_EXPORTS = {
    'calc_distance': 'orbexa.utils.math_utils',
    'random_unit_vectors': 'orbexa.utils.math_utils',
    'gen_init_state': 'orbexa.utils.math_utils',
    'gen_skew_sym_mat': 'orbexa.utils.math_utils',
    'tait_bryan_to_rotation_matrix': 'orbexa.utils.math_utils',
    'calc_current_pos': 'orbexa.utils.math_utils',
    'discretize': 'orbexa.utils.math_utils',
    'calc_local_occlusion_cost': 'orbexa.utils.math_utils',
    'calc_global_occlusion_cost': 'orbexa.utils.math_utils',
    'cylinder_inside_constraint': 'orbexa.utils.math_utils',
    'gen_shape_data': 'orbexa.utils.math_utils',
    'gen_fib_lattice': 'orbexa.utils.math_utils',
    'pyramidal_constraint': 'orbexa.utils.math_utils',
    'load_config': 'orbexa.utils.io_utils',
    'is_key_pressed': 'orbexa.utils.io_utils',
    'thread_worker': 'orbexa.utils.io_utils',
    'stream_printer': 'orbexa.utils.io_utils',
    'get_next_test_folder': 'orbexa.utils.io_utils',
    'save_test_result': 'orbexa.utils.io_utils',
    'save_data': 'orbexa.utils.io_utils',
    'load_data': 'orbexa.utils.io_utils',
    'latest_data_file': 'orbexa.utils.io_utils',
    'convert_numpy_to_list': 'orbexa.utils.io_utils',
    'create_filename': 'orbexa.utils.io_utils',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
