"""Focused utility functions, without wildcard import side effects."""

from importlib import import_module

_EXPORTS = {
    'calc_distance': 'orbexa.utils.geometry.algebra',
    'random_unit_vectors': 'orbexa.utils.geometry.sampling',
    'gen_init_state': 'orbexa.utils.geometry.sampling',
    'gen_skew_sym_mat': 'orbexa.utils.geometry.algebra',
    'tait_bryan_to_rotation_matrix': 'orbexa.utils.geometry.algebra',
    'calc_current_pos': 'orbexa.utils.geometry.algebra',
    'discretize': 'orbexa.utils.discretization',
    'calc_local_occlusion_cost': 'orbexa.utils.geometry.occlusion',
    'calc_global_occlusion_cost': 'orbexa.utils.geometry.occlusion',
    'cylinder_inside_constraint': 'orbexa.utils.geometry.regions',
    'gen_shape_data': 'orbexa.utils.geometry.sampling',
    'gen_fib_lattice': 'orbexa.utils.geometry.sampling',
    'pyramidal_constraint': 'orbexa.utils.geometry.regions',
    'load_config': 'orbexa.utils.io.configuration',
    'is_key_pressed': 'orbexa.utils.runtime',
    'thread_worker': 'orbexa.utils.runtime',
    'stream_printer': 'orbexa.utils.runtime',
    'get_next_test_folder': 'orbexa.utils.io.data',
    'save_test_result': 'orbexa.utils.io.data',
    'save_data': 'orbexa.utils.io.data',
    'load_data': 'orbexa.utils.io.data',
    'latest_data_file': 'orbexa.utils.io.data',
    'convert_numpy_to_list': 'orbexa.utils.io.data',
    'create_filename': 'orbexa.utils.io.data',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
