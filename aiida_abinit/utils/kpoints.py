# -*- coding: utf-8 -*-
"""k-point utility functions."""
import numpy as np

from aiida import orm
from aiida.engine import calcfunction

__all__ = ('create_kpoints_from_distance', 'seekpath_structure_analysis')


@calcfunction
def create_kpoints_from_distance(structure: orm.StructureData, distance: orm.Float) -> orm.KpointsData:
    """Generate a uniformly spaced kpoint mesh for a given structure.

    The spacing between kpoints in reciprocal space is guaranteed to be at least the defined distance.

    :param structure: the StructureData to which the mesh should apply
    :param distance: a Float with the desired distance between kpoints in reciprocal space
    :returns: a KpointsData with the generated mesh
    """
    epsilon = 1E-5

    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh_from_density(distance.value)

    lengths_vector = [np.linalg.norm(vector) for vector in structure.cell]
    lengths_kpoint = kpoints.get_kpoints_mesh()[0]

    is_symmetric_cell = all(abs(length - lengths_vector[0]) < epsilon for length in lengths_vector)
    is_symmetric_mesh = all(length == lengths_kpoint[0] for length in lengths_kpoint)

    # If the vectors of the cell all have the same length, the kpoint mesh should be isotropic as well
    if is_symmetric_cell and not is_symmetric_mesh:
        nkpoints = max(lengths_kpoint)
        kpoints.set_kpoints_mesh([nkpoints, nkpoints, nkpoints])

    return kpoints


@calcfunction
def seekpath_structure_analysis(structure, **kwargs):
    """Primitivize the structure with SeeKpath and generate the high symmetry k-point path through its Brillouin zone.

    This calcfunction will take a structure and pass it through SeeKpath to get the normalized primitive cell and the
    path of high symmetry k-points through its Brillouin zone. Note that the returned primitive cell may differ from the
    original structure in which case the k-points are only congruent with the primitive cell.
    The keyword arguments can be used to specify various Seekpath parameters, such as:
        with_time_reversal: True
        reference_distance: 0.025
        recipe: 'hpkot'
        threshold: 1e-07
        symprec: 1e-05
        angle_tolerance: -1.0
    Note that exact parameters that are available and their defaults will depend on your Seekpath version.
    """
    from aiida.tools import get_explicit_kpoints_path

    # All keyword arugments should be `Data` node instances of base type and so should have the `.value` attribute
    unwrapped_kwargs = {key: node.value for key, node in kwargs.items() if isinstance(node, orm.Data)}

    return get_explicit_kpoints_path(structure, **unwrapped_kwargs)
