#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a simple silicon band structure calculation using AbinitBandsWorkChain.

Use the AbinitBandsWorkChain

Usage: python ./example_bands.py --code abinit-9.2.1-ab@localhost --pseudo_family nc-sr-04_pbe_standard_psp8
"""
# %%
import os

import click
import pymatgen as pmg
from aiida import cmdline
from aiida.engine import run
from aiida.orm import Bool, Float, Dict, Group, StructureData
from aiida_abinit.workflows.bands import AbinitBandsWorkChain

# %%
def example_bands(code, pseudo_family):
    """Run silicon bands calculation."""

    print('Testing the AbinitBandsWorkChain on Silicon')

    thisdir = os.path.dirname(os.path.realpath(__file__))
    structure = StructureData(pymatgen=pmg.core.Structure.from_file(os.path.join(thisdir, 'files', 'Si.cif')))
    pseudo_family = Group.objects.get(label=pseudo_family)
    pseudos = pseudo_family.get_pseudos(structure=structure)
    metadata = {
        'options': {
            'withmpi': True,
            'max_wallclock_seconds': 20 * 60,
            'resources': {
                'num_machines': 1,
                'num_mpiprocs_per_machine': 12
            }
        }
    }

    bands_parameters_dict = {
        'structure': structure,
        'nbands_factor': Float(1.1),
        'nscf_kpoints_distance': Float(0.02),
        'clean_workdir': Bool(True),
        'relax': {
            'kpoints_distance': Float(0.10),
            'abinit': {
                'code': code,
                'pseudos': pseudos,
                'parameters': Dict(dict={
                    'ecut': 36.0,
                    'nstep': 50,
                    'toldfe': 1e-8,
                    'ionmov': 22,
                    'optcell': 2,
                    'ntime': 50,
                    'ecutsm': 0.05,
                    'dilatmx': 1.05
                }),
                'metadata': metadata
            }
        },
        'scf': {
            'kpoints_distance': Float(0.10),
            'abinit': {
                'code': code,
                'pseudos': pseudos,
                'parameters': Dict(dict={
                    'ecut': 36.0,
                    'nstep': 50,
                    'toldfe': 1e-8
                }),
                'metadata': metadata
            }
        },
        'nscf': {
            'abinit': {
                'code': code,
                'pseudos': pseudos,
                'parameters': Dict(dict={
                    'ecut': 36.0,
                    'tolwfr': 1e-5,
                }),
                'metadata': metadata
            }
        }
    }

    print('Running workchain...')
    return run(AbinitBandsWorkChain, **bands_parameters_dict)
# %%
import matplotlib.pyplot as plt
from aiida import orm, load_profile
load_profile('base')

code = orm.load_code('abinit-9.6.2-openmpi@zookspc-slurm')
pseudo_family = 'PseudoDojo/0.4/PBE/SR/standard/psp8'

outputs = example_bands(code, pseudo_family)
# %%
bs = outputs['band_structure']
bands = bs.get_bands()[0]

fig, ax = plt.subplots(dpi=300)
for band in bands.T:
    ax.plot(band, marker='.', markersize=1, linewidth=1, c='k')

ax.set_xticks([0, 56, 75, 76, 135, 183, 222, 249])
ax.set_xticklabels(['$\Gamma$', 'X', 'U', 'K', '$\Gamma$', 'L', 'W', 'X'])
# %%
PSEUDO_FAMILY = cmdline.params.options.OverridableOption(
    '-P', '--pseudo_family', help='Psp8Family identified by its label'
)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODE()
@PSEUDO_FAMILY()
def cli(code, pseudo_family):
    """Run example.

    Example usage: $ python ./example_bands.py --code abinit@localhost --pseudo_family psp8

    Help: $ python ./example_bands.py --help
    """
    example_bands(code, pseudo_family)


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
