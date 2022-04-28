# -*- coding: utf-8 -*-
"""Workchain to compute a band structure for a given structure using ABINIT."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, if_
from aiida.plugins import WorkflowFactory

from ..utils import seekpath_structure_analysis

AbinitBaseWorkChain = WorkflowFactory('abinit.base')


def validate_inputs(inputs, ctx=None):  # pylint: disable=unused-argument
    """Validate input parameters."""
    # pylint: disable=fixme
    # TODO: check that tolwfr is set
    # TODO: check that nstep is unset
    return


class AbinitBandsWorkChain(WorkChain):
    """Workchain to compute a band structure for a given structure using ABINIT."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(AbinitBaseWorkChain, namespace='relax', exclude=('clean_workdir', 'abinit.structure'))
        spec.expose_inputs(AbinitBaseWorkChain, namespace='scf', exclude=('clean_workdir', 'abinit.structure'))
        spec.expose_inputs(
            AbinitBaseWorkChain,
            namespace='nscf',
            exclude=('clean_workdir', 'kpoints', 'kpoints_distance', 'abinit.structure')
        )
        spec.input('structure', valid_type=orm.StructureData, help='The input structure.')
        spec.input(
            'clean_workdir',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, the work directories of all called calculations will be cleaned at the end of execution.'
        )
        spec.input(
            'nbands_factor',
            valid_type=orm.Float,
            required=False,
            help='Factor by which the number of bands from the SCF calculation is multiplied to get the number'
            ' of bands for the bands calculation.'
        )
        spec.input(
            'nscf_kpoints',
            valid_type=orm.KpointsData,
            required=False,
            help='Explicit k-points to use for the bands calculation. Specify either this _or_ `nscf_kpoints_distance`.'
        )
        spec.input(
            'nscf_kpoints_distance',
            valid_type=orm.Float,
            required=False,
            help='Minimum k-points distance to use for the bands calculation. Specify either this _or_ `nscf_kpoints`.'
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_relax)(cls.run_relax, cls.inspect_relax),
            if_(cls.should_run_seekpath)(cls.run_seekpath), cls.run_scf, cls.inspect_scf, cls.run_nscf,
            cls.inspect_nscf, cls.results
        )

        spec.exit_code(
            201,
            'ERROR_INVALID_INPUT_NUMBER_OF_BANDS',
            message='Cannot specify both `nbands_factor` and `bands.abinit.parameters.nband`.'
        )
        spec.exit_code(
            202,
            'ERROR_INVALID_INPUT_KPOINTS',
            message='Cannot specify both `nscf_kpoints` and `nscf_kpoints_distance`.'
        )
        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_RELAX', message='The relax AbinitBaseWorkChain sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_SCF', message='The SCF AbinitBaseWorkChain sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_BANDS', message='The NSCF AbinitBaseWorkChain sub process failed')

        spec.output(
            'primitive_structure',
            valid_type=orm.StructureData,
            required=False,
            help='The normalized and primitivized structure for which the bands are computed.'
        )
        spec.output(
            'seekpath_parameters',
            valid_type=orm.Dict,
            required=False,
            help='The parameters used in the SeeKpath call to normalize the input or relaxed structure.'
        )
        spec.output(
            'scf_parameters', valid_type=orm.Dict, help='The output parameters of the SCF `AbinitBaseWorkChain`.'
        )
        spec.output(
            'nscf_parameters', valid_type=orm.Dict, help='The output parameters of the NSCF `AbinitBaseWorkChain`.'
        )
        spec.output('band_structure', valid_type=orm.BandsData, help='The computed band structure.')

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        self.ctx.current_structure = self.inputs.structure
        self.ctx.current_number_of_bands = None
        self.ctx.nscf_kpoints = self.inputs.get('nscf_kpoints', None)

    def should_run_relax(self):
        """If the `relax` input namespace was specified, we relax the input structure."""
        return 'relax' in self.inputs

    def should_run_seekpath(self):
        """Seekpath should only be run if the `nscf_kpoints` input is not specified."""
        return 'nscf_kpoints' not in self.inputs

    def run_relax(self):
        """Run the AbinitBaseWorkChain to run a relax AbinitCalculation."""
        inputs = AttributeDict(self.exposed_inputs(AbinitBaseWorkChain, namespace='relax'))
        inputs.metadata.call_link_label = 'relax'
        inputs.abinit.structure = self.ctx.current_structure

        running = self.submit(AbinitBaseWorkChain, **inputs)

        self.report(f'launching AbinitBaseWorkChain<{running.pk}> `relax`')

        return ToContext(workchain_relax=running)

    def inspect_relax(self):
        """Verify that the AbinitBaseWorkChain finished successfully."""
        workchain = self.ctx.workchain_relax

        if not workchain.is_finished_ok:
            self.report(f'AbinitBaseWorkChain<{workchain.pk}> `relax` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.ctx.current_structure = workchain.outputs.output_structure
        self.ctx.current_number_of_bands = workchain.outputs.output_parameters.get_attribute('nband')

    def run_seekpath(self):
        """Run the structure through SeeKpath to get the normalized structure and path along high-symmetry k-points .

        This is only called if the `nscf_kpoints` input was not specified.
        """
        inputs = {
            'reference_distance': self.inputs.get('nscf_kpoints_distance', None),
            'metadata': {
                'call_link_label': 'seekpath'
            }
        }
        result = seekpath_structure_analysis(self.ctx.current_structure, **inputs)
        self.ctx.current_structure = result['primitive_structure']
        self.ctx.nscf_kpoints = result['explicit_kpoints']

        self.out('primitive_structure', result['primitive_structure'])
        self.out('seekpath_parameters', result['parameters'])

    def run_scf(self):
        """Run the AbinitBaseWorkChain in scf mode on the primitive cell of (optionally relaxed) input structure."""
        inputs = AttributeDict(self.exposed_inputs(AbinitBaseWorkChain, namespace='scf'))
        inputs.metadata.call_link_label = 'scf'
        inputs.abinit.structure = self.ctx.current_structure
        parameters = inputs.abinit.parameters.get_dict()
        # Need to check `usepaw` (iscf -> 17), usewvl (iscf -> 1), iscf -> 7 otherwise to ensure SCF calc.
        # inputs.abinit.parameters.setdefault('iscf', )
        # inputs.pw.parameters.setdefault('CONTROL', {})['calculation'] = 'scf'

        # Make sure to carry the number of bands from the relax workchain if it was run and it wasn't explicitly defined
        # in the inputs. One of the base workchains in the relax workchain may have changed the number automatically in
        #  the sanity checks on band occupations.
        if self.ctx.current_number_of_bands:
            parameters.setdefault('nband', self.ctx.current_number_of_bands)

        inputs.abinit.parameters = orm.Dict(dict=parameters)

        running = self.submit(AbinitBaseWorkChain, **inputs)

        self.report(f'launching AbinitBaseWorkChain<{running.pk}> `scf`')

        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """Verify that the AbinitBaseWorkChain for the scf run finished successfully."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                f'scf AbinitBaseWorkChain<{workchain.pk}> `scf` failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.ctx.current_number_of_bands = workchain.outputs.output_parameters.get_attribute('nband')

    def run_nscf(self):
        """Run the AbinitBaseWorkChain in nscf mode along the path of high-symmetry determined by seekpath."""
        inputs = AttributeDict(self.exposed_inputs(AbinitBaseWorkChain, namespace='nscf'))
        inputs.metadata.call_link_label = 'bands'
        inputs.kpoints = self.ctx.nscf_kpoints
        inputs.abinit.structure = self.ctx.current_structure
        inputs.abinit.parent_folder = self.ctx.current_folder
        parameters = inputs.abinit.parameters.get_dict()

        # The following flags always have to be set in the parameters, regardless of what caller specified in the inputs
        parameters['irdden'] = 1  # Do a restart calculation
        parameters['iscf'] = -2  # Do an nscf calculation

        # Only set the following parameters if not directly explicitly defined in the inputs
        parameters.setdefault('rmm_diis', 0)
        # parameters.setdefault('paral_kgb', 0)

        # If `nbands_factor` is defined in the inputs we set the `nband` parameter
        if 'nbands_factor' in self.inputs:
            factor = self.inputs.nbands_factor.value
            scf_parameters = self.ctx.workchain_scf.outputs.output_parameters.get_dict()
            nspin_factor = int(scf_parameters['nspinor'])
            nbands = int(scf_parameters['nband'])
            nelectron = int(scf_parameters['nelect'])
            nband = max(
                int(0.5 * nelectron * nspin_factor * factor),
                int(0.5 * nelectron * nspin_factor) + 4 * nspin_factor, nbands
            )
            parameters['nband'] = nband
        # Otherwise set the current number of bands, unless explicitly set in the inputs
        else:
            parameters.setdefault('nband', self.ctx.current_number_of_bands)

        inputs.abinit.parameters = orm.Dict(dict=parameters)

        running = self.submit(AbinitBaseWorkChain, **inputs)

        self.report(f'launching AbinitBaseWorkChain<{running.pk}> `nscf`')

        return ToContext(workchain_nscf=running)

    def inspect_nscf(self):
        """Verify that the AbinitBaseWorkChain for the nscf run finished successfully."""
        workchain = self.ctx.workchain_nscf

        if not workchain.is_finished_ok:
            self.report(
                f'bands AbinitBaseWorkChain<{workchain.pk}> `nscf` failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain succesfully completed')
        self.out('scf_parameters', self.ctx.workchain_scf.outputs.output_parameters)
        self.out('nscf_parameters', self.ctx.workchain_nscf.outputs.output_parameters)
        self.out('band_structure', self.ctx.workchain_nscf.outputs.output_bands)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")
