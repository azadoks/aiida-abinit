"""
Calculations provided by aiida_abinit.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""
import io
from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida.orm import (SinglefileData, StructureData, RemoteData)
from aiida.plugins import DataFactory

from abipy.abio.variable import InputVariable
from pymatgen.io.abinit.abiobjects import structure_to_abivars


class AbinitCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapping the abinit executable.

    Simple AiiDA plugin wrapper for running a basic Abinit DFT calculation.
    """

    # Defaults.
    _DEFAULT_INPUT_FILE = 'aiida.in'
    _DEFAULT_OUTPUT_FILE = 'aiida.out'
    _DEFAULT_PROJECT_NAME = 'aiida'
    _DEFAULT_GSR_FILE_NAME = _DEFAULT_PROJECT_NAME + 'o_GSR.nc'
    _DEFAULT_TRAJECT_FILE_NAME = _DEFAULT_PROJECT_NAME + 'o_HIST.nc'
    _DEFAULT_PARENT_CALC_FLDR_NAME = 'parent_calc/'

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        # yapf: disable
        super(AbinitCalculation, cls).define(spec)

        # Inputs
        spec.input('metadata.options.input_filename', valid_type=str, default=cls._DEFAULT_INPUT_FILE)
        spec.input('metadata.options.output_filename', valid_type=str, default=cls._DEFAULT_OUTPUT_FILE)  
        spec.input('metadata.options.output_gsr', valid_type=str, default=cls._DEFAULT_GSR_FILE_NAME)  
        spec.input('metadata.options.output_hist', valid_type=str, required=False, default=cls._DEFAULT_TRAJECT_FILE_NAME)  

        spec.input('parameters', valid_type=orm.Dict, help='the input parameters')
        spec.input('structure', valid_type=StructureData, required=False, help='the main input structure')
        spec.input('settings', valid_type=orm.Dict, required=False, help='special settings')        
        spec.input('parent_calc_folder', valid_type=RemoteData, required=False, help='remote folder used for restarts')

        # Abinit parser 
        spec.inputs['metadata']['options']['parser_name'].default = 'abinit'

        # Use mpi by default.
        spec.input('metadata.options.withmpi', valid_type=bool, default=True)
        spec.inputs['metadata']['options']['resources'].default = {
                'num_machines': 1, 'num_mpiprocs_per_machine': 1, }

        # Unrecoverable errors: file missing
        spec.exit_code(100, 'ERROR_MISSING_OUTPUT_FILES', message='Calculation did not produce all expected output files.')
        
        # Unrecoverable errors: resources like the retrieved folder or its expected contents are missing.
        spec.exit_code(200, 'ERROR_NO_RETRIEVED_FOLDER', message='The retrieved folder data node could not be accessed.')
        spec.exit_code(210, 'ERROR_OUTPUT_MISSING', message='The retrieved folder did not contain the required output file.')

        # Unrecoverable errors: required retrieved files could not be read, parsed or are otherwise incomplete.
        spec.exit_code(301, 'ERROR_OUTPUT_READ', message='The output file could not be read.')
        spec.exit_code(302, 'ERROR_OUTPUT_PARSE', message='The output file could not be parsed.')
        spec.exit_code(303, 'ERROR_OUTPUT_INCOMPLETE', message='The output file was incomplete.')
        spec.exit_code(304, 'ERROR_OUTPUT_CONTAINS_ABORT', message='The output file contains the word "ABORT"')
        spec.exit_code(312, 'ERROR_STRUCTURE_PARSE', message='The output structure could not be parsed.')
        spec.exit_code(350, 'ERROR_UNEXPECTED_PARSER_EXCEPTION', message='The parser raised an unexpected exception.')

        # Significant errors but calculation can be used to restart
        spec.exit_code(400, 'ERROR_OUT_OF_WALLTIME', message='The calculation stopped prematurely because it ran out of walltime.')
        spec.exit_code(500, 'ERROR_GEOMETRY_CONVERGENCE_NOT_REACHED',
                       message='The ionic minimization cycle did not converge for the given thresholds.')


        # Outputs
        spec.output('output_parameters', valid_type=orm.Dict, required=True, help='The result of the Abinit calculation.')
        spec.output('output_structure', valid_type=orm.StructureData, required=False,
            help='Optional relaxed crystal structure')
        spec.default_output_node = 'output_parameters'

        # SP: Not sure if I should set this ?
        spec.inputs.dynamic = True
        spec.outputs.dynamic = True

    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files needed by
            the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """

        # Add the used defined input parameters to the input file
        inp = self.inputs.parameters.get_dict()

        # Add the structure related variables to the dictionary
        if 'structure' in self.inputs:
            inp.update(structure_to_abivars(self.inputs.structure.get_pymatgen()))

        lines = []
        for inp_var, value in inp.items():
            lines.append(str(InputVariable(inp_var, value))+' \n')

        with io.open(folder.get_abs_path(self._DEFAULT_INPUT_FILE), mode="w", encoding="utf-8") as fobj:
            fobj.writelines(lines)

        # Create code info
        codeinfo = datastructures.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        # codeinfo.stdin_name = self.options.input_filename
        # This gives the path to the input file to Abinit rather than passing the input from standard input
        #codeinfo.cmdline_params = ['<', self.options.input_filename]
        codeinfo.cmdline_params = [self.options.input_filename]
        codeinfo.stdout_name = self.metadata.options.output_filename
        codeinfo.withmpi = self.inputs.metadata.options.withmpi

        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.stdin_name = self.options.input_filename
        calcinfo.stdout_name = self.options.output_filename       
        calcinfo.retrieve_list = [self.metadata.options.output_filename]
        calcinfo.retrieve_list = [self._DEFAULT_OUTPUT_FILE, self._DEFAULT_GSR_FILE_NAME, self._DEFAULT_TRAJECT_FILE_NAME]
        #calcinfo.retrieve_list += settings.pop('additional_retrieve_list', [])

        # Symlinks.
        calcinfo.remote_symlink_list = []
        calcinfo.remote_copy_list = []
        if 'parent_calc_folder' in self.inputs:
            comp_uuid = self.inputs.parent_calc_folder.computer.uuid
            remote_path = self.inputs.parent_calc_folder.get_remote_path()
            copy_info = (comp_uuid, remote_path, self._DEFAULT_PARENT_CALC_FLDR_NAME)

            # If running on the same computer - make a symlink.
            if self.inputs.code.computer.uuid == comp_uuid:
                calcinfo.remote_symlink_list.append(copy_info)
            # If not - copy the folder.
            else:
                calcinfo.remote_copy_list.append(copy_info)

        return calcinfo
