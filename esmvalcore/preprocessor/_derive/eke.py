"""Derivation of variable `eke`."""

import logging

import iris
from cf_units import Unit

from ._baseclass import DerivedVariableBase

logger = logging.getLogger(__name__)


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `eke`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                "short_name": "uo",
            },
            {
                "short_name": "vo",
            },
        ]
        if project == "FESOM":
            required = [
                {"short_name": "uo", "var_type": "u"},
                {"short_name": "vo", "var_type": "v"},
            ]
        return required

    @staticmethod
    def calculate(cubes):
        """
        Compute ocean heat content.

        Arguments
        ---------
        cube: iris.cube.Cube
           input cube.

        Returns
        -------
        iris.cube.Cube
              Output OHC cube.
        """
        cube_u = cubes.extract_cube(iris.Constraint(name="uo"))
        cube_v = cubes.extract_cube(iris.Constraint(name="vo"))

        if len(cube_u.coord("time").points) > 1:
            cube_u_avg = cube_u.collapsed("time", iris.analysis.MEAN)
            cube_v_avg = cube_v.collapsed("time", iris.analysis.MEAN)
            cube_u_anom = cube_u - cube_u_avg
            cube_v_anom = cube_v - cube_v_avg
            cube = 0.5 * (cube_u_anom**2 + cube_v_anom**2)
        else:
            logger.warning("Time coordinate of length 1")
            cube = 0.5 * (cube_u**2 + cube_v**2)

        cube.var_name = "eke"
        cube.units = Unit("m2 s-2")
        cube.long_name = "Eddy kinetic energy in grid cell"

        return cube
