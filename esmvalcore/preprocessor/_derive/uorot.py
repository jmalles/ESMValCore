"""Derivation of variable `rbynr`."""

import dask.array as da
import iris

from ._baseclass import DerivedVariableBase


def rotation(cube_u, cube_v, angles):
    angles.convert_units("radians")
    # uu, vv, aa = (cube.data for cube in (u_out, v_out, angles))
    mags = da.sqrt(cube_u.core_data() ** 2 + cube_v.core_data() ** 2)
    angs = (
        da.arctan2(cube_v.core_data(), cube_u.core_data()) + angles.core_data()
    )
    cube_u.data, cube_v.data = mags * da.cos(angs), mags * da.sin(angs)
    return cube_u, cube_v


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rbnr`."""

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

        Use c_p*rho_0= 4.09169e+6 J m-3 K-1
        (Kuhlbrodt et al., 2015, Clim. Dyn.)

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

        # if np.sign(cube_v.coord("latitude").points[-1, 0]) == -1:
        #     sort_range=np.argsort(np.arange(cube_v.shape[-2]-1,-1,-1))
        #     cube_v.data = cube_v.core_data()[..., sort_range, :]
        #     cube_v.coord("latitude").points = cube_v.coord("latitude").core_points()[sort_range, :]
        #     cube_v.coord("latitude").bounds = cube_v.coord("latitude").core_bounds()[sort_range, :]
        #     cube_v.coord("longitude").points = cube_v.coord("longitude").core_points()[sort_range, :]
        #     cube_v.coord("longitude").bounds = cube_v.coord("longitude").core_bounds()[sort_range, :]
        #     cube_u.data = cube_u.core_data()[..., sort_range, :]
        #     cube_u.coord("latitude").points = cube_u.coord("latitude").core_points()[sort_range, :]
        #     cube_u.coord("latitude").bounds = cube_u.coord("latitude").core_bounds()[sort_range, :]
        #     cube_u.coord("longitude").points = cube_u.coord("longitude").core_points()[sort_range, :]
        #     cube_u.coord("longitude").bounds = cube_u.coord("longitude").core_bounds()[sort_range, :]
        #     flip_dim = cube_v.coord_dims('cell index along second dimension')[0]
        #     extra_dims = int(cube_v.ndim - 2)
        #     cube_v.data = cube_v.core_data() * -1
        #     cube_u.data = da.flip(cube_u.data, flip_dim)
        #     cube_u.coord("latitude").points = da.flip(
        #         cube_u.coord("latitude").points, flip_dim-extra_dims
        #     )
        #     cube_u.coord("latitude").bounds = da.flip(
        #         cube_u.coord("latitude").points, flip_dim-extra_dims
        #     )
        #     cube_u.coord("longitude").points = da.flip(
        #         cube_u.coord("longitude").points, flip_dim-extra_dims
        #     )
        #     cube_u.coord("longitude").bounds = da.flip(
        #         cube_u.coord("longitude").points, flip_dim-extra_dims
        #     )

        angles = iris.analysis.cartography.gridcell_angles(
            cube_u.coord("longitude").points,
            cube_u.coord("latitude").points,
        )
        # angles = iris.analysis.cartography.gridcell_angles(cube_u)
        cube_u_r, _ = rotation(cube_u, cube_v, angles)
        return cube_u_r
