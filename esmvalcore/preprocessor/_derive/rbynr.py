"""Derivation of variable `rbynr`."""

import itertools
import os

import dask
import dask.array as da
import iris
import numpy as np
from cf_units import Unit

from ._baseclass import DerivedVariableBase

# from esmvalcore.preprocessor._area import compute_area_weights_unstructured

OMEGA = 7.2921e-5
earth_radius = 6367470  # 6371009 # in meters
lat_dist = np.pi * earth_radius / 180.0


def compute_x_dists(cube):
    bounds_shape = np.shape(cube.coord("latitude").bounds)
    lat_bounds = cube.coord("latitude").bounds
    lon_bounds = (cube.coord("longitude").bounds + 180) % 360 - 180
    if len(bounds_shape) == 3:
        lat_bounds = lat_bounds.flatten()
        lon_bounds = lon_bounds.flatten()
        lat_bounds = np.reshape(
            lat_bounds,
            (int(len(lat_bounds) / bounds_shape[-1]), int(bounds_shape[-1])),
        )
        lon_bounds = np.reshape(
            lon_bounds,
            (int(len(lon_bounds) / bounds_shape[-1]), int(bounds_shape[-1])),
        )
    elif cube.coord_dims("latitude") != cube.coord_dims("longitude"):
        new_grid1 = np.meshgrid(lon_bounds[:, 0], lat_bounds[:, 0])
        new_grid2 = np.meshgrid(lon_bounds[:, 1], lat_bounds[:, 1])
        lat_bounds = np.stack(
            [
                new_grid2[1].flatten(),
                new_grid1[1].flatten(),
                new_grid1[1].flatten(),
                new_grid2[1].flatten(),
            ],
            axis=1,
        )
        lon_bounds = np.stack(
            [
                new_grid2[0].flatten(),
                new_grid2[0].flatten(),
                new_grid1[0].flatten(),
                new_grid1[0].flatten(),
            ],
            axis=1,
        )

    earth_radius = 6367470  # 6371009 # in meters
    lat_dist = np.pi * earth_radius / 180.0

    def calculate_areas(longitude, latitude):
        if np.any(longitude < -179) & np.any(longitude > 179):
            longitude = (longitude + 360) % 360

        lon0 = np.mean(longitude)
        lat0 = np.mean(latitude)
        r = np.sqrt((longitude - lon0) ** 2 + (latitude - lat0) ** 2)
        angles = np.where(
            (latitude - lat0) > 0,
            np.arccos((longitude - lon0) / r),
            2 * np.pi - np.arccos((longitude - lon0) / r),
        )
        mask = np.argsort(angles)
        longitude = longitude[mask]
        latitude = latitude[mask]

        x = lat_dist * longitude * np.cos(np.deg2rad(latitude))
        y = lat_dist * latitude
        x_dist = np.max(x) - np.min(x)
        y_dist = np.max(y) - np.min(y)
        return ((x_dist**2 + y_dist**2) * 0.5) ** 0.5

    @dask.delayed
    def worker(longitude, latitude):
        areas = []
        for i in range(len(longitude)):
            areas.append(calculate_areas(longitude[i, :], latitude[i, :]))
        return areas

    n_cpus = os.cpu_count()
    n_max = lon_bounds.shape[0]
    idxs = np.array_split(range(n_max), n_cpus)
    results = [worker(lon_bounds[idx, :], lat_bounds[idx, :]) for idx in idxs]
    results = dask.compute(*results)
    weights = list(itertools.chain.from_iterable(list(results)))
    if len(bounds_shape) == 3:
        weights = np.reshape(weights, (bounds_shape[0], bounds_shape[1]))
    elif cube.coord_dims("latitude") != cube.coord_dims("longitude"):
        weights = np.reshape(
            weights,
            (
                len(cube.coord("latitude").points),
                len(cube.coord("longitude").points),
            ),
        )
    return da.array(weights)


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

        # cell_areas = compute_area_weights_unstructured(cube_u)
        lats = da.abs(da.deg2rad(cube_u.coord("latitude").core_points()))
        lengths = compute_x_dists(cube_u)
        # angles = da.arctan(da.abs(cube_v.core_data() / cube_u.core_data()))
        # x_dists = diagonal
        # lengths = dists / da.cos(da.deg2rad(angles))
        # lengths = cell_areas / x_dists
        coriolis = 2 * OMEGA * da.sin(lats)
        # denom = iris.coords.AuxCoord(lengths*coriolis,
        #  units=Unit('m rad s-1'))
        # denom = cube_u * 1
        # denom.data = lengths * coriolis
        if len(np.shape(lengths)) > len(np.shape(coriolis)):
            coriolis = coriolis[:, np.newaxis]
        denom = lengths * coriolis
        # denom.units = Unit("m rad s-1")

        cube = (cube_u**2 + cube_v**2) ** 0.5
        # cube = cube / denom
        cube = iris.analysis.maths.divide(
            cube,
            denom,
            dim=cube.coord_dims("longitude")[0],
        )

        cube.var_name = "rbynr"
        cube.units = Unit(1)
        cube.long_name = "Rossby number in grid cell"

        return cube
