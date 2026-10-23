"""Unstructured grid regridding."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
import shapely
from iris.analysis import UnstructuredNearest as IrisUnstructuredNearest
from iris.cube import Cube
from scipy.spatial import ConvexHull, Delaunay

from esmvalcore.iris_helpers import (
    has_regular_grid,
    has_unstructured_grid,
    rechunk_cube,
)
from esmvalcore.preprocessor._shared import preserve_float_dtype

if TYPE_CHECKING:
    from iris.analysis.trajectory import UnstructuredNearestNeigbourRegridder
    from iris.coords import Coord
    from numpy.typing import DTypeLike

logger = logging.getLogger(__name__)


class UnstructuredNearest(IrisUnstructuredNearest):
    """Unstructured nearest-neighbor regridding scheme.

    This class is a wrapper around :class:`iris.analysis.UnstructuredNearest`
    that removes any additional X or Y coordinates prior to regridding if
    necessary. It can be used in :meth:`iris.cube.Cube.regrid`.

    """

    def regridder(
        self,
        src_cube: Cube,
        tgt_cube: Cube,
    ) -> UnstructuredNearestNeigbourRegridder:
        """Get regridder.

        Parameters
        ----------
        src_cube:
            Cube defining the source grid.
        tgt_cube:
            Cube defining the target grid.

        Returns
        -------
        UnstructuredNearestNeigbourRegridder
            Regridder instance.

        """
        # Unstructured nearest-neighbor regridding requires exactly one X and
        # one Y coordinate (latitude and longitude). Remove any X or Y
        # dimensional coordinates if necessary.
        for axis in ["x", "y"]:
            if src_cube.coords(axis=axis, dim_coords=True):
                coord = src_cube.coord(axis=axis, dim_coords=True)
                src_cube.remove_coord(coord)
        return super().regridder(src_cube, tgt_cube)


class UnstructuredLinearRegridder:
    """Unstructured bilinear regridder.

    Supports lazy regridding and weights caching.

    Warning
    -------
    This will drop all cell measures, ancillary variables and aux factories,
    and any auxiliary coordinate that spans the dimension of the unstrucutred
    grid.

    Parameters
    ----------
    src_cube:
        Cube defining the source grid.
    tgt_cube:
        Cube defining the target grid.

    """

    def __init__(self, src_cube: Cube, tgt_cube: Cube) -> None:
        """Initialize class instance."""
        if not has_unstructured_grid(src_cube):
            msg = (
                f"Source cube {src_cube.summary(shorten=True)} does not have "
                f"unstructured grid"
            )
            raise ValueError(
                msg,
            )
        if not has_regular_grid(tgt_cube):
            msg = (
                f"Target cube {tgt_cube.summary(shorten=True)} does not have "
                f"regular grid"
            )
            raise ValueError(
                msg,
            )
        src_lat = src_cube.coord("latitude").copy()
        src_lon = src_cube.coord("longitude").copy()
        tgt_lat = tgt_cube.coord("latitude").copy()
        tgt_lon = tgt_cube.coord("longitude").copy()
        self.src_coords = [src_lat, src_lon]
        self.tgt_coords = [tgt_lat, tgt_lon]
        self.tgt_n_lat = tgt_lat.core_points().size
        self.tgt_n_lon = tgt_lon.core_points().size

        # Calculate regridding weights and indices
        (self._weights, self._indices, self._convex_hull_idx) = (
            self._get_weights_and_idx(src_lat, src_lon, tgt_lat, tgt_lon)
        )

    def _get_weights_and_idx(
        self,
        src_lat: Coord,
        src_lon: Coord,
        tgt_lat: Coord,
        tgt_lon: Coord,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get regridding weights and indices.

        Note
        ----
        To consider the periodic boundary conditions of a sphere, the source
        points are expanded first with all points on their convex hull wrapped
        around by -360° and +360° before the weights calculation. This needs to
        be considered in the interpolation (see self._interpolate)!

        The output arrays will be numpy arrays here (instead of dask) since
        resulting arrays are only 2D and will be computed anyway later during
        regridding.

        """
        # Make sure that source and target grid have identical units
        src_lat = src_lat.copy()
        src_lon = src_lon.copy()
        tgt_lat = tgt_lat.copy()
        tgt_lon = tgt_lon.copy()
        src_lat.convert_units("degrees")
        src_lon.convert_units("degrees")
        tgt_lat.convert_units("degrees")
        tgt_lon.convert_units("degrees")

        # Bring points into correct format
        # src_points: (N, 2) where N is the number of source grid points
        # tgt_points: (M, 2) where M is the number of target grid points
        (src_points, tgt_points) = dask.compute(
            np.stack((src_lat.core_points(), src_lon.core_points()), axis=-1),
            np.stack(
                tuple(
                    tgt_coord.ravel()
                    for tgt_coord in np.meshgrid(
                        tgt_lat.core_points(),
                        tgt_lon.core_points(),
                        indexing="ij",
                    )
                ),
                axis=-1,
            ),
        )
        src_points = np.array(src_points)  # cannot be masked array
        tgt_points = np.array(tgt_points)  # cannot be masked array

        # Calculate convex hull of source points to consider periodic boundary
        # conditions
        hull = ConvexHull(src_points)
        n_hull = len(hull.vertices)

        # Wrap around points on convex hull by -360° and +360° and add them to
        # list of source points
        src_points_with_convex_hull = self._add_convex_hull_twice(
            src_points,
            hull.vertices,
        )
        lon_period = np.array(360, dtype=src_points_with_convex_hull.dtype)
        src_points_with_convex_hull[-2 * n_hull : -n_hull, 1] -= lon_period
        src_points_with_convex_hull[-n_hull:, 1] += lon_period

        # Actual weights calculation
        (weights, indices) = self._calculate_weights(
            src_points_with_convex_hull,
            tgt_points,
        )

        return (weights, indices, hull.vertices)

    def __call__(self, cube: Cube) -> Cube:
        """Perform regridding.

        Parameters
        ----------
        cube:
            Cube to be regridded.

        Returns
        -------
        Cube
            Regridded cube.

        """
        if not has_unstructured_grid(cube):
            msg = (
                f"Cube {cube.summary(shorten=True)} does not have "
                f"unstructured grid"
            )
            raise ValueError(
                msg,
            )
        coords = [cube.coord("latitude"), cube.coord("longitude")]
        if coords != self.src_coords:
            msg = (
                f"The given cube {cube.summary(shorten=True)} is not defined "
                f"on the same source grid as this regridder"
            )
            raise ValueError(
                msg,
            )

        # Get coordinates of regridded cube

        # (1) New dimensional coordinates are the ones from the source cube
        # (excluding the unstructured grid dimension) plus the (x, y) target
        # grid dimensions. All dimensions to the right of the unstructured grid
        # dimension need to be shifted to the right by 1.
        udim = cube.coord_dims("latitude")[0]
        dim_coords_and_dims = [
            (c, cube.coord_dims(c)[0])
            for c in cube.coords(dim_coords=True)
            if udim not in cube.coord_dims(c)
        ]
        dim_coords_and_dims = [
            (c, d) if d < udim else (c, d + 1)
            for (c, d) in dim_coords_and_dims
        ]
        dim_coords_and_dims.append((self.tgt_coords[0], udim))
        dim_coords_and_dims.append((self.tgt_coords[1], udim + 1))

        # (2) Include all auxiliary coordinates that do not span unstructured
        # grid dimension (also make sure to shift all dimensions to the right
        # of the unstructured grid to the right by 1)
        old_aux_coords_and_dims = [
            (c, cube.coord_dims(c))
            for c in cube.coords(dim_coords=False)
            if udim not in cube.coord_dims(c)
        ]
        aux_coords_and_dims = []
        for aux_coord, old_dims in old_aux_coords_and_dims:
            dims = tuple(d if d < udim else d + 1 for d in old_dims)
            aux_coords_and_dims.append((aux_coord, dims))

        # Create new cube with regridded data
        regridded_data = preserve_float_dtype(self._get_regridded_data)(cube)
        regridded_cube = Cube(
            regridded_data,
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims,
        )
        regridded_cube.metadata = cube.metadata

        return regridded_cube

    def _get_regridded_data(self, cube: Cube) -> np.ndarray | da.Array:
        """Get regridded data."""
        udim = cube.coord_dims("latitude")[0]

        # Cube must not be chunked along latitude and longitude dimension
        rechunk_cube(cube, ["latitude", "longitude"])

        # Make sure that masked arrays are filled with nan's so they are
        # handled properly
        npx = da if cube.has_lazy_data() else np
        src_data = npx.ma.filled(cube.core_data(), np.nan)

        # Perform actual regridding and assign correct mask
        regridded_data: np.ndarray | da.Array
        if cube.has_lazy_data():
            regridded_data = self._regrid_lazy(
                src_data,
                udim,
                self._weights.dtype,
            )
        else:
            regridded_data = self._regrid_eager(src_data, udim)
        regridded_data = npx.ma.masked_invalid(regridded_data)

        return regridded_data

    def _regrid_eager(self, data: np.ndarray, axis: int) -> np.ndarray:
        """Eager regridding."""
        v_interpolate = np.vectorize(
            self._interpolate,
            signature="(i)->(lat,lon)",
        )

        # Make sure that interpolation dimension is rightmost dimension and
        # change it back after regridding
        data = np.moveaxis(data, axis, -1)
        regridded_arr = v_interpolate(data)
        regridded_arr = np.moveaxis(regridded_arr, -2, axis)
        return np.moveaxis(regridded_arr, -1, axis + 1)

    def _regrid_lazy(
        self,
        data: da.Array,
        axis: int,
        dtype: DTypeLike,
    ) -> da.Array:
        """Lazy regridding."""
        return da.apply_gufunc(
            self._interpolate,
            "(i)->(lat,lon)",
            data,
            axes=[(axis,), (axis, axis + 1)],
            vectorize=True,
            output_dtypes=dtype,
            output_sizes={"lat": self.tgt_n_lat, "lon": self.tgt_n_lon},
        )

    def _interpolate(self, data: np.ndarray) -> np.ndarray:
        """Interpolate data.

        Data to interpolate must be an (N,) array, where N is the number of
        source grid points. Indices used to index the data and interpolation
        weights must be (M, 3) arrays, where M is the number of target grid
        points.

        The returned array is of shape (lat, lon), where lat is the number of
        latitudes in the target grid, and lon the number of longitudes in the
        target grid such that lat x lon = M.

        Note
        ----
        Before the interpolation, the input is extended by the data on the
        convex hull to consider the periodic boundary conditions of a sphere
        (this has also been done for the weights calculation).

        """
        data = self._add_convex_hull_twice(data, self._convex_hull_idx)
        interp_data = np.einsum(
            "nj,nj->n",
            np.take(data, self._indices),
            self._weights,
        )
        return interp_data.reshape(self.tgt_n_lat, self.tgt_n_lon)

    @staticmethod
    def _add_convex_hull_twice(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Expand array with convex hull values (given by indices) twice."""
        convex_hull = arr[idx]
        return np.concatenate((arr, convex_hull, convex_hull))

    @staticmethod
    def _calculate_weights(
        src_points: np.ndarray,
        tgt_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate regridding weights using Delaunay triangulation.

        Partly taken from https://stackoverflow.com/a/20930910.

        Output shapes (M: number of target grid points)
        - weights: (M, 3)
        - indices: (M, 3)

        """
        tri = Delaunay(src_points)
        simplex = tri.find_simplex(tgt_points)
        indices = np.take(tri.simplices, simplex, axis=0)
        transform = np.take(tri.transform, simplex, axis=0)
        delta = tgt_points - transform[:, 2]
        bary = np.einsum("njk,nk->nj", transform[:, :2, :], delta)
        weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        extra_idx = simplex == -1
        weights[extra_idx, :] = np.nan  # missing values
        return (weights, indices)


class UnstructuredLinear:
    """Unstructured bilinear regridding scheme.

    This class can be used in :meth:`iris.cube.Cube.regrid`.

    Supports lazy regridding.

    Warning
    -------
    This will drop all cell measures, ancillary variables and aux factories,
    and any auxiliary coordinate that spans the dimension of the unstrucutred
    grid.

    """

    def __repr__(self) -> str:
        """Return string representation of class."""
        return "UnstructuredLinear()"

    def regridder(
        self,
        src_cube: Cube,
        tgt_cube: Cube,
    ) -> UnstructuredLinearRegridder:
        """Get regridder.

        Parameters
        ----------
        src_cube:
            Cube defining the source grid.
        tgt_cube:
            Cube defining the target grid.

        Returns
        -------
        UnstructuredLinearRegridder
            Regridder instance.

        """
        return UnstructuredLinearRegridder(src_cube, tgt_cube)


# def calculate_areas(longitude, latitude):
#     earth_radius = 6367470  # 6371009 # in meters
#     lat_dist = np.pi * earth_radius / 180.0
#     if np.any(longitude < -179) & np.any(longitude > 179):
#         longitude = (longitude + 360) % 360
#     lon0 = np.mean(longitude)
#     lat0 = np.mean(latitude)
#     r = np.sqrt((longitude - lon0) ** 2 + (latitude - lat0) ** 2)
#     angles = np.where(
#         (latitude - lat0) > 0,
#         np.arccos((longitude - lon0) / r),
#         2 * np.pi - np.arccos((longitude - lon0) / r),
#     )
#     mask = np.argsort(angles)
#     longitude = longitude[mask]
#     latitude = latitude[mask]

#     y = lat_dist * latitude
#     x = lat_dist * longitude * np.cos(np.deg2rad(latitude))

#     correction = x[-1] * y[0] - y[-1] * x[0]
#     main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
#     polygon_area = 0.5 * np.abs(main_area + correction)
#     return polygon_area


def UnstructuredAreaWeighted(cube, grid_size, file, lon_offset, lat_offset):
    import itertools
    import math
    import time
    from pathlib import Path

    import dask
    import dask.array as da
    import geopandas as gpd
    import iris
    import pandas as pd
    import scipy
    import sparse
    import xarray as xr
    from dask.distributed import wait
    from distributed.client import _get_global_client

    # from shapely.ops import split
    from esmvalcore.preprocessor._regrid import _generate_cube_from_dimcoords

    t0 = time.time()
    grid_size = float(grid_size.split("x")[0])

    lat_pts = cube.coord("latitude").points
    lon_pts = cube.coord("longitude").points
    lat_bnds = cube.coord("latitude").bounds
    lon_bnds = cube.coord("longitude").bounds

    lon_bnds = (lon_bnds + 360) % 360
    lon_pts = (lon_pts + 360) % 360

    min_lat = (
        math.floor(np.nanmin(lat_bnds)) if np.nanmin(lat_bnds) > -89 else -90
    )
    max_lat = (
        math.ceil(np.nanmax(lat_bnds)) if np.nanmax(lat_bnds) < 89 else 90
    )
    max_lat = max_lat + grid_size / 2

    zero_meri = np.any(
        (np.max(lon_bnds, axis=1) - np.min(lon_bnds, axis=1)) >= 180,
    )

    # if len(ranges) > 1:
    if zero_meri:
        min_lon0 = math.floor(np.min((lon_bnds + 180) % 360 - 180))
        max_lon0 = 360
        min_lon1 = 0
        max_lon1 = math.ceil(np.max((lon_bnds + 180) % 360 - 180))

        if lon_offset:
            min_lon0 = min_lon0 + grid_size / 2
            min_lon1 = min_lon1 + grid_size / 2
            max_lon0 = max_lon0 + grid_size / 2
            max_lon1 = max_lon1 + grid_size / 2

        if lat_offset:
            min_lat = min_lat + grid_size / 2

        range0 = np.arange((min_lon0 + 360) % 360, max_lon0, grid_size)
        range1 = np.arange(min_lon1, (max_lon1 + 360) % 360, grid_size)
        if len(range1) == 1:
            range1 = np.append(range1, range1[0] + grid_size)
        if len(range0) == 1:
            range0 = np.insert(range0, 0, range0[0] - grid_size)
        # ranges = np.concatenate([range1, range0])

        # logger.debug(
        #     "ranges %s, %s, %s, %s", min_lon0, max_lon0, min_lon1, max_lon1
        # )

        tgt_cube1 = _generate_cube_from_dimcoords(
            np.arange(min_lat, max_lat, grid_size),
            range0,
        )
        tgt_cube2 = _generate_cube_from_dimcoords(
            np.arange(min_lat, max_lat, grid_size),
            range1,
        )
        tgt_cube = iris.cube.CubeList([tgt_cube1, tgt_cube2]).concatenate()[0]

    else:
        min_lon = (
            math.floor(np.nanmin(lon_bnds)) if np.nanmin(lon_bnds) > 1 else 0
        )
        max_lon = (
            math.ceil(np.nanmax(lon_bnds))
            if np.nanmax(lon_bnds) < 359
            else 360
        )

        if lon_offset:
            min_lon = min_lon + grid_size / 2
            max_lon = max_lon + grid_size / 2

        if lat_offset:
            min_lat = min_lat + grid_size / 2

        tgt_cube = _generate_cube_from_dimcoords(
            np.arange(min_lat, max_lat, grid_size),
            np.arange(min_lon, max_lon, grid_size),
        )

    client = _get_global_client()
    if client:
        n_workers = len(client.scheduler_info(n_workers=-1)["workers"])
    else:
        n_workers = os.cpu_count()

    c_p_w = 1

    a_tol = file.get("a_tol", 0.5)

    filename = file.get("name", None)
    filepath = Path(str(filename))

    cube.data = da.ma.masked_where(cube.core_data() == 0.0, cube.core_data())
    cube.data = da.ma.masked_where(
        da.isnan(cube.core_data()), cube.core_data()
    )

    if filepath.is_file():
        weights = xr.open_dataset(filepath)
        matrix = scipy.sparse.csc_matrix(
            (
                weights["weights"],
                (
                    weights["tgt_idx"],
                    weights["src_idx"],
                ),
            ),
            shape=(len(weights["cells_tgt"]), len(weights["cells_src"])),
        )

        if (
            len(tgt_cube.coord("latitude").points)
            * len(tgt_cube.coord("longitude").points)
        ) < len(weights.lon_tgt):
            test_lat = [
                np.where(
                    np.isclose(
                        weights.lat_tgt,
                        tgt_cube.coord("latitude").points[i],
                    ),
                )
                for i in range(len(tgt_cube.coord("latitude").points))
            ]
            test_lon = [
                np.where(
                    np.isclose(
                        weights.lon_tgt,
                        tgt_cube.coord("longitude").points[i],
                    ),
                )
                for i in range(len(tgt_cube.coord("longitude").points))
            ]

            test_lat = list(
                itertools.chain.from_iterable(
                    list(itertools.chain.from_iterable(list(test_lat))),
                ),
            )
            test_lon = list(
                itertools.chain.from_iterable(
                    list(itertools.chain.from_iterable(list(test_lon))),
                ),
            )
            idx_tgt = np.unique(np.intersect1d(test_lat, test_lon))

            # find_coord = [
            #     i
            #     for i in range(len(cube.coords()))
            #     if "index" in cube.coords()[i].name()
            # ]
            # if len(find_coord) != 1:
            test_lat = [
                np.where(
                    np.isclose(
                        weights.lat_src,
                        cube.coord("latitude").points[i],
                    ),
                )
                for i in range(len(cube.coord("latitude").points))
            ]
            test_lon = [
                np.where(
                    np.isclose(
                        weights.lon_src,
                        cube.coord("longitude").points[i],
                    ),
                )
                for i in range(len(cube.coord("longitude").points))
            ]
            test_lat = list(
                itertools.chain.from_iterable(
                    list(itertools.chain.from_iterable(list(test_lat))),
                ),
            )
            test_lon = list(
                itertools.chain.from_iterable(
                    list(itertools.chain.from_iterable(list(test_lon))),
                ),
            )
            idx_src = np.unique(np.intersect1d(test_lat, test_lon))
            # else:
            #     idx_src = cube.coords()[find_coord[0]].points
            logger.info("idx_tgt %s", idx_tgt)
            logger.info("idx_src %s", idx_src)
            matrix = matrix[idx_tgt, :][:, idx_src]
            matrix.eliminate_zeros()
            new_grid = np.meshgrid(
                tgt_cube.coord("longitude").points,
                tgt_cube.coord("latitude").points,
            )
            weights = xr.Dataset(
                {
                    "weights": (["n_isect"], matrix.data),
                    "tgt_idx": (["n_isect"], matrix.nonzero()[0]),
                    "src_idx": (["n_isect"], matrix.nonzero()[1]),
                },
                coords={
                    "lon_src": ("cells_src", cube.coord("longitude").points),
                    "lat_src": ("cells_src", cube.coord("latitude").points),
                    "lat_tgt": ("cells_tgt", new_grid[0].flatten()),
                    "lon_tgt": ("cells_tgt", new_grid[1].flatten()),
                },
            )

        isect = pd.DataFrame(
            {
                "a_frac": weights.weights,
                "i_new": weights.tgt_idx,
                "i_old": weights.src_idx,
            }
        )

        new_chunks = ()
        for i in range(len(cube.core_data().shape)):
            new_chunks = new_chunks + tuple(
                [
                    np.max(
                        [
                            1,
                            int(
                                cube.core_data().shape[i]
                                // (n_workers / c_p_w)
                            ),
                        ]
                    )
                ],
            )
        logger.info("chunksize before %s", cube.core_data().chunksize)
        cube.data = cube.core_data().rechunk(
            chunks=new_chunks,
            balance=True,
        )  # block_size_limit="16GiB"
        logger.info("chunksize after %s", cube.core_data().chunksize)

    else:
        test1 = np.where(
            (np.max(lon_bnds, axis=1) - np.min(lon_bnds, axis=1)) >= 180,
        )[0]

        test3 = np.where(
            (
                np.max((lon_bnds[test1] + 180) % 360 - 180, axis=1)
                - np.min((lon_bnds[test1] + 180) % 360 - 180, axis=1)
            )
            >= 180,
        )[0]

        mask = da.ma.getmaskarray(cube.core_data())
        if len(cube.shape) == 2:
            mask_arr = mask[0, :]
            # idx_valid = da.isfinite(cube.core_data().flatten())
        elif len(cube.shape) == 3:
            mask_arr = mask[0, 0, :]
        else:
            mask_arr = mask
        # mask_arr = mask_arr.flatten()[:cube.core_data().shape[-1]]
        # mask_arr = mask_arr[:cube.core_data().shape[-1]]
        n_max = lon_bnds.shape[0]
        valid_idxs = np.arange(n_max)
        valid_idxs = np.delete(valid_idxs, mask_arr.nonzero()[0].compute())
        test1 = np.delete(test1, test3)

        # def compute_geoms(lats, lons):
        #     geoms = [
        #         shapely.geometry.Polygon(zip(lons[i, :], lats[i, :]))
        #         for i in range(lons.shape[0])
        #     ]
        #     return geoms

        def compute_geoms(lats, lons):
            coords = np.stack([lons, lats], axis=-1)
            return shapely.polygons(coords)

        @dask.delayed
        def worker(lats, lons, idx):
            odf = []
            """worker function"""
            # odf.append(compute_geoms(lons, lats))
            geoms = compute_geoms(lats, lons)
            idf = gpd.GeoDataFrame(
                {
                    "grid_lat_old": lat_pts[idx],
                    "grid_lon_old": lon_pts[idx],
                    "lat_min_old": np.min(lats, axis=1),
                    "lat_max_old": np.max(lats, axis=1),
                    "lon_min_old": np.min(lons, axis=1),
                    "lon_max_old": np.max(lons, axis=1),
                    "i_old": idx,
                },
                geometry=geoms,
            )
            odf.append(idf)
            return odf

        t1 = time.time()
        logger.info("time start %s", t0 - t1)

        t0 = time.time()
        # n_max = lon_bnds.shape[0]
        # test = (~np.isin(range(n_max), test1)).nonzero()[0]
        test = valid_idxs[~np.isin(valid_idxs, test1)]
        idxs = np.array_split(np.arange(n_max)[test], n_workers)

        results = [
            worker(lat_bnds[idx, :], lon_bnds[idx, :], idx) for idx in idxs
        ]
        results = dask.compute(*results)
        results = list(itertools.chain.from_iterable(list(results)))
        old_df = pd.concat(results)
        # results = list(itertools.chain.from_iterable(results))
        t1 = time.time()
        logger.info("time old_df %s", t0 - t1)

        if len(test1) > 0:
            t0 = time.time()
            idxs = np.array_split(test1, n_workers)
            results = [
                worker(
                    lat_bnds[idx, :],
                    (lon_bnds[idx, :] + 180) % 360 - 180,
                    idx,
                )
                for idx in idxs
            ]
            results = dask.compute(*results)
            results = list(itertools.chain.from_iterable(list(results)))
            # results = list(itertools.chain.from_iterable(results))
            old_df2 = pd.concat(results)
            t1 = time.time()
            logger.info("time old_df2 %s", t0 - t1)

        t0 = time.time()
        n_add = np.max([5, 1 / (10 * grid_size)])

        new_grid1 = np.meshgrid(
            tgt_cube.coord("longitude").bounds[:, 0],
            tgt_cube.coord("latitude").bounds[:, 0],
        )
        new_grid2 = np.meshgrid(
            tgt_cube.coord("longitude").bounds[:, 1],
            tgt_cube.coord("latitude").bounds[:, 1],
        )
        new_grid3 = np.meshgrid(
            tgt_cube.coord("longitude").points,
            tgt_cube.coord("latitude").points,
        )

        grid_corner_left = new_grid1[0].flatten()
        grid_corner_bottom = new_grid1[1].flatten()
        grid_corner_right = new_grid2[0].flatten()
        grid_corner_top = new_grid2[1].flatten()
        grid_center_lon = new_grid3[0].flatten()
        grid_center_lat = new_grid3[1].flatten()

        new_mesh = np.stack(
            [
                grid_corner_right,
                grid_corner_left,
                grid_corner_top,
                grid_corner_bottom,
            ],
            axis=1,
        )

        areas_list1 = iris.analysis.cartography.area_weights(
            tgt_cube,
        ).flatten()

        # def compute_geoms(new_mesh):
        #     geoms = [
        #         shapely.geometry.Polygon(
        #             [
        #                 [new_mesh[i, 0], new_mesh[i, 2]],
        #                 [new_mesh[i, 1], new_mesh[i, 2]],
        #                 [new_mesh[i, 1], new_mesh[i, 3]],
        #                 [new_mesh[i, 0], new_mesh[i, 3]],
        #             ]
        #         )
        #         for i in range(new_mesh.shape[0])
        #     ]
        #     return geoms

        def compute_geoms(new_mesh):
            p1 = np.stack([new_mesh[:, 0], new_mesh[:, 2]], axis=-1)
            p2 = np.stack([new_mesh[:, 1], new_mesh[:, 2]], axis=-1)
            p3 = np.stack([new_mesh[:, 1], new_mesh[:, 3]], axis=-1)
            p4 = np.stack([new_mesh[:, 0], new_mesh[:, 3]], axis=-1)

            coords = np.stack([p1, p2, p3, p4], axis=1)

            return shapely.polygons(coords)

        @dask.delayed
        def worker(new_mesh, idx):
            odf = []
            """worker function"""
            geoms = compute_geoms(new_mesh)
            idf = gpd.GeoDataFrame(
                {
                    "grid_lat_new": grid_center_lat[idx],
                    "grid_lon_new": grid_center_lon[idx],
                    "i_new": idx,
                },
                geometry=geoms,
            )
            odf.append(idf)
            return odf

        n_max = new_mesh.shape[0]
        idxs = np.array_split(range(n_max), n_workers)

        results = [worker(new_mesh[idx, :], idx) for idx in idxs]
        results = dask.compute(*results)
        results = list(itertools.chain.from_iterable(list(results)))
        new_df = pd.concat(results)
        new_df = new_df.sort_values("i_new")
        new_df["a_new"] = areas_list1
        t1 = time.time()
        logger.info("time new_df %s", t0 - t1)
        if len(test1) > 0:
            t0 = time.time()
            new_mesh2 = (new_mesh + 180) % 360 - 180
            test3 = np.where(
                (
                    new_mesh2[:, 1]
                    > np.min((lon_bnds[test1] + 180) % 360 - 180)
                    - n_add * grid_size
                )
                & (
                    new_mesh2[:, 1]
                    < np.max((lon_bnds[test1] + 180) % 360 - 180)
                    + n_add * grid_size
                ),
            )[0]
            new_mesh2 = new_mesh2[test3, :]
            areas_list2 = areas_list1[test3]

            @dask.delayed
            def worker(new_mesh, idx, idx_2):
                odf = []
                """worker function"""
                geoms = compute_geoms(new_mesh)
                idf = gpd.GeoDataFrame(
                    {
                        "grid_lat_new": (new_mesh2[idx, 2] + new_mesh2[idx, 3])
                        / 2,
                        "grid_lon_new": (new_mesh2[idx, 0] + new_mesh2[idx, 1])
                        / 2,
                        "i_new": idx_2[idx],
                    },
                    geometry=geoms,
                )
                odf.append(idf)
                return odf

            n_max = new_mesh2.shape[0]
            idxs = np.array_split(range(n_max), n_workers)
            results = [worker(new_mesh2[idx, :], idx, test3) for idx in idxs]
            results = dask.compute(*results)
            results = list(itertools.chain.from_iterable(list(results)))
            new_df2 = pd.concat(results)
            new_df2 = new_df2.sort_values("i_new")
            new_df2["a_new"] = areas_list2
            t1 = time.time()
            logger.info("time new_df2 %s", t0 - t1)

        t0 = time.time()
        n_max = old_df.shape[0]
        n_split = int(math.ceil(n_max / n_workers))
        old_df = old_df.sort_values("grid_lat_old")
        chunks_old = [
            old_df.iloc[i : i + n_split] for i in range(0, n_max, n_split)
        ]
        chunks_new = [
            new_df[
                (
                    (
                        new_df.grid_lat_new
                        <= chunks_old[i].lat_max_old.max() + n_add * grid_size
                    )
                    & (
                        new_df.grid_lat_new
                        >= chunks_old[i].lat_min_old.min() - n_add * grid_size
                    )
                )
                & (
                    (
                        new_df.grid_lon_new
                        <= chunks_old[i].lon_max_old.max() + n_add * grid_size
                    )
                    & (
                        new_df.grid_lon_new
                        >= chunks_old[i].lon_min_old.min() - n_add * grid_size
                    )
                )
            ]
            for i in range(len(chunks_old))
        ]

        lazy_results = []
        for chunk_old, chunk_new in zip(chunks_old, chunks_new):
            lazy_result = dask.delayed(gpd.overlay)(
                chunk_old,
                chunk_new,
                keep_geom_type=True,
            )
            lazy_results.append(lazy_result)

        futures = dask.persist(*lazy_results)
        results = dask.compute(*futures)
        wait(futures)
        isect = pd.concat(results)
        t1 = time.time()
        logger.info("time isect %s", t0 - t1)
        if len(test1) > 0:
            t0 = time.time()
            n_max = old_df2.shape[0]
            n_split = int(math.ceil(n_max / n_workers))
            old_df2.lon_max_old = (old_df2.lon_max_old + 180) % 360 - 180
            old_df2.lon_min_old = (old_df2.lon_min_old + 180) % 360 - 180
            old_df2 = old_df2.sort_values("grid_lat_old")
            chunks_old = [
                old_df2.iloc[i : i + n_split] for i in range(0, n_max, n_split)
            ]
            chunks_new = [
                new_df2[
                    (
                        (
                            new_df2.grid_lat_new
                            <= chunks_old[i].lat_max_old.max()
                            + n_add * grid_size
                        )
                        & (
                            new_df2.grid_lat_new
                            >= chunks_old[i].lat_min_old.min()
                            - n_add * grid_size
                        )
                    )
                    & (
                        (
                            new_df2.grid_lon_new
                            <= chunks_old[i].lon_max_old.max()
                            + n_add * grid_size
                        )
                        & (
                            new_df2.grid_lon_new
                            >= chunks_old[i].lon_min_old.min()
                            - n_add * grid_size
                        )
                    )
                ]
                for i in range(len(chunks_old))
            ]
            old_df2.grid_lon_old = (old_df2.grid_lon_old + 360) % 360
            lazy_results = []
            for chunk_old, chunk_new in zip(chunks_old, chunks_new):
                lazy_result = dask.delayed(gpd.overlay)(
                    chunk_old,
                    chunk_new,
                    keep_geom_type=True,
                )
                lazy_results.append(lazy_result)

            futures = dask.persist(*lazy_results)
            results = dask.compute(*futures)
            wait(futures)
            isect2 = pd.concat(results)
            isect = pd.concat([isect, isect2])
            t1 = time.time()
            logger.info("time isect2 %s", t0 - t1)

        t0 = time.time()
        new_chunks = ()
        for i in range(len(cube.core_data().shape)):
            new_chunks = new_chunks + tuple(
                [
                    np.max(
                        [
                            1,
                            int(
                                cube.core_data().shape[i]
                                // (n_workers / c_p_w)
                            ),
                        ]
                    )
                ],
            )

        logger.info("chunksize before %s", cube.core_data().chunksize)
        cube.data = cube.core_data().rechunk(
            chunks=new_chunks,
            balance=True,
        )  # block_size_limit="16GiB"
        logger.info("chunksize after %s", cube.core_data().chunksize)

        isect = isect.set_crs("EPSG:4326")
        isect.set_index("i_old", drop=False, inplace=True)
        # logger.info("isect %s", isect)
        t0 = time.time()
        isect["a_frac"] = (
            isect.to_crs("+proj=sinu +R=6367470").area.values / isect["a_new"]
        )
        t1 = time.time()
        logger.info("time areas %s", t0 - t1)

        t0 = time.time()
        matrix = scipy.sparse.csc_matrix(
            (
                isect["a_frac"],
                (
                    isect["i_new"],
                    isect["i_old"],
                ),
            ),
            shape=(len(grid_center_lon), len(lon_pts)),
        )

        weights = xr.Dataset(
            {
                "weights": (["n_isect"], isect["a_frac"]),
                "tgt_idx": (["n_isect"], isect["i_new"]),
                "src_idx": (["n_isect"], isect["i_old"]),
            },
            coords={
                "lon_src": ("cells_src", lon_pts),
                "lat_src": ("cells_src", lat_pts),
                "lat_tgt": ("cells_tgt", grid_center_lat),
                "lon_tgt": ("cells_tgt", grid_center_lon),
            },
        )

        if file.get("save", True):
            weights.to_netcdf(filename)

    t1 = time.time()
    logger.info("time matrix + xr.dataset %s", t0 - t1)
    # logger.info("chunksize/type matrix %s/%s", matrix.chunksize, type(matrix))

    t0 = time.time()
    try:
        fill_value = cube.core_data().get_fill_value()
    except AttributeError:
        fill_value = np.nan  # 1e+20

    # if len(cube.shape) == 2:
    #     idx_nan = cube.core_data().mask.flatten()
    #     # idx_valid = da.isfinite(cube.core_data().flatten())
    # else:
    #     idx_nan = cube.core_data().mask[0,:,:].flatten()

    # cube.data = da.ma.filled(cube.core_data(), fill_value=np.nan)
    # cube.data = da.where(cube.core_data() == 0.0, np.nan, cube.core_data())
    dims = ((1), (-1))
    # logger.info("isect %s", isect)
    respect_depth = file.get("respect_depth", None)
    if respect_depth is None:
        if cube.coords(axis="Z"):
            if len(cube.coord(axis="Z").points) > 1:
                respect_depth = True
            else:
                respect_depth = False
        else:
            respect_depth = False
    elif cube.coords(axis="Z"):
        if len(cube.coord(axis="Z").points) <= 1:
            respect_depth = False

    if respect_depth:
        mask = da.ma.getmaskarray(cube.core_data())
        idx_nan = mask if len(cube.shape) == 2 else mask[0, :, :]
        idx_valid = da.logical_not(idx_nan)
        # logger.info("idx_valid %s", idx_valid.shape)
        idx_valid = da.flatnonzero(idx_valid).compute()
        # logger.info("idx_valid %s, %s", idx_valid, idx_valid.shape)
        list_old = isect["i_old"].values
        list_new = isect["i_new"].values

        n_depth = len(cube.coord(axis="Z").points)
        len_tgt = len(weights["cells_tgt"])
        len_src = len(weights["cells_src"])
        i_old = [list_old + i * len_src for i in range(n_depth)]
        i_new = [list_new + i * len_tgt for i in range(n_depth)]
        # logger.info("i_old %s", i_old)
        # logger.info("i_new %s", i_new)

        a_frac = np.tile(
            isect["a_frac"].values,
            len(cube.coord(axis="Z").points),
        )
        i_old = np.hstack(i_old)
        i_new = np.hstack(i_new)
        list_valid = np.isin(i_old, idx_valid)
        # logger.info("list_valid %s, %s", list_valid, list_valid.shape)
        # isect = pd.DataFrame(
        #     {
        #         "a_frac": a_frac[list_valid],
        #         "i_new": i_new[list_valid],
        #         "i_old": i_old[list_valid],
        #     }
        # )
        # isect.set_index('i_old', drop=False, inplace=True)
        # logger.info("isect %s", isect)

        a_frac = a_frac[list_valid]
        i_new = i_new[list_valid]
        i_old = i_old[list_valid]

        matrix = scipy.sparse.csc_matrix(
            (
                a_frac,
                (
                    i_new,
                    i_old,
                ),
            ),
            shape=(
                len(weights["cells_tgt"]) * len(cube.coord(axis="Z").points),
                len(weights["cells_src"]) * len(cube.coord(axis="Z").points),
            ),
        )

        # matrix = scipy.sparse.csc_matrix(
        #     (
        #         isect["a_frac"],
        #         (
        #             isect["i_new"],
        #             isect["i_old"],
        #             ),
        #         ),
        #     shape=(
        #         len(weights["cells_tgt"]) * len(cube.coord(axis="Z").points),
        #         len(weights["cells_src"]) * len(cube.coord(axis="Z").points),
        #     ),
        # )
        # logger.info("matrix %s", matrix)

        # chunks_tgt = [np.arange(len_tgt) + i * len_tgt for i in range(n_depth)]
        # chunks_src = [np.arange(len_src) + i * len_src for i in range(n_depth)]

        cube.data = da.ma.filled(cube.core_data(), fill_value=np.nan)

        # matrix = da.from_array(
        #     matrix,
        #     chunks=(
        #         int(matrix.shape[0] // (n_workers / c_p_w)),
        #         int(matrix.shape[1] // (n_workers / c_p_w)),
        #     ),
        # )
        # matrix = matrix.map_blocks(sparse.COO)

        matrix_coo = sparse.COO.from_scipy_sparse(matrix)

        row_chunks = max(1, int(matrix_coo.shape[0] // n_workers))

        matrix = da.from_array(
            matrix_coo,
            chunks=(
                row_chunks,
                -1,
            ),
        )

        matrix = matrix.map_blocks(sparse.COO)

        ds = cube.core_data()
        ds = ds.reshape(*ds.shape[:-2], -1)
        ds = da.tensordot(
            matrix,
            ds,
            axes=dims,
        )
        # depth_list = list(np.arange(n_depth))

        # def worker(chunks1, chunks2, i):
        #     matrix_t = da.from_array(
        #         matrix[chunks1, :][:, chunks2],
        #         chunks=(
        #             int(matrix.shape[0] // (n_workers / c_p_w)),
        #             int(matrix.shape[1] // (n_workers / c_p_w)),
        #         ),
        #     )
        #     matrix_t = matrix_t.map_blocks(sparse.COO)
        #     ds = da.tensordot(
        #         # matrix[chunks1, :][:, chunks2],
        #         matrix_t,
        #         cube.core_data()[..., i, :],
        #         axes=dims,
        #     )
        #     return ds
        # results = [worker(chunks_tgt[i], chunks_src[i], i) for i in depth_list]

        # ds = da.stack(results, axis=-1)
        # # logger.info("ds/shape %s, %s", ds, ds.shape)
        # ds = da.moveaxis(ds, 0, -1)
        # ds = ds.reshape(*ds.shape[:-2], -1)
        # ds = da.moveaxis(ds, -1, 0)

        t1 = time.time()
        logger.info("time tensordot %s", t0 - t1)
        t0 = time.time()

        # test_frac = isect["a_frac"].groupby(isect["i_new"]).sum()
        # i_new = da.from_array(
        #     isect.i_new.values,
        #     chunks=isect.shape[0] // (n_workers / 2),
        # )
        # a_frac = da.from_array(
        #     isect.a_frac.values,
        #     chunks=isect.shape[0] // (n_workers / 2),
        # )

        # i_new = isect.i_new.values
        # a_frac = isect.a_frac.values
        # i_new = da.from_array(
        #     i_new,
        #     chunks=i_new.shape[0] // (n_workers / c_p_w),
        # )
        # a_frac = da.from_array(
        #     a_frac,
        #     chunks=a_frac.shape[0] // (n_workers / c_p_w),
        # )
        c1 = np.unique(i_new)
        test_frac = np.bincount(i_new, weights=a_frac)
        # test_idx = np.where(np.isin(np.arange(len(test_frac)), c1))[0]
        # test_frac = test_frac[test_idx]
        test_frac = test_frac[c1]
        # logger.info("test_frac %s", test_frac.shape)
        # print(test_frac[:100])
        idx_test = np.where(test_frac >= a_tol)[0]
        idx_test2 = np.where(test_frac < a_tol)[0]
        idx = c1[idx_test].astype(int)
        idx2 = c1[idx_test2].astype(int)
        # logger.info("idx, idx2 %s, %s", idx.shape, idx2.shape)
        test_frac = da.from_array(
            test_frac[idx_test],
            chunks=test_frac.shape[0] // (n_workers / c_p_w),
        )

        # print(c1[:100])
        # print(idx_test[:100])
        # print(idx_test2[:100])
        # print(idx[:100])
        # print(idx2[:100])
        # isect = dask.dataframe.from_pandas(
        #     isect, npartitions=int(n_workers / c_p_w)
        # )
        # test_frac = isect["a_frac"].groupby(isect["i_new"]).sum().compute()
        # idx_test = np.where(test_frac.values >= a_tol)[0]
        # idx_test2 = np.where(test_frac.values < a_tol)[0]
        # idx = test_frac.index[idx_test].values
        # idx2 = test_frac.index[idx_test2].values
        # # logger.info("idx, idx2 %s, %s", idx.shape, idx2.shape)
        # logger.info("test_frac %s", test_frac)
        # print(test_frac.values[:100])

        # logger.info("test_frac %s", test_frac)
        # print(test_frac.index[:100].values)
        # print(idx_test[:100])
        # print(idx_test2[:100])
        # print(idx[:100])
        # print(idx2[:100])
        # test_frac = da.from_array(
        #     test_frac.values[idx_test],
        #     chunks=test_frac.shape[0] // (n_workers / 2),
        # )

        test_frac = da.reshape(
            test_frac,
            test_frac.shape + (1,) * (len(cube.shape) - 2),
        )
        # logger.info("test_frac %s", test_frac)
        ds[idx, ...] = ds[idx, ...] / test_frac
        ds[idx2, ...] = fill_value
        ds[ds == 0.0] = fill_value

        if len(cube.shape) == 3:
            ds = ds.reshape(
                (
                    len(cube.coord(axis="Z").points),
                    weights["lat_tgt"].shape[0],
                    cube.shape[0],
                ),
            )
            ds = da.moveaxis(ds, -1, 0)
            ds = da.moveaxis(ds, -1, 0)
        else:
            ds = ds.reshape(
                (
                    len(cube.coord(axis="Z").points),
                    weights["lat_tgt"].shape[0],
                ),
            )
            ds = da.moveaxis(ds, -1, 0)

    # elif velocity:
    #     cube.data = da.ma.filled(cube.core_data(), fill_value=0.0)
    #     matrix = da.from_array(
    #         matrix,
    #         chunks=(
    #             int(matrix.shape[0] // (n_workers / c_p_w)),
    #             int(matrix.shape[1] // (n_workers / c_p_w)),
    #         ),
    #     )
    #     matrix = matrix.map_blocks(sparse.COO)
    #     ds = da.tensordot(matrix, cube.core_data(), axes=dims)

    #     t1 = time.time()
    #     logger.info("time tensordot %s", t0 - t1)
    #     t0 = time.time()

    #     ds[ds == 0.0] = fill_value

    else:
        cube.data = da.ma.filled(cube.core_data(), fill_value=np.nan)
        # matrix = da.from_array(
        #     matrix,
        #     chunks=(
        #         int(matrix.shape[0] // (n_workers / c_p_w)),
        #         int(matrix.shape[1] // (n_workers / c_p_w)),
        #     ),
        # )
        # matrix = matrix.map_blocks(sparse.COO)

        matrix_coo = sparse.COO.from_scipy_sparse(matrix)

        row_chunks = max(1, int(matrix_coo.shape[0] // n_workers))

        matrix = da.from_array(
            matrix_coo,
            chunks=(
                row_chunks,
                -1,
            ),
        )

        matrix = matrix.map_blocks(sparse.COO)

        ds = da.tensordot(matrix, cube.core_data(), axes=dims)

        t1 = time.time()
        logger.info("time tensordot %s", t0 - t1)
        t0 = time.time()

        # isect["a_frac"].iloc[np.isin(isect["i_old"], idx_nan).nonzero()] = 0
        test_frac = isect["a_frac"].groupby(isect["i_new"]).sum()
        idx_test = np.where(test_frac.values >= a_tol)[0]
        idx_test2 = np.where(test_frac.values < a_tol)[0]
        idx = test_frac.index[idx_test].values
        idx2 = test_frac.index[idx_test2].values
        # idx2 = (~np.isin(range(ds.shape[0]), idx)).nonzero()[0]
        # logger.info("shape, test_frac %s, %s", test_frac.shape, test_frac)
        logger.info("idx, idx2 %s, %s", idx, idx2)
        test_frac = da.from_array(
            test_frac.values[idx_test],
            chunks=test_frac.shape[0] // (n_workers / c_p_w),
        )
        test_frac = da.reshape(
            test_frac,
            test_frac.shape + (1,) * (len(cube.shape) - 1),
        )
        ds[idx, ...] = ds[idx, ...] / test_frac
        ds[idx2, ...] = fill_value
        ds[ds == 0.0] = fill_value
    # if dims == ((1), (1)):
    #     ds[idx, :] = ds[idx, :] / da.from_array(
    #         test_frac.values[idx_test, np.newaxis]
    #     )
    #     ds[idx2, :] = fill_value #np.nan
    # elif dims == ((1), (2)):
    #     ds[idx, :, :] = ds[idx, :, :] / da.from_array(
    #         test_frac.values[idx_test, np.newaxis, np.newaxis]
    #     )
    #     ds[idx2, :, :] = fill_value #np.nan
    # else:
    #     ds[idx] = ds[idx] / da.from_array(test_frac.values[idx_test])
    #     ds[idx2] = fill_value #np.nan

    t1 = time.time()
    logger.info("time mask a_tol %s", t0 - t1)
    t0 = time.time()
    # with dask.config.set(**{"array.slicing.split_large_chunks": True}):
    logger.info("ds %s", ds)
    ds = ds.reshape(
        np.unique(weights["lat_tgt"]).shape
        + np.unique(weights["lon_tgt"]).shape
        + ds.shape[1:],
        # merge_chunks=False,
    )
    t1 = time.time()
    logger.info("time rechunk ds %s", t0 - t1)
    t0 = time.time()
    coords_add = []
    if len(cube.shape) == 1:
        if len(cube.coords("time")) == 1:
            coords_add.append(cube.coord("time"))
        if len(cube.coords(axis="Z")) == 1:
            coords_add.append(cube.coord(axis="Z"))
        new_cube = iris.cube.Cube(
            da.ma.masked_array(
                da.ma.masked_invalid(ds), fill_value=fill_value
            ),
        )
        if len(coords_add) > 0:
            for coord in coords_add:
                new_cube.add_aux_coord(coord)
    else:
        new_shape = (
            cube.shape[:-1]
            + np.unique(weights["lat_tgt"]).shape
            + np.unique(weights["lon_tgt"]).shape
        )
        # dim_map = (2,3,0,1) if len(cube.shape) == 3 else (2,0,1)
        dim_map = (2, 3, 0, 1) if len(cube.shape) == 3 else (1, 2, 0)
        ds = iris.util.broadcast_to_shape(ds, new_shape, dim_map)
        # TODO: get a better handle on dimension idx (also for lon/lat below)
        for coord in cube.coords():
            if (
                isinstance(coord, iris.coords.DimCoord)
                and coord.shape[0] in ds.shape
            ):
                dim = next(
                    i
                    for i in range(len(ds.shape))
                    if coord.shape[0] == ds.shape[i]
                )
                coords_add.append((coord, dim))

        new_cube = iris.cube.Cube(
            da.ma.masked_array(
                da.ma.masked_invalid(ds),
                fill_value=fill_value,
            ),
            dim_coords_and_dims=coords_add,
        )
    t1 = time.time()
    logger.info("time create new_cube %s", t0 - t1)
    t0 = time.time()
    lat_coord = tgt_cube.coord("latitude")
    lat_coord.points = np.round(lat_coord.points, 4)
    lat_coord.bounds = np.round(lat_coord.bounds, 4)
    lon_coord = tgt_cube.coord("longitude")
    lon_coord.points = np.round(lon_coord.points, 4)
    lon_coord.bounds = np.round(lon_coord.bounds, 4)

    lon_dim = next(
        i for i in range(len(ds.shape)) if lon_coord.shape[0] == ds.shape[i]
    )
    lat_dim = next(
        i for i in range(len(ds.shape)) if lat_coord.shape[0] == ds.shape[i]
    )

    new_cube.add_dim_coord(lat_coord, lat_dim)
    new_cube.add_dim_coord(lon_coord, lon_dim)

    new_cube.standard_name = cube.standard_name
    new_cube.long_name = cube.long_name
    new_cube.var_name = cube.var_name
    new_cube.units = cube.units
    t1 = time.time()
    logger.info("time end %s", t0 - t1)
    return new_cube
