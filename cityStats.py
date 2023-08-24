import json
import math
import csv
import gzip
import pathlib

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas
import rtree.index
import scipy.spatial as ss
from pymeshfix import MeshFix
from tqdm import tqdm
from shapely.geometry import Polygon, box
from shapely import MultiPolygon
from pgutils import PostgresConnection
from psycopg import sql

import cityjson
import geometry

def get_bearings(values, num_bins, weights):
    """Divides the values depending on the bins"""

    n = num_bins * 2

    bins = np.arange(n + 1) * 360 / n

    count, bin_edges = np.histogram(values, bins=bins, weights=weights)

    # move last bin to front, so eg 0.01° and 359.99° will be binned together
    count = np.roll(count, 1)
    bin_counts = count[::2] + count[1::2]

    # because we merged the bins, their edges are now only every other one
    bin_edges = bin_edges[range(0, len(bin_edges), 2)]

    return bin_counts, bin_edges

def get_wall_bearings(dataset, num_bins):
    """Returns the bearings of the azimuth angle of the normals for vertical
    surfaces of a dataset"""

    normals = dataset.face_normals

    if "semantics" in dataset.cell_data:
        wall_idxs = [s == "WallSurface" for s in dataset.cell_data["semantics"]]
    else:
        wall_idxs = [n[2] == 0 for n in normals]

    normals = normals[wall_idxs]

    azimuth = [point_azimuth(n) for n in normals]

    sized = dataset.compute_cell_sizes()
    surface_areas = sized.cell_data["Area"][wall_idxs]

    return get_bearings(azimuth, num_bins, surface_areas)

def get_roof_bearings(dataset, num_bins):
    """Returns the bearings of the (vertical surfaces) of a dataset"""

    normals = dataset.face_normals

    if "semantics" in dataset.cell_data:
        roof_idxs = [s == "RoofSurface" for s in dataset.cell_data["semantics"]]
    else:
        roof_idxs = [n[2] > 0 for n in normals]

    normals = normals[roof_idxs]

    xz_angle = [azimuth(n[0], n[2]) for n in normals]
    yz_angle = [azimuth(n[1], n[2]) for n in normals]

    sized = dataset.compute_cell_sizes()
    surface_areas = sized.cell_data["Area"][roof_idxs]

    xz_counts, bin_edges = get_bearings(xz_angle, num_bins, surface_areas)
    yz_counts, bin_edges = get_bearings(yz_angle, num_bins, surface_areas)

    return xz_counts, yz_counts, bin_edges

def orientation_plot(
    bin_counts,
    bin_edges,
    num_bins=36,
    title=None,
    title_y=1.05,
    title_font=None,
    show=False
):
    if title_font is None:
        title_font = {"family": "DejaVu Sans", "size": 12, "weight": "bold"}

    width = 2 * np.pi / num_bins

    positions = np.radians(bin_edges[:-1])

    radius = bin_counts / bin_counts.sum()

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_ylim(top=radius.max())

    # configure the y-ticks and remove their labels
    ax.set_yticks(np.linspace(0, radius.max(), 5))
    ax.set_yticklabels(labels="")

    # configure the x-ticks and their labels
    xticklabels = ["N", "", "E", "", "S", "", "W", ""]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=xticklabels)
    ax.tick_params(axis="x", which="major", pad=-2)

    # draw the bars
    ax.bar(
        positions,
        height=radius,
        width=width,
        align="center",
        bottom=0,
        zorder=2
    )

    if title:
        ax.set_title(title, y=title_y, fontdict=title_font)

    if show:
        plt.show()
    
    return plt

def get_surface_plot(
    dataset,
    num_bins=36,
    title=None,
    title_y=1.05,
    title_font=None
):
    """Returns a plot for the surface normals of a polyData"""
    
    bin_counts, bin_edges = get_wall_bearings(dataset, num_bins)

    return orientation_plot(bin_counts, bin_edges)
    
def azimuth(dx, dy):
    """Returns the azimuth angle for the given coordinates"""
    
    return (math.atan2(dx, dy) * 180 / np.pi) % 360

def point_azimuth(p):
    """Returns the azimuth angle of the given point"""

    return azimuth(p[0], p[1])

def point_zenith(p):
    """Return the zenith angle of the given 3d point"""

    z = [0.0, 0.0, 1.0]

    cosine_angle = np.dot(p, z) / (np.linalg.norm(p) * np.linalg.norm(z))
    angle = np.arccos(cosine_angle)

    return (angle * 180 / np.pi) % 360

def compute_stats(values, percentile = 90, percentage = 75):
    """
    Returns the stats (mean, median, max, min, range etc.) for a set of values.
    """
    hDic = {'Mean': np.mean(values), 'Median': np.median(values),
    'Max': max(values), 'Min': min(values), 'Range': (max(values) - min(values)),
    'Std': np.std(values)}
    m = max([values.count(a) for a in values])
    if percentile:
        hDic['Percentile'] = np.percentile(values, percentile)
    if percentage:
        hDic['Percentage'] = (percentage/100.0) * hDic['Range'] + hDic['Min']
    if m>1:
        hDic['ModeStatus'] = 'Y'
        modeCount = [x for x in values if values.count(x) == m][0]
        hDic['Mode'] = modeCount
    else:
        hDic['ModeStatus'] = 'N'
        hDic['Mode'] = np.mean(values)
    return hDic

def add_value(dict, key, value):
    """Does dict[key] = dict[key] + value"""

    if key in dict:
        dict[key] = dict[key] + value
    else:
        dict[key] = value

def convexhull_volume(points):
    """Returns the volume of the convex hull"""

    try:
        return ss.ConvexHull(points).volume
    except:
        return 0

def boundingbox_volume(points):
    """Returns the volume of the bounding box"""
    
    minx = min(p[0] for p in points)
    maxx = max(p[0] for p in points)
    miny = min(p[1] for p in points)
    maxy = max(p[1] for p in points)
    minz = min(p[2] for p in points)
    maxz = max(p[2] for p in points)

    return (maxx - minx) * (maxy - miny) * (maxz - minz)

def get_errors_from_report(report, objid, cm):
    """Return the report for the feature of the given obj"""

    if not "features" in report:
        return []
    
    fid = objid

    obj = cm["CityObjects"][objid]
    primidx = 0

    if not "geometry" in obj or len(obj["geometry"]) == 0:
        return []

    if "parents" in obj:
        parid = obj["parents"][0]

        primidx = cm["CityObjects"][parid]["children"].index(objid)
        fid = parid

    for f in report["features"]:
        if f["id"] == fid:
            if "errors" in f["primitives"][primidx]:
                return list(map(lambda e: e["code"], f["primitives"][primidx]["errors"]))
            else:
                return []

    return []

def validate_report(report, cm):
    """Returns true if the report is actually for this file"""

    # TODO: Actually validate the report and that it corresponds to this cm
    return True

def tree_generator_function(building_meshes):
    for i, (bid, mesh) in enumerate(building_meshes.items()):
        xmin, ymin, zmin = np.min(mesh.points, axis=0)
        xmax, ymax, zmax = np.max(mesh.points, axis=0)
        yield (i, (xmin, ymin, zmin, xmax, ymax, zmax), bid)

def get_neighbours(building_meshes, bid, r):
    """Return the neighbours of the given building"""
    
    xmin, ymin, zmin = np.min(building_meshes[bid].points, axis=0)
    xmax, ymax, zmax = np.max(building_meshes[bid].points, axis=0)
    bids = [n.object
            for n in r.intersection((xmin,
                                    ymin,
                                    zmin,
                                    xmax,
                                    ymax,
                                    zmax),
                                    objects=True)
            if n.object != bid]

    # if len(bids) == 0:
    #     bids = [n.object for n in r.nearest((xmin, ymin, zmin, xmax, ymax, zmax), 5, objects=True) if n.object != bid]

    return bids

class StatValuesBuilder:

    def __init__(self, values, indices_list) -> None:
        self.__values = values
        self.__indices_list = indices_list

    def compute_index(self, index_name):
        """Returns True if the given index is supposed to be computed"""

        return self.__indices_list is None or index_name in self.__indices_list
    
    def add_index(self, index_name, index_func):
        """Adds the given index value to the dict"""

        if self.compute_index(index_name):
            self.__values[index_name] = index_func() 
        else:
            self.__values[index_name] = "NC"

def filter_lod(cm, lod='2.2'):
    for co_id in cm["CityObjects"]:
        co = cm["CityObjects"][co_id]

        new_geom = []

        for geom in co["geometry"]:
            if str(geom["lod"]) == str(lod):
                new_geom.append(geom)
        
        co["geometry"] = new_geom

def process_building(building,
                     building_id,
                     filter,
                     repair,
                     plot_buildings,
                     vertices,
                     building_meshes,
                     neighbour_ids=[],
                     custom_indices=None,
                     goffset=None):

    if not filter is None and filter != building_id:
        return building_id, None

    # TODO: Add options for all skip conditions below

    # Skip if type is not Building or Building part
    if not building["type"] in ["Building", "BuildingPart"]:
        return building_id, None

    # Skip if no geometry
    if not "geometry" in building or len(building["geometry"]) == 0:
        return building_id, None

    geom = building["geometry"][0]
    
    mesh = cityjson.to_polydata(geom, vertices).clean()

    try:
        tri_mesh = building_meshes[building_id]
    except:
        print(f"{building_id} geometry parsing crashed! Omitting...")
        return building_id, {"type": building["type"]}

    if plot_buildings:
        print(f"Plotting {building_id}")
        tri_mesh.plot(show_grid=True)

    t_origin = np.min(tri_mesh.points, axis=0)

    # get_surface_plot(dataset, title=building_id)

    # bin_count, bin_edges = get_wall_bearings(mesh, 36)

    # xzc, yzc, be = get_roof_bearings(mesh, 36)
    # plot_orientations(xzc, be, title=f"XZ orientation [{building_id}]")
    # plot_orientations(yzc, be, title=f"YZ orientation [{building_id}]")

    # total_xy = total_xy + bin_count
    # total_xz = total_xz + xzc
    # total_yz = total_yz + yzc

    if repair:
        mfix = MeshFix(tri_mesh)
        mfix.repair()

        fixed = mfix.mesh
    else:
        fixed = tri_mesh

    # holes = mfix.extract_holes()

    # plotter = pv.Plotter()
    # plotter.add_mesh(dataset, color=True)
    # plotter.add_mesh(holes, color='r', line_width=5)
    # plotter.enable_eye_dome_lighting() # helps depth perception
    # _ = plotter.show()

    # points = cityjson.get_points(geom, vertices)

    # aabb_volume = boundingbox_volume(points)

    # ch_volume = convexhull_volume(points)

    area, point_count, surface_count = geometry.area_by_surface(mesh)

    if "semantics" in geom:
        roof_points = geometry.get_points_of_type(mesh, "RoofSurface")
        ground_points = geometry.get_points_of_type(mesh, "GroundSurface")
    else:
        roof_points = []
        ground_points = []

    if len(roof_points) == 0:
        height_stats = compute_stats([0])
        ground_z = 0
    else:
        height_stats = compute_stats([v[2] for v in roof_points])
        if len(ground_points) > 0:
            ground_z = min([v[2] for v in ground_points])
        else:
            ground_z = mesh.bounds[4]
    
    if len(ground_points) > 0:
        shape = cityjson.to_shapely(geom, vertices)
    else:
        shape = cityjson.to_shapely(geom, vertices, ground_only=False)

    # obb_2d = cityjson.to_shapely(geom, vertices, ground_only=False).minimum_rotated_rectangle

    # Compute OBB with shapely
    # min_z = np.min(mesh.clean().points[:, 2])
    # max_z = np.max(mesh.clean().points[:, 2])
    # obb = geometry.extrude(obb_2d, min_z, max_z)

    # Get the dimensions of the 2D oriented bounding box
    # S, L = si.get_box_dimensions(obb_2d)

    values = {
        # "type": building["type"],
        # "lod": geom["lod"],
        # "point_count": len(points),
        # "unique_point_count": fixed.n_points,
        # "surface_count": len(cityjson.get_surface_boundaries(geom)),
        # "actual_volume": fixed.volume,
        # "convex_hull_volume": ch_volume,
        # "obb_volume": obb.volume,
        # "aabb_volume": aabb_volume,
        # "footprint_perimeter": shape.length,
        # "obb_width": S,
        # "obb_length": L,
        # "surface_area": mesh.area,
        "area_ground": area["GroundSurface"],
        # "total_wall_area": area["WallSurface"],
        "area_roof_flat": area["RoofSurfaceFlat"],
        "area_roof_sloped": area["RoofSurfaceSloped"],
        # "ground_point_count": point_count["GroundSurface"],
        # "wall_point_count": point_count["WallSurface"],
        # "roof_point_count": point_count["RoofSurface"],
        # "ground_surface-count": surface_count["GroundSurface"],
        # "wall_surface_count": surface_count["WallSurface"],
        # "roof_surface_count": surface_count["RoofSurface"],
        # "max_Z": height_stats["Max"],
        # "min_Z": height_stats["Min"],
        # "height_range": height_stats["Range"],
        # "mean_Z": height_stats["Mean"],
        # "median_Z": height_stats["Median"],
        # "std_Z": height_stats["Std"],
        # "mode_Z": height_stats["Mode"] if height_stats["ModeStatus"] == "Y" else "NA",
        # "ground_Z": ground_z,
        # "orientation_values": str(bin_count),
        # "orientation_edges": str(bin_edges),
        # "errors": str(errors),
        # "valid": len(errors) == 0,
        # "hole_count": tri_mesh.n_open_edges,
        # "geometry": shape,
        # "neighbours": ";".join(neighbour_ids)
    }

    if custom_indices is None or len(custom_indices) > 0:

        shared_area = 0
        shared_polys = []

        if len(neighbour_ids) > 0:
            # Get neighbour meshes
            n_meshes = [building_meshes[nid]
                        for nid in neighbour_ids]
            
            # Compute shared walls

            # need to translate to origin to make the clustering work well (both quality of results and performance)
            fixed.points -= t_origin
            for neighbour in n_meshes:
                neighbour.points -= t_origin

            walls = np.hstack([geometry.intersect_surfaces([fixed, neighbour])
                            for neighbour in n_meshes])
            
            shared_area = sum([wall["area"][0] for wall in walls])
            shared_polys = [Polygon(wall["pts"]+(t_origin+goffset)) for wall in walls]
            # undo translate to not mess up future calculations with these geometries
            fixed.points += t_origin
            for neighbour in n_meshes:
                neighbour.points += t_origin

            # Find the closest distance
            # for mesh in n_meshes:
            #     mesh.compute_implicit_distance(fixed, inplace=True)
                            
            #     closest_distance = min(closest_distance, np.min(mesh["implicit_distance"]))
            
            # closest_distance = max(closest_distance, 0)
        # else:
            # closest_distance = "NA"

        builder = StatValuesBuilder(values, custom_indices)

        # builder.add_index("2d_grid_point_count", lambda: len(si.create_grid_2d(shape, density=density_2d)))
        # builder.add_index("3d_grid_point_count", lambda: len(grid))

        # builder.add_index("circularity_2d", lambda: si.circularity(shape))
        # builder.add_index("hemisphericality_3d", lambda: si.hemisphericality(fixed))
        # builder.add_index("convexity_2d", lambda: shape.area / shape.convex_hull.area)
        # builder.add_index("convexity_3d", lambda: fixed.volume / ch_volume)
        # builder.add_index("convexity_3d", lambda: fixed.volume / ch_volume)
        # builder.add_index("fractality_2d", lambda: si.fractality_2d(shape))
        # builder.add_index("fractality_3d", lambda: si.fractality_3d(fixed))
        # builder.add_index("rectangularity_2d", lambda: shape.area / shape.minimum_rotated_rectangle.area)
        # builder.add_index("rectangularity_3d", lambda: fixed.volume / obb.volume)
        # builder.add_index("squareness_2d", lambda: si.squareness(shape))
        # builder.add_index("cubeness_3d", lambda: si.cubeness(fixed))
        # builder.add_index("horizontal_elongation", lambda: si.elongation(S, L))
        # builder.add_index("min_vertical_elongation", lambda: si.elongation(L, height_stats["Max"]))
        # builder.add_index("max_vertical_elongation", lambda: si.elongation(S, height_stats["Max"]))
        # builder.add_index("form_factor_3D", lambda: shape.area / math.pow(fixed.volume, 2/3))
        # builder.add_index("equivalent_rectangularity_index_2d", lambda: si.equivalent_rectangular_index(shape))
        # builder.add_index("equivalent_prism_index_3d", lambda: si.equivalent_prism_index(fixed, obb))
        # builder.add_index("proximity_index_2d_", lambda: si.proximity_2d(shape, density=density_2d))
        # builder.add_index("proximity_index_3d", lambda: si.proximity_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
        # builder.add_index("exchange_index_2d", lambda: si.exchange_2d(shape))
        # builder.add_index("exchange_index_3d", lambda: si.exchange_3d(tri_mesh, density=density_3d))
        # builder.add_index("spin_index_2d", lambda: si.spin_2d(shape, density=density_2d))
        # builder.add_index("spin_index_3d", lambda: si.spin_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
        # builder.add_index("perimeter_index_2d", lambda: si.perimeter_index(shape))
        # builder.add_index("circumference_index_3d", lambda: si.circumference_index_3d(tri_mesh))
        # builder.add_index("depth_index_2d", lambda: si.depth_2d(shape, density=density_2d))
        # builder.add_index("depth_index_3d", lambda: si.depth_3d(tri_mesh, density=density_3d) if len(grid) > 2 else "NA")
        # builder.add_index("girth_index_2d", lambda: si.girth_2d(shape))
        # builder.add_index("girth_index_3d", lambda: si.girth_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
        # builder.add_index("dispersion_index_2d", lambda: si.dispersion_2d(shape, density=density_2d))
        # builder.add_index("dispersion_index_3d", lambda: si.dispersion_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
        # builder.add_index("range_index_2d", lambda: si.range_2d(shape))
        # builder.add_index("range_index_3d", lambda: si.range_3d(tri_mesh))
        # builder.add_index("roughness_index_2d", lambda: si.roughness_index_2d(shape, density=density_2d))
        # builder.add_index("roughness_index_3d", lambda: si.roughness_index_3d(tri_mesh, grid, density_2d) if len(grid) > 2 else "NA")
        builder.add_index("area_shared_wall", lambda: shared_area)
        builder.add_index("area_exterior_wall", lambda: area["WallSurface"] - shared_area)
        builder.add_index("shared_wall_geometry", lambda: ";".join([poly.wkt for poly in shared_polys]))
        # builder.add_index("closest_distance", lambda: closest_distance)

    return building_id, values

class CityModel:
    def __init__(self, cm) -> None:
        filter_lod(cm)
        self.cm = cm
        
        if "transform" in cm:
            s = cm["transform"]["scale"]
            t = cm["transform"]["translate"]
            self.verts = [[v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]
                    for v in cm["vertices"]]
        else:
            self.verts = cm["vertices"]
        self.vertices = np.array(self.verts)

# Assume semantic surfaces
@click.command()
@click.argument("inputs", nargs=-1, type=str)
@click.option('-o', '--output', type=click.Path(resolve_path=True,
                                                path_type=pathlib.Path))
@click.option('-g', '--gpkg')
@click.option('-v', '--val3dity-report', type=click.File("rb"))
@click.option('-f', '--filter')
@click.option('-r', '--repair', flag_value=True)
@click.option('-p', '--plot-buildings', flag_value=True)
@click.option('--without-indices', flag_value=True)
@click.option('-s', '--single-threaded', flag_value=True)
@click.option('-b', '--break-on-error', flag_value=True)
@click.option('-j', '--jobs', default=1)
@click.option('-dsn')
@click.option('--precision', default=2)
# @click.option('--density-2d', default=1.0)
# @click.option('--density-3d', default=1.0)
def main(inputs,
         output,
         gpkg,
         val3dity_report,
         filter,
         repair,
         plot_buildings,
         without_indices,
         single_threaded,
         break_on_error,
         jobs,
         dsn,
         precision):
    cms = []

    # Check if we can connect to Postgres before we would start processing anything
    conn = PostgresConnection(dsn=dsn)

    for input in inputs:
        with gzip.open(input, 'r') as fin:
            cms.append(CityModel(json.loads(fin.read().decode('utf-8'))))

    # we assume the first tile is the current tile we need to compute shared walls for
    active_tile_name = pathlib.Path(inputs[0]).name.replace(".city.json.gz", "").replace("-", "/")
    
    ge = cms[0].cm['metadata']['geographicalExtent']
    tile_bb = box(ge[0], ge[1], ge[3], ge[4])
    t_origin = [(p[0], p[1], 0) for p in tile_bb.centroid.coords]

    building_meshes = {}

    # convert geometries to polydata and select from the neighbour tiles only the ones that intersect with current tile boundary
    for i, cm in enumerate(cms):
        for coid, co in cm.cm['CityObjects'].items():
            if co['type'] == "BuildingPart":
                if i>0:
                    minx, maxx, miny, maxy, _, _ = cityjson.get_bbox(co['geometry'][0], cm.verts)
                    if not tile_bb.intersects( box(minx, miny, maxx, maxy) ):
                        continue
                mesh = cityjson.to_triangulated_polydata(co['geometry'][0], cm.vertices).clean()
                mesh.points -= t_origin
                building_meshes[coid] = mesh

    if len(building_meshes) == 0:
        click.echo("Aborting, no building meshes found...")
        return

    # Build the index of the city model
    p = rtree.index.Property()
    p.dimension = 3
    r = rtree.index.Index(tree_generator_function(building_meshes), properties=p)

    stats = {}

    if single_threaded or jobs == 1:
        for obj in tqdm(cms[0].cm["CityObjects"]):
            if cms[0].cm["CityObjects"][obj]["type"] == "BuildingPart":
                neighbour_ids = get_neighbours(building_meshes, obj, r)

                indices_list = [] if without_indices else None
                
                try:
                    obj, vals = process_building(cms[0].cm["CityObjects"][obj],
                                    obj,
                                    filter,
                                    repair,
                                    plot_buildings,
                                    cms[0].vertices,
                                    building_meshes,
                                    neighbour_ids,
                                    indices_list,
                                    goffset=t_origin)
                    if not vals is None:
                        parent = cms[0].cm["CityObjects"][obj]["parents"][0]
                        for key, val in cms[0].cm["CityObjects"][parent]["attributes"].items():
                            if key in ["identificatie", "status", "oorspronkelijkbouwjaar"]:
                                vals[key] = val
                        stats[obj] = vals
                except Exception as e:
                    print(f"Problem with {obj}")
                    if break_on_error:
                        raise e

    else:
        from concurrent.futures import ProcessPoolExecutor

        num_objs = len(cms[0].cm["CityObjects"])
        num_cores = jobs

        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            with tqdm(total=num_objs) as progress:
                futures = []

                for obj in cms[0].cm["CityObjects"]:

                    if cms[0].cm["CityObjects"][obj]["type"] == "BuildingPart":

                        neighbour_ids = get_neighbours(building_meshes, obj, r)

                        indices_list = [] if without_indices else None

                        future = pool.submit(process_building,
                                            cms[0].cm["CityObjects"][obj],
                                            obj,
                                            filter,
                                            repair,
                                            plot_buildings,
                                            cms[0].vertices,
                                            building_meshes,
                                            neighbour_ids,
                                            indices_list,
                                            goffset=t_origin)
                        future.add_done_callback(lambda p: progress.update())
                        futures.append(future)
                
                results = []
                for future in futures:
                    try:
                        obj, vals = future.result()
                        if not vals is None:
                            parent = cms[0].cm["CityObjects"][obj]["parents"][0]
                            for key, val in cms[0].cm["CityObjects"][parent]["attributes"].items():
                                if key in ["identificatie", "status", "oorspronkelijkbouwjaar"]:
                                    vals[key] = val 
                    except Exception as e:
                        print(f"Problem with {obj}")
                        if break_on_error:
                            raise e

    # orientation_plot(total_xy, bin_edges, title="Orientation plot")
    # orientation_plot(total_xz, bin_edges, title="XZ plot")
    # orientation_plot(total_yz, bin_edges, title="YZ plot")

    cm_ids = sql.Literal(list(cms[0].cm["CityObjects"].keys()))
    query = sql.SQL(
        """
        SELECT p.identificatie::text    AS identificatie
             , st_area(p.geometrie)     AS oppervlakte_bag_geometrie
        FROM lvbag.pandactueelbestaand p
        WHERE p.identificatie = ANY({cm_ids});
        """
    ).format(cm_ids=cm_ids)

    click.echo("Building data frame...")
    df = pd.DataFrame.from_dict(stats, orient="index").round(precision)
    df.index.name = "id"
    df["identificatie"] = df["identificatie"].astype(str)

    click.echo("Getting BAG footprint areas...")
    df = df.join(other=pd.DataFrame
                         .from_records(conn.get_dict(query))
                         .set_index("identificatie")
                         .round(precision),
                 on="identificatie", how="left")
    df['tile'] = active_tile_name

    if output is None:
        print(df)
    else:
        click.echo(f"Writing shared walls output to {output}...")
        df.to_csv(output, sep=",", quoting=csv.QUOTE_ALL)
    
    if not gpkg is None:
        gdf = geopandas.GeoDataFrame(df, geometry="geometry")
        gdf.to_file(gpkg, driver="GPKG")

    click.echo("Done")


if __name__ == "__main__":
    main()
