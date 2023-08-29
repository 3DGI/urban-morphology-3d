"""Module to manipulate geometry of pyvista meshes"""

import numpy as np
import pyvista as pv
from helpers.geometry import plane_params, project_mesh, to_3d
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
from shapely import intersects, Polygon

def get_points_of_type(mesh, surface_type):
    """Returns the points that belong to the given surface type"""

    if not "semantics" in mesh.cell_data:
        return []
    
    idxs = [s == surface_type for s in mesh.cell_data["semantics"]]

    points = np.array([mesh.cell_points(i) for i in range(mesh.number_of_cells)], dtype=object)

    if all([i == False for i in idxs]):
        return []

    return np.vstack(points[idxs])

def move_to_origin(mesh):
    """Moves the object to the origin"""
    pts = mesh.points
    t = np.min(pts, axis=0)
    mesh.points = mesh.points - t

    return mesh, t

def extrude(shape, min, max):
    """Create a pyvista mesh from a polygon"""

    points = np.array([[p[0], p[1], min] for p in shape.boundary.coords])
    mesh = pv.PolyData(points).delaunay_2d()

    if min == max:
        return mesh

    # Transform to 0, 0, 0 to avoid precision issues
    pts = mesh.points
    t = np.mean(pts, axis=0)
    mesh.points = mesh.points - t
    
    mesh = mesh.extrude([0.0, 0.0, max - min], capping=True)
    
    # Transform back to origina coords
    # mesh.points = mesh.points + t

    mesh = mesh.clean().triangulate()

    return mesh

def area_by_surface(mesh, sloped_angle_threshold=3, tri_mesh=None):
    """Compute the area per semantic surface"""

    sloped_threshold = np.cos(np.radians(sloped_angle_threshold))

    area = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurfaceFlat": 0,
        "RoofSurfaceSloped": 0
    }

    point_count = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurface": 0
    }

    surface_count = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurface": 0
    }

    # Compute the triangulated surfaces to fix issues with areas
    if tri_mesh is None:
        tri_mesh = mesh.triangulate()

    if "semantics" in mesh.cell_data:
        # Compute area per surface type
        sized = tri_mesh.compute_cell_sizes()
        surface_areas = sized.cell_data["Area"]

        points_per_cell = np.array([mesh.cell_n_points(i) for i in range(mesh.number_of_cells)])

        for surface_type in ["GroundSurface", "WallSurface", "RoofSurface"]:
            triangle_idxs_mask = [s == surface_type for s in tri_mesh.cell_data["semantics"]]
            triangle_idxs = [i for i,s in enumerate(tri_mesh.cell_data["semantics"]) if s == surface_type]

            if surface_type == "RoofSurface":
                for idx in triangle_idxs:
                    if sized.cell_normals[idx].dot([0,0,1]) < sloped_threshold:
                        area["RoofSurfaceSloped"] += surface_areas[idx]
                    else:
                        area["RoofSurfaceFlat"] += surface_areas[idx]
            else:
                area[surface_type] = sum(surface_areas[triangle_idxs_mask])

            face_idxs = [s == surface_type for s in mesh.cell_data["semantics"]]

            point_count[surface_type] = sum(points_per_cell[face_idxs])
            surface_count[surface_type] = sum(face_idxs)
    
    return area, point_count, surface_count

def face_planes(mesh):
    """Return the params of all planes in a given mesh"""

    return [plane_params(mesh.face_normals[i], mesh.cell_points(i)[0])
            for i in range(mesh.n_cells)]

def cluster_meshes(meshes, angle_degree_threshold=5, dist_threshold=0.5, old_cluster_method=True):
    """Clusters the faces of the given meshes"""
    
    n_meshes = len(meshes)
    
    # Compute the "absolute" plane params for every face of the two meshes
    planes = [face_planes(mesh) for mesh in meshes]
    mesh_ids = [[m for _ in range(meshes[m].n_cells)] for m in range(n_meshes)]
    
    # convert to cosine distance value
    # cos_distance = 1 - cos_similarity
    # angle_rad = arccos(cos_similarity)
    # angle_deg = angle_rad * (180/pi)
    cos_dist_thres = 1 - np.cos((np.pi/180) * angle_degree_threshold)
    # Find the common planes between the two faces
    all_planes = np.concatenate(planes)
    if old_cluster_method:
        all_labels, n_clusters = cluster_faces_simple(all_planes)
    else:
        all_labels, n_clusters = cluster_faces_alternative(all_planes, cos_dist_thres, dist_threshold)
    areas = []
    
    labels = np.array_split(all_labels, [meshes[m].n_cells for m in range(n_meshes - 1)])
    
    return labels, n_clusters

def cluster_faces_simple(data, threshold=0.1):
    """Clusters the given planes"""
    # we can delete the third column because it is all 0's for vertical planes
    ndata = np.delete(data, 2, 1)

    # flip normals so that they can not be pointing in opposite direction for same plane
    neg_x = ndata[:,0] < 0
    ndata[neg_x,:] = ndata[neg_x,:] * -1

    dist_mat = distance_matrix(ndata, ndata)
    # dm2 = distance_matrix(ndata, -ndata)
    # dist_mat = np.minimum(dm1, dm2)

    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=threshold,
                                         metric='precomputed',
                                         linkage='average').fit(dist_mat)
    
    return clustering.labels_, clustering.n_clusters_

def cluster_faces_alternative(data, angle_threshold=0.005, dist_threshold=0.2):
    """Clusters the given planes"""
    def groupby(a, clusterids):
        # Get argsort indices, to be used to sort a and clusterids in the next steps
        sidx = clusterids.argsort(kind='mergesort')
        a_sorted = a[sidx]
        clusterids_sorted = clusterids[sidx]

        # Get the group limit indices (start, stop of groups)
        cut_idx = np.flatnonzero(np.r_[True,clusterids_sorted[1:] != clusterids_sorted[:-1],True])

        # Split input array with those start, stop ones
        out = [a_sorted[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])]
        return out, sidx

    ndata = np.array(data)

    # original method
    # dm1 = distance_matrix(ndata, ndata)
    # dm2 = distance_matrix(ndata, -ndata)

    # dist_mat = np.minimum(dm1, dm2)
    # clustering = AgglomerativeClustering(n_clusters=None,
    #                                      distance_threshold=threshold,
    #                                      affinity='precomputed',
    #                                      linkage='average').fit(dist_mat)

    # new method - angle clustering
    # pl_abc = ndata
    angle_clustering = AgglomerativeClustering(n_clusters=None,
                                         metric='cosine',
                                         distance_threshold=angle_threshold,
                                         linkage='average').fit(ndata[:,:3])
    # group angle clusters
    angle_clusters, remap = groupby(ndata[:,3:], angle_clustering.labels_)

    # get dist clusters for each angle cluster
    labels_ = np.empty(0, dtype=int)
    min_label = 0
    for angle_cluster in angle_clusters:
        if angle_cluster.size == 1:
            labels_ = np.hstack((labels_, min_label))
            min_label += 1
        else:
            dist_clustering = AgglomerativeClustering(n_clusters=None,
                                                metric='euclidean',
                                                distance_threshold=dist_threshold,
                                                linkage='average').fit(angle_cluster)
            labels_ = np.hstack((labels_, dist_clustering.labels_ + min_label))
            min_label = labels_.max()+1
    
    # re order back to input data order
    n_planes = ndata.shape[0]
    labels = np.empty(n_planes, dtype=int)
    labels[remap] = labels_

    n_clusters = (np.bincount(labels)!=0).sum()
    return labels, n_clusters

def intersect_surfaces(meshes, onlywalls=True):
    """Return the intersection between the surfaces of multiple meshes. Note first mesh is the one we are computing the surfaces for, following ones are neighbors"""

    def get_area_from_ring(areas, area, geom, normal, origin, subtract=False):
        pts = to_3d(geom.coords, normal, origin)
        common_mesh = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))))
        if subtract:
            common_mesh["area"] = [-area]
        else:
            common_mesh["area"] = [area]
        common_mesh["pts"]=pts
        areas.append(common_mesh)

    def get_area_from_polygon(areas, geom, normal, origin):
        # polygon with holes:
        if geom.boundary.type == 'MultiLineString':
            get_area_from_ring(areas, geom.area, geom.boundary.geoms[0], normal, origin)
            holes = [g for g in geom.boundary.geoms][1:]
            for hole in holes:
                get_area_from_ring(areas, 0, hole, normal, origin, subtract=True)
        # polygon without holes:
        elif geom.boundary.type == 'LineString':
            get_area_from_ring(areas, geom.area, geom.boundary, normal, origin)
    
    n_meshes = len(meshes)

    meshes_to_cluster = []
    if onlywalls:
        for mesh in meshes:
            meshes_to_cluster.append( mesh.remove_cells( [s != 'WallSurface' for s in mesh.cell_data["semantics"]], inplace=False ) )
    else:
        meshes_to_cluster = meshes
    
    areas = []
    
    labels, n_clusters = cluster_meshes(meshes_to_cluster)
    
    for plane in range(n_clusters):
        # For every common plane, extract the faces that belong to it
        idxs = [[i for i, p in enumerate(labels[m]) if p == plane] for m in range(n_meshes)]
                
        if any([len(idx) == 0 for idx in idxs]):
            continue
        
        msurfaces = [mesh.extract_cells(idxs[i]).extract_surface() for i, mesh in enumerate(meshes_to_cluster)]
                
        # Set the normal and origin point for a plane to project the faces
        origin = msurfaces[0].clean().points[0]
        normal = msurfaces[0].face_normals[0]
        
        if np.linalg.norm(normal) == 0: continue

        # Create the two 2D polygons by projecting the faces
        polys = [project_mesh(msurface, normal, origin) for msurface in msurfaces]
        
        # Intersect the 2D polygons
        inter = Polygon()
        poly_0 = polys[0]
        for i in range(1, len(polys)):
            if intersects(poly_0, polys[i]):
                inter = inter.union(poly_0.intersection(polys[i]))

        if len(polys) != 2: 
            print(len(polys))
        
        if inter.area > 0.001:
            if inter.type == "MultiPolygon" or inter.type == "GeometryCollection":
                for geom in inter.geoms:
                    if geom.type != "Polygon":
                        continue
                    get_area_from_polygon(areas, geom, normal, origin)
            elif inter.type == "Polygon":
                get_area_from_polygon(areas, inter, normal, origin)
    
    return areas
