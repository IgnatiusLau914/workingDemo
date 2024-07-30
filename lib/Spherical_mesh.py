import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

###################################
### spherical mesh
###################################
def rotation_matrix(angle_x, angle_y, angle_z):
    """Create a composite rotation matrix for rotating around x, y, and z axes."""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx

def create_icosahedron(angle_x=0, angle_y=0, angle_z=0):
    """Create an icosahedron with 12 vertices and 20 faces and rotate it."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1],
    ])
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    faces = np.array([
        [0, 11,  5], [0,  5,  1], [0,  1,  7], [0,  7, 10], [0, 10, 11],
        [1,  5,  9], [5, 11,  4], [11, 10,  2], [10,  7,  6], [7,  1,  8],
        [3,  9,  4], [3,  4,  2], [3,  2,  6], [3,  6,  8], [3,  8,  9],
        [4,  9,  5], [2,  4, 11], [6,  2, 10], [8,  6,  7], [9,  8,  1],
    ])

    # Apply rotation
    R = rotation_matrix(angle_x, angle_y, angle_z)
    vertices = vertices @ R.T

    return vertices, faces

def subdivide(vertices, faces, n):
    """Subdivide each triangle face into smaller triangles."""
    for _ in range(n):
        new_faces = []
        midpoints = {}
        for tri in faces:
            v1, v2, v3 = tri
            for edge in [(v1, v2), (v2, v3), (v3, v1)]:
                if edge not in midpoints:
                    midpoint = (vertices[edge[0]] + vertices[edge[1]]) / 2
                    midpoint = midpoint / np.linalg.norm(midpoint)
                    midpoints[edge] = len(vertices)
                    midpoints[edge[::-1]] = midpoints[edge]
                    vertices = np.vstack([vertices, midpoint])
            m1, m2, m3 = midpoints[(v1, v2)], midpoints[(v2, v3)], midpoints[(v3, v1)]
            new_faces.extend([[v1, m1, m3], [v2, m2, m1], [v3, m3, m2], [m1, m2, m3]])
        faces = np.array(new_faces)
    return vertices, faces

def create_icosphere(subdivisions, angle_x=0, angle_y=0, angle_z=0):
    """Create an icosphere by subdividing an icosahedron and rotating it."""
    vertices, faces = create_icosahedron(angle_x, angle_y, angle_z)
    vertices, faces = subdivide(vertices, faces, subdivisions)
    return vertices, faces

def icosphere_to_graph(vertices, faces):
    """Convert the icosphere to a NetworkX graph."""
    G = nx.Graph()
    for i, vertex in enumerate(vertices):
        G.add_node(i, pos=vertex)
    for face in faces:
        for i in range(3):
            G.add_edge(face[i], face[(i + 1) % 3])
    return G

def plot_graph_3d(G, vertices, faces, node_colors=None, edge_colors=None, alpha=0.0):
    """Plot the NetworkX graph in 3D using Matplotlib."""
    pos = nx.get_node_attributes(G, 'pos')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    # Normalize edge colors to [0, 1] for colormap
    norm = plt.Normalize(-1, 1)
    cmap = cm.get_cmap('seismic')

    # Plot the surface mesh
    poly3d = [[vertices[face] for face in faces]]
    ax.add_collection3d(Poly3DCollection(vertices[faces], facecolors='lightgrey', linewidths=0.2, edgecolors='black', alpha=alpha))

    # Plot edges with colors based on edge_colors
    for idx, edge in reversed(list(enumerate(G.edges()))):
        x = np.array([pos[edge[0]][0], pos[edge[1]][0]])
        y = np.array([pos[edge[0]][1], pos[edge[1]][1]])
        z = np.array([pos[edge[0]][2], pos[edge[1]][2]])
        if edge_colors is not None:
            ax.plot(x, y, z, color=cmap(norm(edge_colors[idx])), linewidth=2)
        else:
            ax.plot(x, y, z, color='lightgrey', linewidth=2)

    # Plot nodes
    xs = np.array([pos[node][0] for node in G.nodes()])
    ys = np.array([pos[node][1] for node in G.nodes()])
    zs = np.array([pos[node][2] for node in G.nodes()])
    if node_colors is not None:
        ax.scatter(xs, ys, zs, c=node_colors, cmap=cmap, vmin=-1, vmax=1,s=20)
    else:
        ax.scatter(xs, ys, zs, color='grey', s=10)
    
    ### Add colorbar
    # mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    # mappable.set_array(edge_colors)
    # plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    ### Remove the axes
    ax.set_axis_off()

    plt.show()