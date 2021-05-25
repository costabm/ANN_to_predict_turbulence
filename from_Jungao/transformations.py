import copy
import numpy as np

def normalize(v):
    # Normalizing vectors:
    if len(np.shape(v)) == 1 and np.shape(v)[0] == 3:  # if v is 3D vector.
        return v / np.linalg.norm(v)
    elif len(np.shape(v)) == 2 and np.shape(v[0])[0] == 3:  # if v is an array of 3D row-vectors.
        return np.array([v[i] / np.linalg.norm(v[i]) for i in range(np.shape(v)[0])])
    else:
        raise Exception

def truncate(v, n):
    """Truncates/pads a vector v to n decimal places without rounding"""
    # (sometimes rounding up gives nan values when calculating angles between vectors)
    trunc = []
    for f in v:
        s = '{}'.format(f)
        if 'e' in s or 'E' in s:
            trunc.append(float('{0:.{1}f}'.format(f, n)))
        else:
            i, p, d = s.partition('.')
            trunc.append(float('.'.join([i, (d+'0'*n)[:n]])))
    return np.array(trunc)

def beta_within_minus_Pi_and_Pi_func(beta_any):
    """Converts any beta in rads, to a beta inside the interval [-np.pi, np.pi], in rads"""
    beta_within_minus_Pi_and_Pi = np.arctan2(np.sin(beta_any), np.cos(beta_any))
    return beta_within_minus_Pi_and_Pi

def rotate_v1_about_v2_func(v1, v2, angle):
    """
    Olinde Rodrigues rotation formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    Returns v1_rotated, which is the vector v1 after being rotated by "angle" about vector v2.
    :param v1: vector to be rotated
    :param v2: vector or axis of rotation
    :param angle: rotation angle in radians
    :return: v1_rotated
    """
    v2_hat = v2 / np.linalg.norm(v2)  # v2 needs to be a unit vector (normalized)
    return v1 * np.cos(angle) + np.cross(v2_hat,v1) * np.sin(angle) + v2_hat * np.dot(v2_hat, v1)*(1-np.cos(angle))

def g_elem_nodes_func(g_node_coor):
    g_node_num = len(g_node_coor)
    g_nodes = np.array(list(range(g_node_num)))  # starting at 0
    elemnode1 = g_nodes[:-1]
    elemnode2 = g_nodes[:-1] + 1
    return np.column_stack((elemnode1, elemnode2))

def g_elem_L_2D_func(g_node_coor):
    # Back calculate it from g_node_coor, instead of arc_length, pontoon_s
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    g_elem_nodes = g_elem_nodes_func(g_node_coor)
    g_node_coor_z0 = copy.deepcopy(g_node_coor)
    g_node_coor_z0[:, 2] = 0  # ... now with z=0.
    g_elem_L_2D = np.array([np.linalg.norm(g_node_coor_z0[g_elem_nodes[i, 1]] - g_node_coor_z0[g_elem_nodes[i, 0]]) for i in range(g_elem_num)])
    return g_elem_L_2D

def g_elem_L_3D_func(g_node_coor):
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    g_elem_nodes = g_elem_nodes_func(g_node_coor)
    # 3D lengths (real, not projected)
    g_elem_L_3D = np.array([np.linalg.norm(g_node_coor[g_elem_nodes[i, 1]] - g_node_coor[g_elem_nodes[i, 0]]) for i in range(g_elem_num)])
    return g_elem_L_3D

def g_node_L_3D_func(g_node_coor):
    g_node_num = len(g_node_coor)
    g_nodes = np.array(list(range(g_node_num)))  # starting at 0
    g_elem_nodes = g_elem_nodes_func(g_nodes)
    g_elem_num = g_node_num - 1
    g_elem = np.array(list(range(g_elem_num)))  # Generating elements, by respective g_nodes.
    g_elem_L_3D = np.array([np.linalg.norm(g_node_coor[g_elem_nodes[i, 1]] - g_node_coor[g_elem_nodes[i, 0]]) for i in range(g_elem_num)])
    # Influence length of each node:
    g_node_L_3D = np.zeros(g_node_num)
    g_node_L_3D[0] = g_elem_L_3D[0] / 2
    g_node_L_3D[-1] = g_elem_L_3D[-1] / 2
    g_node_L_3D[1:-1] = [(g_elem_L_3D[i] + g_elem_L_3D[i+1]) / 2 for i in g_elem[:-1]]
    return g_node_L_3D

def T_xyzXYZ(x, y, z, X, Y, Z, dim='3x3'):
    """
    Generic transformation matrix, using the vectors of each axis of both reference systems as an input.
    Returns the transformation matrix T_xyzXYZ so that V_xyz = T_xyzXYZ @ V_XYZ, in formats 3x3, 6x6 or 12x12.
    V_XYZ and V_xyz are the same vector, expressed in the coordinate system XYZ and xyz, respectively.
    Note:
     A transformation matrix is the inverse (or the transpose) of a rotation matrix. To rotate a vector
     clockwise is the same as rotating the axes counter-clockwise.
    :param x: vector of the axis x, in any consistent reference frame.
    :param y: vector of the axis y, in any consistent reference frame.
    :param z: vector of the axis z, in any consistent reference frame.
    :param X: vector of the axis X, in any consistent reference frame.
    :param Y: vector of the axis Y, in any consistent reference frame.
    :param Z: vector of the axis Z, in any consistent reference frame.
    :param dim: dimension of the matrix: '3x3', '6x6', '12x12'
    :return:
    """
    cosxX = np.dot(x, X) / (np.dot( np.linalg.norm(x), np.linalg.norm(X)))
    cosxY = np.dot(x, Y) / (np.dot( np.linalg.norm(x), np.linalg.norm(Y)))
    cosxZ = np.dot(x, Z) / (np.dot( np.linalg.norm(x), np.linalg.norm(Z)))
    cosyX = np.dot(y, X) / (np.dot( np.linalg.norm(y), np.linalg.norm(X)))
    cosyY = np.dot(y, Y) / (np.dot( np.linalg.norm(y), np.linalg.norm(Y)))
    cosyZ = np.dot(y, Z) / (np.dot( np.linalg.norm(y), np.linalg.norm(Z)))
    coszX = np.dot(z, X) / (np.dot( np.linalg.norm(z), np.linalg.norm(X)))
    coszY = np.dot(z, Y) / (np.dot( np.linalg.norm(z), np.linalg.norm(Y)))
    coszZ = np.dot(z, Z) / (np.dot( np.linalg.norm(z), np.linalg.norm(Z)))

    if dim == '3x3':
        return np.array([[cosxX, cosxY, cosxZ],
                         [cosyX, cosyY, cosyZ],
                         [coszX, coszY, coszZ]])

    if dim == '6x6':
        return np.array([[cosxX, cosxY, cosxZ, 0, 0, 0],
                         [cosyX, cosyY, cosyZ, 0, 0, 0],
                         [coszX, coszY, coszZ, 0, 0, 0],
                         [0, 0, 0, cosxX, cosxY, cosxZ],
                         [0, 0, 0, cosyX, cosyY, cosyZ],
                         [0, 0, 0, coszX, coszY, coszZ]])

    if dim == '12x12':
        return np.array([[cosxX, cosxY, cosxZ, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [cosyX, cosyY, cosyZ, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [coszX, coszY, coszZ, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, cosxX, cosxY, cosxZ, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, cosyX, cosyY, cosyZ, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, coszX, coszY, coszZ, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, cosxX, cosxY, cosxZ, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, cosyX, cosyY, cosyZ, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, coszX, coszY, coszZ, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, cosxX, cosxY, cosxZ],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, cosyX, cosyY, cosyZ],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, coszX, coszY, coszZ]])

    else:
        raise ValueError("dim needs to be '3x3', '6x6' or '12x12'")

def T_LnwLs_func(theta, dim='3x3'):
    """from Local structural to Local-normal-wind coordinates, whose axes are defined as:
        x-axis: along-normal-wind (i.e. a "cos-rule-drag"), aligned with the (U+u)*cos(beta) that lies in a 2D plane normal to the bridge girder.
        y-axis: along bridge girder but respecting a M rotation in SOH report where wind is from left and leading edge goes up.
        z-axis: cross product of x and y (i.e. a "cos-rule-lift"), in the same 2D normal plane as x-axis
    :param theta: in radians
    :param dim: dimenison of matrix"""
    size = len(np.array(theta))
    vector_Ls_x = np.array([1, 0, 0])  # local structural
    vector_Ls_y = np.array([0, 1, 0])  # local structural
    vector_Ls_z = np.array([0, 0, 1])  # local structural
    vector_Lnw_x = np.einsum('i,j->ji', vector_Ls_y, np.cos(theta)) + np.einsum('i,j->ji', vector_Ls_z, np.sin(theta))  # local normal wind
    vector_Lnw_y = - vector_Ls_x  # local normal wind
    vector_Lnw_z = np.einsum('i,j->ji', vector_Ls_z, np.cos(theta)) - np.einsum('i,j->ji', vector_Ls_y, np.sin(theta))  # local normal wind
    return np.array([T_xyzXYZ(vector_Lnw_x[i], vector_Lnw_y, vector_Lnw_z[i], vector_Ls_x, vector_Ls_y, vector_Ls_z, dim=dim) for i in range(size)])

def T_LsSOH_func(dim='3x3'):
    """from the xy(z) coordinates in the fig. C.7 of SOH report, to Local structural coordinates. They are similar, but with swaped axes"""
    # Transformation matrices
    vector_Ls_x = np.array([1,0,0])  # local structural
    vector_Ls_y = np.array([0,1,0])  # local structural
    vector_Ls_z = np.array([0,0,1])  # local structural
    vector_SOH_x = vector_Ls_y  # local SOH (similar to Ls, but swaped axes)
    vector_SOH_y = vector_Ls_z  # local SOH (similar to Ls, but swaped axes)
    vector_SOH_z = vector_Ls_x  # local SOH (similar to Ls, but swaped axes)
    return T_xyzXYZ(vector_Ls_x, vector_Ls_y, vector_Ls_z, vector_SOH_x, vector_SOH_y, vector_SOH_z, dim=dim)

def T_LsGs_12b_func(g_node_coor, alpha):
    """
    (g_elem_num x 12x12) transformation matrix for the beams of the bridge girder, from global XYZ to local xyz
    Beams: Local x-axis: along bridge girder; y-axis: horizontal transversal; z-axis: vertical upwards.
    If A_local is a matrix in local coordinates then:
    T_LsGs_transpose @ A_local @ T_LsGs = A_global  (look at the "Ls" indexes canceling out, leaving "Gs")
    alpha is the vector with len(g_node_num) containing the local torsional angles (0 (or None) -> vector_y is horizontal)
    """
    g_elem_nodes = g_elem_nodes_func(g_node_coor)
    g_elem_L_3D = g_elem_L_3D_func(g_node_coor)
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    g_elem = np.array(list(range(g_elem_num)))  # Generating elements, by respective g_nodes.

    # Global axes "XYZ" and local axes "xyz"... Global:
    vector_X = np.array([1, 0, 0])
    vector_Y = np.array([0, 1, 0])
    vector_Z = np.array([0, 0, 1])
    vector_x = np.einsum('ij,i->ij', g_node_coor[g_elem_nodes[:, 1]] - g_node_coor[g_elem_nodes[:, 0]], 1 / g_elem_L_3D)
    if alpha is None:
        vector_y = normalize(-np.cross(vector_x, vector_Z))  # perpendicular to plane containing loc_x & glob_Z. Without rotating alpha
    else:
        vector_y_0 = normalize(-np.cross(vector_x, vector_Z)) # perpendicular to plane containing loc_x & glob_Z. Before rotating alpha
        alpha_elem = vec_Ls_elem_Ls_node_girder_func(alpha)
        vector_y = np.array([rotate_v1_about_v2_func(vector_y_0[i], vector_x[i], a) for i,a in enumerate(alpha_elem)])
    vector_z = normalize(np.cross(vector_x, vector_y))
    # Cosine of angle between each global axis X,Y,Z and local axis x,y,z:
    cosXx = np.einsum('i,ji->j', vector_X, vector_x)
    cosXy = np.einsum('i,ji->j', vector_X, vector_y)
    cosXz = np.einsum('i,ji->j', vector_X, vector_z)
    cosYx = np.einsum('i,ji->j', vector_Y, vector_x)
    cosYy = np.einsum('i,ji->j', vector_Y, vector_y)
    cosYz = np.einsum('i,ji->j', vector_Y, vector_z)
    cosZx = np.einsum('i,ji->j', vector_Z, vector_x)
    cosZy = np.einsum('i,ji->j', vector_Z, vector_y)
    cosZz = np.einsum('i,ji->j', vector_Z, vector_z)
    # Rotational Matrix:
    rot_mat = np.zeros((g_elem_num, 12, 12))
    for n in g_elem:
        rot_mat[n] = np.array([[cosXx[n],cosYx[n],cosZx[n],0,0,0,0,0,0,0,0,0],
                               [cosXy[n],cosYy[n],cosZy[n],0,0,0,0,0,0,0,0,0],
                               [cosXz[n],cosYz[n],cosZz[n],0,0,0,0,0,0,0,0,0],
                               [0,0,0,cosXx[n],cosYx[n],cosZx[n],0,0,0,0,0,0],
                               [0,0,0,cosXy[n],cosYy[n],cosZy[n],0,0,0,0,0,0],
                               [0,0,0,cosXz[n],cosYz[n],cosZz[n],0,0,0,0,0,0],
                               [0,0,0,0,0,0,cosXx[n],cosYx[n],cosZx[n],0,0,0],
                               [0,0,0,0,0,0,cosXy[n],cosYy[n],cosZy[n],0,0,0],
                               [0,0,0,0,0,0,cosXz[n],cosYz[n],cosZz[n],0,0,0],
                               [0,0,0,0,0,0,0,0,0,cosXx[n],cosYx[n],cosZx[n]],
                               [0,0,0,0,0,0,0,0,0,cosXy[n],cosYy[n],cosZy[n]],
                               [0,0,0,0,0,0,0,0,0,cosXz[n],cosYz[n],cosZz[n]]])
    return rot_mat

def T_LsGs_12c_func(g_node_coor, p_node_coor):
    """
    (n_columns x 12x12) transformation matrix for the columns of the bridge columns.
    Columns: Local x-axis: vertical upwards; y-axis: horizontal transversal; z-axis: along bridge girder, negative.
    If A_local is a matrix in local coordinates (with x-axis along the column height) then:
    rot_mat_transpose @ A_local @ rot_mat = A_global
    """
    g_elem_nodes = g_elem_nodes_func(g_node_coor)
    g_elem_L_3D = g_elem_L_3D_func(g_node_coor)
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    g_elem = np.array(list(range(g_elem_num)))  # Generating elements, by respective g_nodes.
    g_L_2D = g_elem_L_2D_func(g_node_coor)
    n_pontoons = len(p_node_coor)

    # Global axes "XYZ" and local axes "xyz"... Global:
    vector_X = np.array([1, 0, 0])
    vector_Y = np.array([0, 1, 0])
    vector_Z = np.array([0, 0, 1])
    node_coord_z0_j = copy.deepcopy(g_node_coor[g_elem_nodes[:, 1]])  # all g_nodes except first
    node_coord_z0_j[:, 2] = 0  # ... now with z=0.
    node_coord_z0_i = copy.deepcopy(g_node_coor[g_elem_nodes[:, 0]])  # all g_nodes except last
    node_coord_z0_i[:, 2] = 0  # ... now with z=0.
    vector_z = -np.einsum('ij,i->ij', node_coord_z0_j - node_coord_z0_i, 1 / g_L_2D)  # vector_z = (vector_j - vector_i) / ||vector_j - vector_i ||. g_L_2D is used because we want the projected length.
    vector_y = normalize(-np.cross(vector_Z, vector_z))  # right hand rule, so negative sign.
    vector_x = normalize(-np.cross(vector_z, vector_y))
    # Cosine of angle between each global axis X,Y,Z and local axis x,y,z:
    cosXx = np.einsum('i,ji->j', vector_X, vector_x)
    cosXy = np.einsum('i,ji->j', vector_X, vector_y)
    cosXz = np.einsum('i,ji->j', vector_X, vector_z)
    cosYx = np.einsum('i,ji->j', vector_Y, vector_x)
    cosYy = np.einsum('i,ji->j', vector_Y, vector_y)
    cosYz = np.einsum('i,ji->j', vector_Y, vector_z)
    cosZx = np.einsum('i,ji->j', vector_Z, vector_x)
    cosZy = np.einsum('i,ji->j', vector_Z, vector_y)
    cosZz = np.einsum('i,ji->j', vector_Z, vector_z)
    # Rotational Matrix:
    rot_mat = np.zeros((n_pontoons, 12, 12))

    for p, n in zip(range(n_pontoons), p_node_idx):
        rot_mat[p] = np.array([[cosXx[n],cosYx[n],cosZx[n],0,0,0,0,0,0,0,0,0],
                               [cosXy[n],cosYy[n],cosZy[n],0,0,0,0,0,0,0,0,0],
                               [cosXz[n],cosYz[n],cosZz[n],0,0,0,0,0,0,0,0,0],
                               [0,0,0,cosXx[n],cosYx[n],cosZx[n],0,0,0,0,0,0],
                               [0,0,0,cosXy[n],cosYy[n],cosZy[n],0,0,0,0,0,0],
                               [0,0,0,cosXz[n],cosYz[n],cosZz[n],0,0,0,0,0,0],
                               [0,0,0,0,0,0,cosXx[n],cosYx[n],cosZx[n],0,0,0],
                               [0,0,0,0,0,0,cosXy[n],cosYy[n],cosZy[n],0,0,0],
                               [0,0,0,0,0,0,cosXz[n],cosYz[n],cosZz[n],0,0,0],
                               [0,0,0,0,0,0,0,0,0,cosXx[n],cosYx[n],cosZx[n]],
                               [0,0,0,0,0,0,0,0,0,cosXy[n],cosYy[n],cosZy[n]],
                               [0,0,0,0,0,0,0,0,0,cosXz[n],cosYz[n],cosZz[n]]])
    return rot_mat

def T_LsGs_6p_func(g_node_coor, p_node_coor):
    """
    (n_pontoons x 6x6) transformation matrix for the pontoons. z-component is kept constant vertical (whereas bridge girder might not if 3D).
    Mean of the adjacent beams (except first and last one), and all z=0)
    A 90deg rotation is performed so that pontoon_vector_x, pontoon_vector_y = girder2D_v_y, -girder2D_v_x !
    The approximation made here (mean angle of adjacent beams) is the same as (cos(44)+cos(46))/2 = cos(45), with 100m
    long elements. Good enough approximation. (alternatively the angle at each node could be used)"""
    g_elem_nodes = g_elem_nodes_func(g_node_coor)
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    g_elem = np.array(list(range(g_elem_num)))  # Generating elements, by respective g_nodes.
    g_L_2D = g_elem_L_2D_func(g_node_coor)
    n_pontoons = len(p_node_coor)
    # Global axes "XYZ" and local axes "xyz"... Global:
    vector_X = np.array([1, 0, 0])
    vector_Y = np.array([0, 1, 0])
    vector_Z = np.array([0, 0, 1])
    node_coord_z0_j = copy.deepcopy(g_node_coor[g_elem_nodes[:, 1]])  # all g_nodes except first
    node_coord_z0_j[:, 2] = 0  # ... now with z=0.
    node_coord_z0_i = copy.deepcopy(g_node_coor[g_elem_nodes[:, 0]])  # all g_nodes except last
    node_coord_z0_i[:, 2] = 0  # ... now with z=0.
    vector_x = np.einsum('ij,i->ij', node_coord_z0_j - node_coord_z0_i, 1 / g_L_2D)  # vector_x = (vector_j - vector_i) / ||vector_j - vector_i ||
    vector_y = normalize(np.cross(vector_Z, vector_x))  # perpendicular to plane containing loc_x & glob_Z
    vector_z = normalize(np.cross(vector_x, vector_y))
    # 90deg rotation! So that vector_x becomes pontoon-surge dir. and vector_y becomes pontoon-sway dir.
    vector_x, vector_y = vector_y, -vector_x
    # Cosine of angle between each global axis X,Y,Z and local axis x,y,z:
    cosXx = np.einsum('i,ji->j', vector_X, vector_x)
    cosXy = np.einsum('i,ji->j', vector_X, vector_y)
    cosXz = np.einsum('i,ji->j', vector_X, vector_z)
    cosYx = np.einsum('i,ji->j', vector_Y, vector_x)
    cosYy = np.einsum('i,ji->j', vector_Y, vector_y)
    cosYz = np.einsum('i,ji->j', vector_Y, vector_z)
    cosZx = np.einsum('i,ji->j', vector_Z, vector_x)
    cosZy = np.einsum('i,ji->j', vector_Z, vector_y)
    cosZz = np.einsum('i,ji->j', vector_Z, vector_z)
    # Rotational Matrix. First, per beam element (but already rotated 90deg):
    rot_mat_elem = np.zeros((g_elem_num, 6, 6))
    for n in g_elem:
        rot_mat_elem[n] = np.array([[cosXx[n],cosYx[n],cosZx[n],0,0,0],
                                    [cosXy[n],cosYy[n],cosZy[n],0,0,0],
                                    [cosXz[n],cosYz[n],cosZz[n],0,0,0],
                                    [0,0,0,cosXx[n],cosYx[n],cosZx[n]],
                                    [0,0,0,cosXy[n],cosYy[n],cosZy[n]],
                                    [0,0,0,cosXz[n],cosYz[n],cosZz[n]]])
    # Nodal rotational Matrix:
    rot_mat = np.zeros([g_node_num, 6, 6])
    rot_mat[0] = rot_mat_elem[0]
    rot_mat[-1] = rot_mat_elem[-1]
    rot_mat[1:-1] = [(rot_mat_elem[i] + rot_mat_elem[i+1]) / 2 for i in g_elem[:-1]]
    # Final Matrix, with shape (n_pontoons,6,6) instead of (g_node_num,6,6)
    final_rot_mat = np.zeros([n_pontoons, 6, 6])
    for idx,i in zip(p_node_idx, range(n_pontoons)):
        final_rot_mat[i] = rot_mat[idx]
    return final_rot_mat

def T_LsGs_3g_func(g_node_coor, alpha):
    """
    (g_node_num x 3x3) Transformation matrix for the nodes of the bridge girder, from Global XYZ, to local xyz.
    Nodes: Mean of adjacent beams. Beams: Local x-axis: along bridge girder; y-axis: horizontal transversal (or rotated alpha);
    z-axis: vertical upwards. If A_local is a matrix in local coordinates then:
    rot_mat_transpose @ A_local @ rot_mat = A_global
    :param g_node_coor: node coordinates. shape: (n_nodes, 3)
    :param alpha: node local torsional rotation. shape: n_nodes. (optional)
    """
    g_node_num = len(g_node_coor)
    g_elem_nodes = g_elem_nodes_func(g_node_coor)
    g_elem_num = g_node_num - 1
    g_elem = np.array(list(range(g_elem_num)))  # Generating elements, by respective g_nodes.
    L_3D = np.array([np.linalg.norm(g_node_coor[g_elem_nodes[i, 1]] - g_node_coor[g_elem_nodes[i, 0]]) for i in range(g_elem_num)])

    # Global structural axes "XYZ" and local structural axes "xyz":
    vector_X = np.array([1, 0, 0])
    vector_Y = np.array([0, 1, 0])
    vector_Z = np.array([0, 0, 1])
    vector_x = np.einsum('ij,i->ij', g_node_coor[g_elem_nodes[:, 1]] - g_node_coor[g_elem_nodes[:, 0]], 1 / L_3D)
    if alpha is None:
        vector_y = normalize(-np.cross(vector_x, vector_Z))  # perpendicular to plane containing loc_x & glob_Z. Without rotating alpha
    else:
        vector_y_0 = normalize(-np.cross(vector_x, vector_Z))  # perpendicular to plane containing loc_x & glob_Z. Before rotating alpha
        alpha_elem = vec_Ls_elem_Ls_node_girder_func(alpha)
        vector_y = np.array([rotate_v1_about_v2_func(vector_y_0[i], vector_x[i], a) for i,a in enumerate(alpha_elem)])
    vector_z = normalize(np.cross(vector_x, vector_y))
    # Cosine of angle between each Global axis X,Y,Z and local axis x,y,z:
    cosXx = np.einsum('i,ji->j', vector_X, vector_x)
    cosXy = np.einsum('i,ji->j', vector_X, vector_y)
    cosXz = np.einsum('i,ji->j', vector_X, vector_z)
    cosYx = np.einsum('i,ji->j', vector_Y, vector_x)
    cosYy = np.einsum('i,ji->j', vector_Y, vector_y)
    cosYz = np.einsum('i,ji->j', vector_Y, vector_z)
    cosZx = np.einsum('i,ji->j', vector_Z, vector_x)
    cosZy = np.einsum('i,ji->j', vector_Z, vector_y)
    cosZz = np.einsum('i,ji->j', vector_Z, vector_z)
    # Rotational Matrix:
    trans_mat_elem = np.zeros((g_elem_num, 3, 3))
    for n in g_elem:
        trans_mat_elem[n] = np.array([[cosXx[n],cosYx[n],cosZx[n]],
                                      [cosXy[n],cosYy[n],cosZy[n]],
                                      [cosXz[n],cosYz[n],cosZz[n]]])
    # Nodal rotational Matrix:
    trans_mat = np.zeros([g_node_num, 3, 3])
    trans_mat[0] = trans_mat_elem[0]
    trans_mat[-1] = trans_mat_elem[-1]
    trans_mat[1:-1] = [(trans_mat_elem[i] + trans_mat_elem[i+1]) / 2 for i in g_elem[:-1]]
    return trans_mat

def T_LsGs_6g_func(g_node_coor, alpha):
    """
    (g_node_num x 6x6) Transformation matrix for the nodes of the bridge girder, from Global XYZ, to local xyz.
    Nodes: Mean of adjacent beams. Beams: Local x-axis: along bridge girder; y-axis: horizontal transversal (or rotated alpha);
    z-axis: vertical upwards. If A_local is a matrix in local coordinates then:
    rot_mat_transpose @ A_local @ rot_mat = A_global
    :param g_node_coor: node coordinates. shape: (n_nodes, 3)
    :param alpha: node local torsional rotation. shape: n_nodes. (optional)
    """
    g_node_num = len(g_node_coor)
    T_LsGs_3g = T_LsGs_3g_func(g_node_coor, alpha)
    T_LsGs_6g = np.zeros((g_node_num, 6, 6))
    T_LsGs_6g[:, :3, :3] = T_LsGs_3g
    T_LsGs_6g[:, 3:, 3:] = T_LsGs_3g
    return T_LsGs_6g

def T_LsGs_all_12b_12c_matrix_func(g_node_coor, p_node_coor, alpha):
    """
    Element transformation matrix from all elements in Global XYZ to all elements Local xyz. Shape: (n_all_elem, 12, 12)
    """
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    n_columns = len(p_node_coor)

    T_LsGs_12b = T_LsGs_12b_func(g_node_coor, alpha)
    T_LsGs_12c = T_LsGs_12c_func(g_node_coor, p_node_coor)

    T_LsGs_full_2D_elem_matrix = np.zeros((g_elem_num+n_columns, 12, 12))
    for el in range(g_elem_num):
        T_LsGs_full_2D_elem_matrix[el] = T_LsGs_12b[el]  # first nodes of the 12x12 beams
    for el, c in zip(range(g_elem_num, g_elem_num+n_columns), range(n_columns)):
        T_LsGs_full_2D_elem_matrix[el] = T_LsGs_12c[c]  # last node is the second node of the 12x12.
    return T_LsGs_full_2D_elem_matrix

def T_LsGs_all_6g_6p_matrix_func(g_node_coor, p_node_coor, alpha):
    """
    Node transformation matrix from all nodes in Global XYZ to all nodes Local xyz. Shape: (n_all_nodes, 12, 12)
    This is different from a full 2D node transformation matrix, because it only sees node by node, not off-diag interactions.
    """
    g_node_num = len(g_node_coor)
    p_node_num = len(p_node_coor)

    T_LsGs_6g = T_LsGs_6g_func(g_node_coor, alpha)
    T_LsGs_6p = T_LsGs_6p_func(g_node_coor, p_node_coor)

    T_LsGs_all_6g_6p_matrix = np.zeros((g_node_num+p_node_num, 6, 6))
    for g in range(g_node_num):
        T_LsGs_all_6g_6p_matrix[g] = T_LsGs_6g[g]  # first girder nodes of the 6x6 beams
    for p_after_g, p in zip(range(g_node_num, g_node_num+p_node_num), range(p_node_num)):
        T_LsGs_all_6g_6p_matrix[p_after_g] = T_LsGs_6p[p]
    return T_LsGs_all_6g_6p_matrix

def T_LsGs_full_2D_node_matrix_func(g_node_coor, p_node_coor, alpha):
    """
    2D Nodal transformation matrix from all nodes in Global XYZ to all nodes Local xyz. Shape: (n_all_nodes*6, n_all_nodes*6)
    """
    g_node_num = len(g_node_coor)
    p_node_num = len(p_node_coor)
    n_all_nodes = g_node_num + p_node_num

    T_LsGs_3g = T_LsGs_3g_func(g_node_coor, alpha)
    T_LsGs_6g = np.zeros((g_node_num, 6, 6))
    T_LsGs_6g[:, :3, :3] = T_LsGs_3g
    T_LsGs_6g[:, 3:, 3:] = T_LsGs_3g
    T_LsGs_6p = T_LsGs_6p_func(g_node_coor, p_node_coor)
    T_LsGs_full_2D_node_matrix = np.zeros((n_all_nodes*6, n_all_nodes*6))
    for n in range(g_node_num):
        T_LsGs_full_2D_node_matrix[n*6:n*6+6, n*6:n*6+6] = T_LsGs_6g[n]
    for p in range(p_node_num):
        T_LsGs_full_2D_node_matrix[g_node_num*6+p*6:g_node_num*6+p*6+6, g_node_num*6+p*6:g_node_num*6+p*6+6] = T_LsGs_6p[p]
    return T_LsGs_full_2D_node_matrix

def T_LrLs_func(g_node_coor):
    # Identity matrix since the Local structural and Local reference coordinate systems are assumed to be the same.
    g_node_num = len(g_node_coor)
    trans_mat = np.array([[[1,0,0],
                           [0,1,0],
                           [0,0,1]]]*g_node_num)
    return trans_mat

def T_GsGw_func(beta_0,theta_0):
    # Transformation matrix, from Gw (Global wind) to Gs (Global structural)
    return np.array([[-np.cos(theta_0) * np.sin(beta_0), -np.cos(beta_0), np.sin(theta_0) * np.sin(beta_0) ],
                     [np.cos(theta_0) * np.cos(beta_0) , -np.sin(beta_0), -np.sin(theta_0) * np.cos(beta_0)],
                     [np.sin(theta_0)                  , 0              , np.cos(theta_0)                  ]])

def T_LwGw_func():
    return np.array([[0., -1., 0.],
                     [1.,  0., 0.],
                     [0.,  0., 1.]])

def T_LrLw_func(betas, thetas, dim='3x3'):
    assert len(betas) == len(thetas)
    size = len(betas)
    T_LrLw_3 = np.array([
        [[np.cos(betas[i]), -np.cos(thetas[i]) * np.sin(betas[i]), np.sin(thetas[i]) * np.sin(betas[i])],
         [np.sin(betas[i]), np.cos(thetas[i]) * np.cos(betas[i]), -np.sin(thetas[i]) * np.cos(betas[i])],
         [0               , np.sin(thetas[i])                    , np.cos(thetas[i])                   ]]
        for i in range(size)])
    if dim == '3x3':
        return T_LrLw_3
    if dim == '6x6':
        T_LrLw_6 = np.zeros((size, 6, 6))
        T_LrLw_6[:, :3, :3] = T_LrLw_3
        T_LrLw_6[:, 3:, 3:] = T_LrLw_3
        return T_LrLw_6

def T_LwLnw_func(beta, theta, dim='6x6'):
    T_LsLnw = np.transpose(T_LnwLs_func(theta, dim=dim), axes=(0, 2, 1))
    T_LwLs = np.transpose(T_LrLw_func(beta, theta, dim=dim), axes=(0, 2, 1))
    T_LwLnw = np.einsum('nij,njk->nik', T_LwLs, T_LsLnw)
    return T_LwLnw

def T_LSOHLwbar_func(beta_bar):
    """
    The Local SOH coordinate system q*p*h*, in which the wind tunnel coefficients are obtained, is the same
    as the Lwbar system when rotated by beta back to non-skew position. So this transformation is the same
    as the transpose T_LrLwbar when the theta rotation is not performed (= 0 deg). Note that this assumes that
    the Cm axis has already been inverted from SOH's clockwise, to counter-clockwise (conforming to "q" in "qph")
    """
    trans_mat = np.array([
        [[np.cos(beta_bar[i]), -np.sin(beta_bar[i]), 0],
         [np.sin(beta_bar[i]),  np.cos(beta_bar[i]), 0],
         [0                  ,                    0, 1]]
        for i in range(len(beta_bar))])
    return trans_mat

def mat_Gs_elem_Gs_node_all_func(mat_Gs_node, g_node_coor, p_node_coor):
    """
    Not a transformation matrix.
    Converts a global nodal matrix Mat_Gs_node (e.g. displacements matrix) with shape (n_nodes_all, 6), to a global element Mat_Gs_elem (n_elem_all, 12).
    """
    # D = Displacement matrix. shape (total num nodes, 6)
    g_node_num = len(g_node_coor)
    g_elem_num = g_node_num - 1
    n_columns = len(p_node_coor)

    mat_Gs_elem = np.zeros((g_elem_num+n_columns, 12))
    for i in range(g_elem_num):  # girder nodal to girder element
        mat_Gs_elem[i, :6] = mat_Gs_node[i]
        mat_Gs_elem[i, 6:] = mat_Gs_node[i + 1]
    for c, p_idx in zip(range(g_elem_num, g_elem_num+n_columns), p_node_idx): # pontoon & girder nodal to column element
        mat_Gs_elem[c, :6] = mat_Gs_node[c + 1]
        mat_Gs_elem[c, 6:] = mat_Gs_node[p_idx]  # second node of column belongs to girder
    return mat_Gs_elem

def mat_Ls_elem_Gs_elem_all_func(mat_Gs_elem, g_node_coor, p_node_coor, alpha):
    """
    Not a transformation matrix.
    Converts e.g. a global element displacement matrix D, to the local one. Shape (total num elem, 12).
    """
    T_LsGs_all_12b_12c_matrix = T_LsGs_all_12b_12c_matrix_func(g_node_coor, p_node_coor, alpha)
    mat_Ls_elem = np.einsum('nij,nj->ni', T_LsGs_all_12b_12c_matrix, mat_Gs_elem)
    return mat_Ls_elem

def mat_Ls_node_Gs_node_all_func(mat_Gs_node, g_node_coor, p_node_coor, alpha):
    """
    Converts e.g. a global nodal displacement matrix with shape (total num nodes, 6), to a local one (same shape).
    matrix with shape (total num elem, 12).
    """
    T_LsGs_all_6g_6p_matrix = T_LsGs_all_6g_6p_matrix_func(g_node_coor, p_node_coor, alpha)
    mat_Ls_node = np.einsum('nij,nj->ni', T_LsGs_all_6g_6p_matrix, mat_Gs_node)
    return mat_Ls_node

def vec_Ls_elem_Ls_node_girder_func(vec_Ls_node):
    """
    Not a transformation matrix.
    Converts a local nodal vector vec_Ls_node (e.g. alpha vector) with shape (g_node_num), to a local element vec_Ls_elem  with shape (g_elem_num).
    """
    return np.array([(vec_Ls_node[i]+vec_Ls_node[i+1])/2 for i in range(len(vec_Ls_node)-1)])

def R_x(alpha):
    """
    Visit Wikipedia page, not Wolfram (which is wrong)
    Rotation matrix around axis x"""
    return np.array([[1, 0, 0],
                     [0, np.cos(alpha), -np.sin(alpha)],
                     [0, np.sin(alpha), np.cos(alpha)]])

def R_y(alpha):
    """Rotation matrix around axis y"""
    return np.array([[np.cos(alpha), 0, np.sin(alpha)],
                     [0, 1, 0],
                     [-np.sin(alpha), 0, np.cos(alpha)]])

def R_z(alpha):
    """Rotation matrix around axis z"""
    return np.array([[np.cos(alpha), -np.sin(alpha), 0],
                     [np.sin(alpha), np.cos(alpha), 0],
                     [0, 0, 1]])
