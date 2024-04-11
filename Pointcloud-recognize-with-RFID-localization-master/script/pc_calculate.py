import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, AffinityPropagation
import pandas as pd

# Load point cloud
# pcd = o3d.io.read_point_cloud("/home/adan/pcd/house.pcd")
pcd = o3d.io.read_point_cloud("/home/haoran/pcd/house.pcd")

origin_pcd = np.asarray(pcd.points)

# Downsample point cloud
pcd_down = pcd.voxel_down_sample(voxel_size=0.1)

# Filter point cloud
pcd_down_filtered, ln = pcd_down.remove_statistical_outlier(nb_neighbors=60, std_ratio=2)

points = np.asarray(pcd_down_filtered.points)

# delete points with z < -0.6 and z > 1.3
points = points[points[:,2] > 0.5]
points = points[points[:,2] < 1.3]

# numpy convert to open3d
pcd_processed = o3d.geometry.PointCloud()
pcd_processed.points = o3d.utility.Vector3dVector(points)

# Filter point cloud
pcd_processed, ln = pcd_processed.remove_statistical_outlier(nb_neighbors=600, std_ratio=1)

# Show point cloud
# o3d.visualization.draw_geometries([pcd_processed])

# open3d convert to numpy
points = np.asarray(pcd_processed.points)

x_max = np.max(points[:,0])
x_min = np.min(points[:,0])
y_max = np.max(points[:,1])
y_min = np.min(points[:,1])

points = points[points[:,0] < (x_max - 1)]
points = points[points[:,0] > (x_min + 1)]

points = points[points[:,1] < (y_max - 1)]
points = points[points[:,1] > (y_min + 1)]


# PCA to 2D
pca_first = PCA(n_components=2)
pca_first.fit(points)
points = pca_first.transform(points)
trans_3d = pca_first.components_
# DBSCAN
db = cluster.DBSCAN(eps=0.5, min_samples=100).fit(points)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)
print(labels)

# Get different clusters
clusters = []
for i in range(n_clusters_):
    # No noise
    if i != -1:
        cluster = points[labels == i]
        clusters.append(cluster)

pca_2d = []
boundary = []
origin_boundary = []
# cluster PCA
for cluster in clusters:
    cluster_copy = cluster.copy()
    pca = PCA(n_components=2)
    pca.fit(cluster_copy)
    cluster_copy = pca.transform(cluster_copy)

    x_delta = np.max(cluster_copy[:,0]) - np.min(cluster_copy[:,0])
    y_delta = np.max(cluster_copy[:,1]) - np.min(cluster_copy[:,1])
    area_box = x_delta * y_delta

    x_max = np.max(cluster_copy[:,0])
    x_min = np.min(cluster_copy[:,0])
    y_max = np.max(cluster_copy[:,1])
    y_min = np.min(cluster_copy[:,1])
    # Grid
    grid_size = 0.1
    x_num = int((x_max - x_min) / grid_size) + 1
    y_num = int((y_max - y_min) / grid_size) + 1
    grid = np.zeros((x_num, y_num))
    area_grid = 0
    for point in cluster_copy:
        x_index = int((point[0] - x_min) / grid_size)
        y_index = int((point[1] - y_min) / grid_size)
        if grid[x_index][y_index] == 0:
            grid[x_index][y_index] = 1
            area_grid += grid_size * grid_size
    
    ratio = area_grid / area_box
    # plt.scatter(cluster[:,0], cluster[:,1])
    # plt.show()
    if ratio > 0.8:
        pca_2d.append(pca)
        pointarr = []
        oripointarr = []
        # 将四个点转换到原始坐标系
        cluster_xmean = np.mean(cluster[:,0])
        cluster_ymean = np.mean(cluster[:,1])
        point1 = np.array([x_max, y_max])
        point2 = np.array([x_max, y_min])
        point3 = np.array([x_min, y_max])
        point4 = np.array([x_min, y_min])

        oripointarr.append(point1)
        oripointarr.append(point2)
        oripointarr.append(point3)
        oripointarr.append(point4)

        origin_boundary.append(oripointarr)

        # 获取PCA的转换矩阵
        pca_matrix = pca.components_

        point1n = pca.inverse_transform(point1)
        point2n = pca.inverse_transform(point2)
        point3n = pca.inverse_transform(point3)
        point4n = pca.inverse_transform(point4)

        pointarr.append(point1n)
        pointarr.append(point2n)
        pointarr.append(point3n)
        pointarr.append(point4n)

        boundary.append(pointarr)

        # origin_cluster = pca.inverse_transform(cluster_copy)
        plt.plot([point1n[0], point2n[0]], [point1n[1], point2n[1]], color='r')
        plt.plot([point1n[0], point3n[0]], [point1n[1], point3n[1]], color='r')
        plt.plot([point4n[0], point2n[0]], [point4n[1], point2n[1]], color='r')
        plt.plot([point4n[0], point3n[0]], [point4n[1], point3n[1]], color='r')
        
# plt.scatter(points[:,0], points[:,1], c=labels)
# plt.show()

boundary_3D = []

for i in range(len(boundary)):
    point1 = pca_first.inverse_transform(boundary[i][0])
    point2 = pca_first.inverse_transform(boundary[i][1])
    point3 = pca_first.inverse_transform(boundary[i][2])
    point4 = pca_first.inverse_transform(boundary[i][3])
    pointarr = []
    pointarr.append(point1)
    pointarr.append(point2)
    pointarr.append(point3)
    pointarr.append(point4)
    boundary_3D.append(pointarr)

# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name='绘制多边形')
# 如果三维电云在四个点围成的区域内，则保留成一类
shelf_cluster = []
for i in range(len(boundary_3D)):
    point1 = boundary_3D[i][0]
    point2 = boundary_3D[i][1]
    point3 = boundary_3D[i][2]
    point4 = boundary_3D[i][3]
    for point in pcd_processed.points:
        shelf_points = []
        # 判断点是否在四个点围成的区域内
    x_mean = (point1[0] + point2[0] + point3[0] + point4[0]) / 4
    y_mean = (point1[1] + point2[1] + point3[1] + point4[1]) / 4
    for point in origin_pcd:
        if (point[0]-x_mean)**2 + (point[1]-y_mean)**2 < 0.04:
            shelf_points.append(point)
    shelf_points = np.array(shelf_points)
    z_max = np.max(shelf_points[:,2])
    z_min = np.min(shelf_points[:,2]) + 0.2
    corner1 = np.array([point1[0], point1[1], z_max])
    corner2 = np.array([point2[0], point2[1], z_max])
    corner3 = np.array([point3[0], point3[1], z_max])
    corner4 = np.array([point4[0], point4[1], z_max])
    corner5 = np.array([point1[0], point1[1], z_min])
    corner6 = np.array([point2[0], point2[1], z_min])
    corner7 = np.array([point3[0], point3[1], z_min])
    corner8 = np.array([point4[0], point4[1], z_min])
    polygon_points = np.array([corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8])
    lines = [[0, 1], [0, 2], [3, 1], [3, 2], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [4, 6], [7, 5], [7, 6]]
    color = [[1, 0, 0] for i in range(len(lines))] 
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    points_pcd.paint_uniform_color([0, 0.3, 0]) #点云颜色
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(color)
#     vis.add_geometry(lines_pcd)
#     vis.add_geometry(points_pcd)

# vis.add_geometry(pcd_down)
# vis.run()
# vis.destroy_window()
        
data = pd.read_excel('/home/haoran/GitHub/Pointcloud-recognize-with-RFID-localization/data/real_position.xlsx')
x_data = data['Y']
y_data = data['X']
z_data = data['Z']

position = []
for i in range(len(x_data)):
    position.append([x_data[i], y_data[i], z_data[i]])

new_pos = pca_first.transform(position)

# print(new_pos)

cluster_index = []
for i in range(len(pca_2d)):
    pos_index = []
    cluster_pos = pca_2d[i].transform(new_pos)
    #print(cluster_pos)
    x_max = origin_boundary[i][0][0]
    x_min = origin_boundary[i][3][0]
    y_max = origin_boundary[i][0][1]
    y_min = origin_boundary[i][3][1]
    # print("-----------")
    for j in range(len(cluster_pos)):
        if cluster_pos[j][0] <= x_max and cluster_pos[j][0] >= x_min and cluster_pos[j][1] <= y_max and cluster_pos[j][1] >= y_min:
            pos_index.append(j)
    cluster_index.append(pos_index)
print(cluster_index)

rfid_tag = []
for index in cluster_index:
    tag_cluster = []
    for i in index:
        tag_cluster.append(position[i])
    rfid_tag.append(tag_cluster)

# Use AffinityPropagation to cluster the points
for tag in rfid_tag:
    tag = np.array(tag)
    z_pos = tag[:, 2]
    z_pos = np.array(z_pos).reshape(1, -1)
    z_pos = z_pos.T

    isAgin = True
    pref_index = 100
    aff_centers = []
    aff_labels = []
    while(isAgin):
        model = AffinityPropagation(damping=0.8, max_iter=200, convergence_iter=15, copy=True, preference=-2, affinity='euclidean', verbose=False)
        model.fit(z_pos)
        z_centers = model.cluster_centers_
        z_labels = model.labels_
        z_centers = np.sort(z_centers, axis=0)
        if len(z_centers) == 1:
            isAgin = False
            aff_centers.append(z_centers)
            aff_labels.append(z_labels)
        else:
            for i in range(len(z_centers)-1):
                if z_centers[i+1] - z_centers[i] < 0.25:
                    isAgin = True
                    pref_index -= 0.1
                    break
                else:
                    isAgin = False
                    aff_centers.append(z_centers)
                    aff_labels.append(z_labels)

    print(aff_centers)
    print(aff_labels)
    print("------------")


