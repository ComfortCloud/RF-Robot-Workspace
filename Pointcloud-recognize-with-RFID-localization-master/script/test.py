import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load point cloud
pcd = o3d.io.read_point_cloud("/home/haoran/pcd/house.pcd")

# Downsample point cloud
pcd_down = pcd.voxel_down_sample(voxel_size=0.1)

# Filter point cloud
pcd_down_filtered, ln = pcd_down.remove_statistical_outlier(nb_neighbors=60, std_ratio=2)

# Display point cloud
# o3d.visualization.draw_geometries([pcd_down_filtered])

points = np.asarray(pcd_down_filtered.points)

# PCA
pca = PCA(n_components=3)
pca.fit(points)
points = pca.transform(points)

print(pca.explained_variance_ratio_)
print (pca.components_ )
# print(points)

# Delete points when z > 1.2 or z < -0.6
points = points[points[:,2] < 1.2]
points = points[points[:,2] > -0.55]

# numpy convert to open3d
pcd_processed = o3d.geometry.PointCloud()
pcd_processed.points = o3d.utility.Vector3dVector(points)

# Filter point cloud
pcd_processed, ln = pcd_processed.remove_statistical_outlier(nb_neighbors=200, std_ratio=1)

points = np.asarray(pcd_processed.points)

# Plot PCA
plt.scatter(points[:,0], points[:,1])
plt.show()

# Display point cloud
o3d.visualization.draw_geometries([pcd_processed])

# 计算二维点云连线的曲率，曲率小的点云为墙壁，曲率大的点云为拐角
# 将点云网格化，计算每个网格的曲率，曲率小的网格为墙壁，曲率大的网格为拐角
grid_size = 0.1
x_min = np.min(points[:,0])
x_max = np.max(points[:,0])
y_min = np.min(points[:,1])
y_max = np.max(points[:,1])
x_num = int((x_max - x_min) / grid_size)
y_num = int((y_max - y_min) / grid_size)
grid = np.zeros((x_num, y_num))

for point in points:
    x = int((point[0] - x_min) / grid_size)
    y = int((point[1] - y_min) / grid_size)
    if grid[x][y] == 0:
        grid[x][y] = 1

# 计算每个网格的曲率
curvature = np.zeros((x_num, y_num))
for i in range(1, x_num - 1):
    for j in range(1, y_num - 1):
        if grid[i][j] == 1:
            curvature[i][j] = grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1] - 4