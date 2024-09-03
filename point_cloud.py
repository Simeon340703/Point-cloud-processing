import open3d as o3d
import numpy as np
import open3d.core as o3c
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#1. Create an empty point cloud
pcd = o3d.t.geometry.PointCloud()
print(pcd)

#2. Creating a Point Cloud from Arrays


# Create a point cloud from a numpy array
pcd = o3d.t.geometry.PointCloud(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))
print(pcd)

# Create a point cloud from a tensor
pcd = o3d.t.geometry.PointCloud(o3c.Tensor([[0, 0, 0], [1, 1, 1]], o3c.float32))
print(pcd)

#3. Visualize point cloud
# Load a PLY point cloud and visualize it
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.t.io.read_point_cloud(ply_point_cloud.path)
o3d.visualization.draw_geometries([pcd.to_legacy()],
                                  window_name="Visualization",
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

#4. Downsampling
#Voxel downsampling
# Downsample the point cloud with a voxel size of 0.03
downpcd = pcd.voxel_down_sample(voxel_size=0.03)
o3d.visualization.draw_geometries([downpcd.to_legacy()],
                                  window_name="Voxel downsampling",
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
#Farthest point downsampling
# Downsample the point cloud by selecting 5000 farthest points
downpcd_farthest = pcd.farthest_point_down_sample(5000)
o3d.visualization.draw_geometries([downpcd_farthest.to_legacy()],
                                  window_name="Farthest point downsampling",
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

#5. Normal estimation
# Estimate normals
downpcd.estimate_normals(max_nn=30, radius=0.1)
o3d.visualization.draw_geometries([downpcd.to_legacy()],
                                  window_name="Normals",
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)
#6. Clustering
#DBSCAN
# Apply DBSCAN clustering
labels = pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)
max_label = labels.max().item()
colors = plt.get_cmap("tab20")(labels.numpy() / (max_label if max_label > 0 else 1))
colors = o3c.Tensor(colors[:, :3], o3c.float32)
colors[labels < 0] = 0
pcd.point.colors = colors

# Visualize the clusters
o3d.visualization.draw_geometries([pcd.to_legacy()],
                                  window_name="DBSCAN visualization",
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
#K-MEANS

from sklearn.cluster import KMeans

# Extract point positions (tensor) and convert to a 2D numpy array for clustering
points = np.asarray(pcd.point.positions.cpu().numpy())

# Apply K-Means clustering
# Apply K-Means clustering with explicit n_init parameter
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(points)
labels_kmeans = kmeans.labels_

# Visualize the K-Means clustering result by assigning colors to each cluster
colors_kmeans = plt.get_cmap("tab10")(labels_kmeans / (labels_kmeans.max() if labels_kmeans.max() > 0 else 1))
pcd.point.colors = o3d.core.Tensor(colors_kmeans[:, :3], o3d.core.float32)

# Convert to legacy point cloud for visualization
pcd_legacy = pcd.to_legacy()

# Save and visualize the K-Means clustering result
o3d.io.write_point_cloud("kmeans_playground.ply", pcd_legacy)
o3d.visualization.draw_geometries([pcd_legacy], window_name="K-Means Clustering")

#7. plane segmentation
# Plane segmentation using RANSAC
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model.numpy().tolist()
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# Visualize the segmented plane
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud = inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud.to_legacy(), outlier_cloud.to_legacy()],
                                  window_name="K-MEANS Visualization",
                                  zoom=0.8,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
#8. Removing outliers
# Statistical outlier removal
# Downsample the point cloud with a voxel size of 0.03
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.03)

# Remove statistical outliers
_, ind = voxel_down_pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)

# Convert the boolean mask to indices
ind = np.asarray(ind).nonzero()[0]

# Select inliers and outliers
inlier_cloud = voxel_down_pcd.select_by_index(ind)
outlier_cloud = voxel_down_pcd.select_by_index(ind, invert=True)

# Convert to legacy point cloud for visualization
inlier_cloud_legacy = inlier_cloud.to_legacy()
outlier_cloud_legacy = outlier_cloud.to_legacy()

# Visualize the inliers and outliers
print("Showing inliers (white) and outliers (red):")
inlier_cloud_legacy.paint_uniform_color([1, 1, 1])
outlier_cloud_legacy.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([inlier_cloud_legacy, outlier_cloud_legacy],
                                  window_name="Statistical outlier removal",
                                  zoom=0.8,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
#Radius Outlier Removal
# Remove statistical outliers
_, ind = voxel_down_pcd.remove_radius_outliers(nb_points=16,
                                                search_radius=0.05)

# Convert the boolean mask to indices
ind = np.asarray(ind).nonzero()[0]

# Select inliers and outliers
inlier_cloud = voxel_down_pcd.select_by_index(ind)
outlier_cloud = voxel_down_pcd.select_by_index(ind, invert=True)

# Convert to legacy point cloud for visualization
inlier_cloud_legacy = inlier_cloud.to_legacy()
outlier_cloud_legacy = outlier_cloud.to_legacy()

# Visualize the inliers and outliers
print("Showing inliers (white) and outliers (red):")
inlier_cloud_legacy.paint_uniform_color([1, 1, 1])
outlier_cloud_legacy.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([inlier_cloud_legacy, outlier_cloud_legacy],
                                  window_name="Radius outlier removal",
                                  zoom=0.8,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])