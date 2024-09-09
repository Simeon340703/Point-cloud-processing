import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
# Load a point cloud
pcd = o3d.io.read_point_cloud("data/car.ply")
# Visualize the point cloud with normals
o3d.visualization.draw_geometries([pcd], window_name="Original point cloud")
# Save the point cloud to a new file
o3d.io.write_point_cloud("normals.ply", pcd)
# 1.  Estimate normals for each point
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Visualize the point cloud with normals
o3d.visualization.draw_geometries([pcd], point_show_normal=True, window_name="Normal Estimation")
#2. FPFH Descriptor Extraction
# Downsample the point cloud to speed up computation
downpcd = pcd.voxel_down_sample(voxel_size=0.2)

# Estimate normals for the downsampled point cloud
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Compute FPFH descriptors
fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    downpcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))

# Print FPFH descriptor for the first point
print(f"FPFH descriptor for point 0: {fpfh.data[:, 0]}")
#3. Point Cloud Registration with ICP

# Load source and target point clouds
source = o3d.io.read_point_cloud("data/car.ply")
target = o3d.io.read_point_cloud("data/000000.pcd")

# Initial transformation (create a 4x4 transformation matrix from rotation)
trans_init = np.eye(4)  # Start with an identity matrix
trans_init[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0.1, 0.1, 0.1])  # Insert the 3x3 rotation matrix

# Run ICP for fine alignment
threshold = 0.02  # Distance threshold for ICP convergence
icp_result = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

# Apply the ICP transformation to the source point cloud
source.transform(icp_result.transformation)

# Visualize the aligned point clouds
o3d.visualization.draw_geometries([source, target], window_name="ICP Registration")

#4. a) Surface Reconstruction with Poisson Algorithm
# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Perform Poisson surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Visualize the reconstructed surface
o3d.visualization.draw_geometries([mesh], window_name="Poisson Surface Reconstruction")
# b) Apply Alpha Shape reconstruction
alpha = 0.1  # Adjust the alpha parameter for a smoother or more detailed shape
alpha_shape_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

# Visualize the Alpha Shape reconstruction
o3d.visualization.draw_geometries([alpha_shape_mesh], window_name="Alpha Shape Reconstruction")
#5. PCA
# Convert point cloud to numpy array
points = np.asarray(pcd.points)

# Perform PCA to find principal components
cov_matrix = np.cov(points.T)  # Compute covariance matrix
eigvals, eigvecs = np.linalg.eig(cov_matrix)  # Perform eigen decomposition

# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

# Project points onto the first two principal components (reduce to 2D)
projected_points = points @ eigvecs[:, :2]
print("Projected Points (2D):", projected_points)
# Visualize original point cloud (3D)
o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

# Visualize the reduced 2D point cloud using Matplotlib
plt.scatter(projected_points[:, 0], projected_points[:, 1], s=1, c='blue')
plt.title("2D Projection of Point Cloud (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

#6. Multi-View Fusion with ICP (Iterative Closest Point)
# Load two point clouds from different views
pcd1 = o3d.io.read_point_cloud("data/car.ply")
pcd2 = o3d.io.read_point_cloud("data/000000.pcd")  # Assuming a different viewpoint

# Apply an initial transformation to simulate two different views
rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([0.1, 0.2, 0.3])

# Create a 4x4 transformation matrix from the 3x3 rotation matrix
trans_init = np.eye(4)  # Start with an identity matrix (4x4)
trans_init[:3, :3] = rotation_matrix  # Assign the 3x3 rotation to the top-left of the 4x4 matrix

# Apply the initial transformation to the second point cloud
pcd2.transform(trans_init)

# Perform ICP registration to align the two point clouds
threshold = 0.02  # ICP convergence threshold
icp_result = o3d.pipelines.registration.registration_icp(
    pcd2, pcd1, threshold, trans_init,  # Correctly passing the initial transformation matrix
    o3d.pipelines.registration.TransformationEstimationPointToPoint()  # Correctly passing the estimation method
)

# Apply the ICP transformation to the second point cloud
pcd2.transform(icp_result.transformation)

# Merge the two aligned point clouds
merged_pcd = pcd1 + pcd2

# Visualize the fused point cloud
o3d.visualization.draw_geometries([merged_pcd], window_name="Fused Point Cloud")
#7.  Voxel-Based Point Cloud Compression
# Apply voxel grid downsampling for compression
downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.05)

# Compare the original and downsampled point cloud sizes
print(f"Original point cloud has {len(pcd.points)} points")
print(f"Downsampled point cloud has {len(downsampled_pcd.points)} points")

# Visualize the compressed point cloud
o3d.visualization.draw_geometries([downsampled_pcd], window_name="Voxel Compressed Point Cloud")
#8. Octree Compression for Point Clouds
# Convert the point cloud to an octree structure
octree = o3d.geometry.Octree(max_depth=4)
octree.convert_from_point_cloud(pcd)

# Visualize the octree-compressed point cloud
o3d.visualization.draw_geometries([octree], window_name="Octree Compressed Point Cloud")