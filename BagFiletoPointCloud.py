import open3d as o3d
import copy
import numpy as np
import os

# Print the absolute path to verify file location
print(os.path.abspath("20241101_212304.bag"))

# Create bag reader
bag_reader = o3d.t.io.RSBagReader()

# Try to open the bag file and check if successful
if not bag_reader.open("20241101_212304.bag"):
    print("Failed to open the bag file.")
    exit()

# Get intrinsic parameters
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

# Read the first frame
try:
    im_rgbd = bag_reader.next_frame()
except RuntimeError as e:
    print(f"Error reading frame: {e}")
    bag_reader.close()
    exit()

pcds = []
voxel_size = 0.2

# Initialize an empty point cloud to store the stitched result
stitched_pcd = o3d.geometry.PointCloud()

# Loop through frames and stitch them together
i = 0

while not bag_reader.is_eof():
    try:
        if i > 100 and i < 300:
            i += 1
            print("Processing frame...")

            # Convert to Open3D geometry images
            depth = o3d.geometry.Image(np.array(im_rgbd.depth).astype('uint8'))
            color = o3d.geometry.Image(np.asarray(im_rgbd.color))

            # Create RGBD image
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth,
                depth_scale=1000.0,  # Adjust this value based on your depth unit
                depth_trunc=0.01,     # Maximum depth in meters
                convert_rgb_to_intensity=False)

            # Generate point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

            # Apply transformation to the point cloud
            pcd.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

            # If this is not the first frame, register it with the previous one using point-to-plane ICP
            if len(pcds) > 0:
                prev_pcd = pcds[-1]

                # Estimate normals for both point clouds (required for point-to-plane ICP)
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                prev_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                # Perform point-to-plane ICP using the correct API
                reg_icp = o3d.pipelines.registration.registration_icp(
                    pcd, prev_pcd, max_correspondence_distance=0.05,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())

                # Apply the transformation
                pcd.transform(reg_icp.transformation)

            # Append the current point cloud to the list
            pcds.append(pcd)

            # Downsample and add it to the global stitched point cloud
            pcd_down = pcd.uniform_down_sample(every_k_points=5)
            stitched_pcd += pcd_down

        elif i > 300:
            break
        else:
            i += 1

        # Move to the next frame
        im_rgbd = bag_reader.next_frame()

    except RuntimeError as e:
        print(f"Error processing frame: {e}")
        break

# Visualize the stitched result
o3d.visualization.draw_geometries([stitched_pcd])

# Close the bag reader
bag_reader.close()
