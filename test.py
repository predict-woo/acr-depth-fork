import numpy as np
import matplotlib.pyplot as plt

# Load the depth data from the .npy file
depth_data = np.load("depth_org_0_0.npy")

# Check if the depth data has an extra dimension and squeeze it
if depth_data.ndim == 3 and depth_data.shape[0] == 1:
    depth_data = np.squeeze(depth_data, axis=0)

# Mask zero values to ignore them in the plot
masked_depth_data = np.ma.masked_equal(depth_data, 0)

# Calculate the average depth ignoring zero values
average_depth = masked_depth_data.mean()
print(f"Average Depth (ignoring zeros): {average_depth}")

# Plot the depth data
plt.imshow(masked_depth_data, cmap="viridis")
plt.colorbar(label="Depth Value")
plt.title("Depth Data Visualization")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Save the figure instead of showing it
plt.savefig("depth_data_visualization.png")
plt.close()
