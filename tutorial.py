import numpy as np

# Coordinates of the points
P1 = np.array([-2.508, -2.540, 1.956])
P2 = np.array([-1.722, -2.409, 1.874])

# Vectors representing edges of the parallelepiped
u = P2 - P1
v = np.array([0, 0, 1])  # Choosing w as the unit vector along the positive z-axis

# Create the matrix M
M = np.vstack((u, v, np.array([0, 0, 1])))

# Calculate the volume using the determinant
volume = np.abs(np.linalg.det(M))

# Area calculation
area = np.linalg.norm(np.cross(u, v))

print("Volume:", volume)
print("Area:", area)
