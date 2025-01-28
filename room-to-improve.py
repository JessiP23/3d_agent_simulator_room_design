# num of users: 5000
# num of songs: 10000
# mp3 audio: 5MB
# Total audio: 5MB * 10000 = 50GB
# Replication Factor: 3 * 50GB = 150GB
# Metadata = 10B per song = 10000 * 10B = 100KB
# 1KB per user = 5000KB = 5MB

# Frontend (APP) UI
# Load Balancer
# Web Server
# Database
# CDN






'''
MiDaS or DPT-Hybrid: Models that are better for depth estimation and 3D reconstruction
Apply smoothing or edge-araw filters to the depth map to reduce noise and improve the quality of the 3d mesh

Mesh simplification: Reduce the number of vertices and faces in the mesh to improve performance and rendering speed. pymeshlab or trimesh can help

Normal Estimation: Calculate the normals for each vertex in the mesh to improve lighting and shading effects

Texture mapping: Map the image onto the 3d mesh for a more realistic visualization instead of using vertex colors

Better lighning and shading: Use Phong shading or Physhically-based rendering (PBR) to improve the realism of the 3d scene

Interactive Visualization: Use plotly or pyvista to allow users to rotate, zoom, and pan the 3D model.

'''
