from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
from transformers import pipeline
import trimesh
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

# define generate room image by using stable diffusion pipeline
def generate_room_image(prompt):
    # utilizing stable diffusion pipeline to generate room image
    model_id = "stabilityai/stable-diffusion-2"

    
    # euler discrete scheduler
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        scheduler=scheduler, 
        torch_dtype=torch.float16
    )

    # designed to use GPU 
    pipe = pipe.to("cuda")
    return pipe(prompt).images[0]


# estimate depth for given image and convert in 3d
def estimate_depth(image):
    # Use DPT for depth estimation
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
    depth = depth_estimator(image)
    depth_map = np.array(depth["depth"])

    # Apply smoothing to the depth map
    depth_map = gaussian_filter(depth_map, sigma=1.0)

    return np.array(depth["depth"])

def create_3d_mesh(image, depth_map):
    # Convert image to numpy array
    img_array = np.array(image)
    height, width = depth_map.shape
    
    # Create mesh grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Scale and normalize coordinates
    x = (x - width/2) / width
    y = (y - height/2) / height
    z = depth_map / depth_map.max()  # Normalize depth values
    
    # Create vertices and faces
    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    faces = []
    for i in range(height-1):
        for j in range(width-1):
            v0 = i * width + j
            v1 = v0 + 1
            v2 = (i + 1) * width + j
            v3 = v2 + 1
            # Create two triangles for each quad
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    faces = np.array(faces)

    # Add color information
    colors = img_array.reshape(-1, 3) / 255.0

    # simplify mest using trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)

    # reduce faces for better performance
    simplified_mesh = mesh.simplify_quadric_decimation(face_count=5000)
    
    return simplified_mesh.vertices, simplified_mesh.faces, colors[simplified_mesh.faces].mean(axis=1)

def visualize_3d_room(vertices, faces, colors):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh with colors
    mesh = ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        shade=True
    )
    
    # Set the color of each face based on the average color of its vertices
    face_colors = np.mean(colors[faces], axis=1)
    mesh.set_facecolors(face_colors)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()


def visualize_3d_room_interactive(vertices, faces, colors):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, vertexcolor=colors, intensity=z, colorscale='Viridis')

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[mesh], layout=layout)
    fig.show()

def process_room_to_3d(prompt):
    # Generate room image
    image = generate_room_image(prompt)
    
    # Estimate depth
    depth_map = estimate_depth(image)
    
    # Create 3D mesh
    vertices, faces, colors = create_3d_mesh(image, depth_map)

    # Visualize result using matplotlib
    visualize_3d_room(vertices, faces, colors)

    # Visualize result interactively using Plotly
    visualize_3d_room_interactive(vertices, faces, colors)
    
    return vertices, faces, colors

# Define prompt for room generation
prompt = "a modern living room with a large window, a comfortable sofa, and a coffee table"
vertices, faces, colors = process_room_to_3d(prompt)