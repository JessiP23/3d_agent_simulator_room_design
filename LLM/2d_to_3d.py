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

    return depth_map

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

    # Create a Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Simplify mesh
    simplified_mesh = mesh.simplify_quadric_decimation(face_count=5000)

    # Texture coordinates (UV mapping)
    uv = np.stack([x.flatten() / width, y.flatten() / height], axis=1)

    return simplified_mesh.vertices, simplified_mesh.faces, uv, img_array

def visualize_3d_room(vertices, faces, uv, texture):
    fig = go.Figure()

    # Create the mesh
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=texture.reshape(-1, 3) / 255.0,
        intensity=vertices[:, 2],
        colorscale='Viridis'
    )

    fig.add_trace(mesh)

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.show()


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

    plt.imshow(image)
    plt.axis('off')
    plt.title("Generated Room Image")
    plt.show()
    
    # Estimate depth
    depth_map = estimate_depth(image)
    
    # Create 3D mesh with texture
    vertices, faces, uv, texture = create_3d_mesh(image, depth_map)

    # Visualize result interactively using Plotly with texture
    visualize_3d_room(vertices, faces, uv, texture)
    
    return vertices, faces, uv, texture
    
# Define prompt for room generation
prompt = "a modern living room with a large window, a comfortable sofa, and a coffee table"
vertices, faces, uv, colors = process_room_to_3d(prompt)