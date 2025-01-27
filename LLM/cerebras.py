from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
from transformers import pipeline

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

def estimate_depth(image):
    # Use DPT for depth estimation
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
    depth = depth_estimator(image)
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
    
    return vertices, faces, colors

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

def process_room_to_3d(prompt):
    # Generate room image
    image = generate_room_image(prompt)
    
    # Estimate depth
    depth_map = estimate_depth(image)
    
    # Create 3D mesh
    vertices, faces, colors = create_3d_mesh(image, depth_map)
    
    # Visualize result
    visualize_3d_room(vertices, faces, colors)
    
    return vertices, faces, colors

# Example usage
prompt = "a modern living room with a large window, a comfortable sofa, and a coffee table"
vertices, faces, colors = process_room_to_3d(prompt)