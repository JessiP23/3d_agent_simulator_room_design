import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation, AutoFeatureExtractor
import trimesh
# pytorch for 3D rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)

class Interior2DTo3D:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load segmentation model
        self.segmenter = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade20k"
        ).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade20k"
        )
        
        # Initialize 3D model database
        self.furniture_models = self.load_furniture_models()
        
        # Setup renderer
        self.renderer = self.setup_renderer()
        
    def load_furniture_models(self):
        # Dictionary mapping furniture types to base 3D models
        # You can expand this with more models
        models = {
            "sofa": trimesh.load("base_models/sofa.obj"),
            "chair": trimesh.load("base_models/chair.obj"),
            "table": trimesh.load("base_models/table.obj"),
            "lamp": trimesh.load("base_models/lamp.obj"),
            "cabinet": trimesh.load("base_models/cabinet.obj"),
        }
        return models
    
    def setup_renderer(self):
        cameras = PerspectiveCameras(device=self.device)
        raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=0.0,
            faces_per_pixel=1
        )
        
        # Create a renderer
        renderer = MeshRenderer(\

            # rasterizer = MeshRasterizer(
            # Example of using a custom rasterizer
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras
            )
        )
        return renderer
    
    def segment_image(self, image):
        # Prepare image for segmentation
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get segmentation
        outputs = self.segmenter(**inputs)
        segments = outputs.logits.argmax(dim=1)
        
        return segments
    
    def extract_furniture_info(self, segments, confidence_threshold=0.7):
        furniture_items = []
        
        # Process each segment
        for label in torch.unique(segments):
            mask = segments == label
            if mask.sum() > 100:  # Size threshold
                # Get bounding box
                y_indices, x_indices = torch.where(mask)
                bbox = {
                    'x1': x_indices.min().item(),
                    'y1': y_indices.min().item(),
                    'x2': x_indices.max().item(),
                    'y2': y_indices.max().item()
                }
                
                # Calculate dimensions
                width = bbox['x2'] - bbox['x1']
                height = bbox['y2'] - bbox['y1']
                
                # Estimate furniture type and properties
                furniture_type = self.classify_furniture_type(mask)
                if furniture_type:
                    furniture_items.append({
                        'type': furniture_type,
                        'bbox': bbox,
                        'dimensions': {
                            'width': width,
                            'height': height
                        },
                        'position': {
                            'x': (bbox['x1'] + bbox['x2']) / 2,
                            'y': (bbox['y1'] + bbox['y2']) / 2
                        }
                    })
        
        return furniture_items
    
    def classify_furniture_type(self, mask):
        # Use size and aspect ratio to estimate furniture type
        # This can be improved with a dedicated classifier
        y_indices, x_indices = torch.where(mask)
        height = y_indices.max() - y_indices.min()
        width = x_indices.max() - x_indices.min()
        aspect_ratio = width / height if height > 0 else 0
        
        if aspect_ratio > 2:
            return "sofa"
        elif 0.8 < aspect_ratio < 1.2:
            return "table"
        elif aspect_ratio < 0.8:
            return "cabinet"
        else:
            return "chair"
    
    def create_3d_scene(self, furniture_items):
        scene_meshes = []
        
        for item in furniture_items:
            # Get base model
            base_model = self.furniture_models[item['type']]
            
            # Scale model based on detected dimensions
            scale = np.array([
                item['dimensions']['width'] / 100,  # Normalize scale
                1.0,  # Keep height constant for now
                item['dimensions']['width'] / 100  # Assume depth = width
            ])
            
            # Create transformed mesh
            verts = torch.tensor(base_model.vertices * scale).float()
            faces = torch.tensor(base_model.faces).long()
            
            # Position the mesh
            verts = verts + torch.tensor([
                item['position']['x'],
                0,  # Place on ground
                item['position']['y']
            ])
            
            # Create texture (simple colored texture for now)
            verts_rgb = torch.ones_like(verts)[None] * torch.tensor([0.7, 0.7, 0.7])[None, None]
            textures = TexturesVertex(verts_features=verts_rgb)
            
            # Create mesh
            mesh = Meshes(
                verts=[verts],
                faces=[faces],
                textures=textures
            )
            
            scene_meshes.append(mesh)
        
        # Combine all meshes
        return self.combine_meshes(scene_meshes)
    
    def combine_meshes(self, meshes):
        if not meshes:
            return None
            
        all_verts = []
        all_faces = []
        vert_offset = 0
        
        for mesh in meshes:
            verts = mesh.verts_packed()
            faces = mesh.faces_packed()
            
            all_verts.append(verts)
            all_faces.append(faces + vert_offset)
            vert_offset += len(verts)
        
        combined_verts = torch.cat(all_verts, dim=0)
        combined_faces = torch.cat(all_faces, dim=0)
        
        return Meshes(
            verts=[combined_verts],
            faces=[combined_faces]
        )
    
    def convert_2d_to_3d(self, image_2d):
        # Main conversion pipeline
        segments = self.segment_image(image_2d)
        furniture_items = self.extract_furniture_info(segments)
        scene_3d = self.create_3d_scene(furniture_items)
        
        # Render the 3D scene
        if scene_3d is not None:
            rendered_image = self.renderer(scene_3d)
            return rendered_image, scene_3d
        else:
            return None, None
    
    def export_scene(self, scene, filename="scene.glb"):
        # Export the scene to a file
        if scene is not None:
            vertices = scene.verts_packed().cpu().numpy()
            faces = scene.faces_packed().cpu().numpy()
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.export(filename)