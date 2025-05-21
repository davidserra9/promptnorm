import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from glob import glob
import os

def depth_to_normal(depth):
    # Ensure depth is float32 for precision
    depth = depth.astype(np.float32)
    
    # Compute gradients using NumPy's gradient function
    dy, dx = np.gradient(depth)
    
    # Stack the gradients with ones to create normal vectors
    normal = np.dstack((-dx, -dy, np.ones_like(depth)))
    
    # Normalize the vectors
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / (norm + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Scale and offset to [0, 1] range
    normal = normal * 0.5 + 0.5
    
    # Scale to [0, 255] range and convert to uint8
    return (normal * 255).astype(np.uint8)

DEPTH_PATH = "path/to/your/depth/images"  # Replace with your depth images path
for i in tqdm(glob(os.path.join(DEPTH_PATH, "*"))):
    depth_map = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    normal_map = depth_to_normal(depth_map)
    out_path = os.path.join(DEPTH_PATH.replace("/depth", "/normal"), i.split("/")[-1].replace("_depth.png", "_normal.png"))
    cv2.imwrite(out_path, normal_map)