import torch
import numpy as np

def depth_to_normal_torch(depth: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of depth maps to normal maps.
    
    Args:
        depth (torch.Tensor): Depth maps of shape [B, 1, H, W]
    
    Returns:
        torch.Tensor: Normal maps of shape [B, 3, H, W], scaled to [0, 255] as uint8.
    """
    B, _, H, W = depth.shape
    depth = depth.float()

    # Sobel kernels for dx and dy
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3) / 8.0

    # Compute gradients
    dx = F.conv2d(depth, sobel_x, padding=1)
    dy = F.conv2d(depth, sobel_y, padding=1)

    # Stack to get normal vectors
    dz = torch.ones_like(depth)
    normal = torch.cat([-dx, -dy, dz], dim=1)

    # Normalize the vectors
    norm = torch.linalg.norm(normal, dim=1, keepdim=True)
    normal = normal / (norm + 1e-10)

    # Scale from [-1, 1] to [0, 1]
    normal = (normal * 0.5) + 0.5

    # Convert to uint8
    normal = (normal * 255).clamp(0, 255).to(torch.uint8)

    return normal

def depth_to_normal_numpy(depth):
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
