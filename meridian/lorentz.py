"""
All formulation are taken from "Hyperbolic Image-Text Representation (https://arxiv.org/abs/2304.09172)".

Implementation of common operations in the Lorentz model of hyperbolic geometry.

Only the space coordinates are stored explicitly. The time coordinate
is reconstructed using the hyperboloid constraint

x_time = sqrt(1/curv + ||x_space||^2).
"""

import math
import torch
from torch import Tensor

def lorentz_inner_product(x: Tensor, y: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """
    Args:
        x: Tensor of shape (B1, D)
        y: Tensor of shape (B2, D)
        curv: Positive scalar representing the negative curvature of the Lorentzian space

    Returns:
        Tensor of shape (B1, B2) representing the Lorentzian inner product between x and y
    """

    # Calculate the time component of the Lorentzian coordinates
    x_time = torch.sqrt((1.0/curv) + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt((1.0/curv) + torch.sum(y**2, dim=-1, keepdim=True))

    # Calculate the Lorentzian inner product
    _lorentz_product = x @ y.T - x_time @ y_time.T
    return _lorentz_product

def lorentz_distance(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8)-> Tensor:
    """
    Args:
        x: Tensor of shape (B1, D)
        y: Tensor of shape (B2, D)
        curv: Positive scalar representing the negative curvature of the Lorentzian space
        eps: Small constant to prevent numerical instability

    Returns:
        Tensor of shape (B1, B2) representing the Lorentzian distance between x and y
    """
    
    curv_lorentz_product = -curv * lorentz_inner_product(x, y, curv)
    # Ensure numerical stability: clamp to valid range for acosh (min=1.0)
    # But also set a max to prevent log(inf)
    clamped_product = torch.clamp(curv_lorentz_product, min=1.0 + eps)
    _lorentz_distance = torch.acosh(clamped_product)
    return _lorentz_distance / curv**0.5

def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map Points from Tangent space to vertex of the hyperboloid, on the hyperboloid.
    Args:
        x: Tensor of shape (B, D) giving the batch of Euclidean vectors to project onto the hyperboloid
        curv: Positive scalar representing the negative curvature of the Lorentzian space
        eps: Small constant to prevent numerical instability

    Returns:
        Tensor of shape (B, D) representing the points on the hyperboloid corresponding to the input Euclidean vectors
    """    
    rootc_xnorm = (curv**0.5) * torch.norm(x , dim=-1, keepdim=True)
    # Clamp sinh_input more conservatively to avoid overflow
    sinh_input = torch.clamp(rootc_xnorm, min=eps, max=math.asinh(2**15))
    _sinh_output = (torch.sinh(sinh_input) * x) / torch.clamp(rootc_xnorm, min=eps)
    return _sinh_output

def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Inverse of ``exp_map0``. Maps points from the hyperboloid back to the
    tangent space at the hyperboloid origin using the logarithmic map.

    This implementation assumes the base point is the hyperboloid origin

        [1/sqrt(curv), 0, ..., 0]

    (or equivalently [1/sqrt(curv), 0, ..., 0] depending on coordinate
    ordering). For this special case, the projection onto the tangent space
    at the origin reduces to the spatial coordinates of the point. Since
    this implementation stores only the spatial coordinates, the projection
    term does not need to be computed explicitly.

    Args:
        x: Tensor of shape (B, D) containing the spatial components of
            points on the hyperboloid.
        curv: Positive scalar representing the magnitude of the negative
            curvature of the hyperbolic space.
        eps: Small constant used for numerical stability.

    Returns:
        Tensor of shape (B, D) containing the corresponding vectors in the
        tangent space at the hyperboloid origin.
    """    
    # Calculate distance of vectors to the hyperboloid vertex.
    rootc_xtime = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance = torch.acosh(torch.clamp(rootc_xtime, min=1 + eps))

    rootc_xnorm = (curv**0.5) * torch.norm(x , dim=-1, keepdim=True)
    _log_output = _distance * x / torch.clamp(rootc_xnorm, min=eps)
    return _log_output

def half_aperture_angle(x: Tensor, curv: float | Tensor = 1.0, k: float = 0.1, eps: float = 1e-8) -> Tensor:
    """
    Calculate the half-aperture angle of the tangent cone at a point on the hyperboloid.
    Args:
        x: Tensor of shape (B, D) containing the spatial components of points on the hyperboloid.
        curv: Positive scalar representing the magnitude of the negative curvature of the hyperbolic space.
        k: Scaler used for setting the boundry condition near the origin of the hyperboloid.
        eps: Small constant used for numerical stability.

    Returns:
        Tensor of shape (B,) containing the half-aperture angle of the tangent cone at each point on the hyperboloid.
        Values are in the range of (0, pi/2)
    """

    asin_input = torch.clamp((2* k ) / ((curv**0.5) * torch.norm(x , dim=-1)), min = -1 + eps, max = 1 - eps)
    return torch.asin(asin_input)

def exterior_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Calculate the exterior angle between two points on the hyperboloid.
    Args:
        x: Tensor of shape (B, D) containing the spatial components of points on the hyperboloid.
        y: Tensor of shape (B, D) containing the spatial components of points on the hyperboloid.
        curv: Positive scalar representing the magnitude of the negative curvature of the hyperbolic space.
        eps: Small constant used for numerical stability.

    Returns:
        Tensor of shape (B,) containing the exterior angle between each pair of points on the hyperboloid.
        Values are in the range of (0, pi) 
    """

    x_time = torch.sqrt((1.0/curv) + torch.sum(x**2, dim = -1))
    y_time = torch.sqrt((1.0/curv) + torch.sum(y**2, dim = -1))

    curv_lorentz_product = curv * (torch.sum(x * y, dim = -1) - x_time * y_time)
    acos_input_num = y_time + x_time * curv_lorentz_product
    acos_input_den = (
        torch.sqrt(torch.clamp(curv_lorentz_product**2 - 1, min = eps))
        * torch.norm(x + 1e-15, dim = -1) + eps
    )
    return torch.acos(torch.clamp(acos_input_num / acos_input_den, min = -1 + eps, max = 1 - eps))
 
def spatial_norm(x: Tensor) -> Tensor:
    """
    Euclidean norm of the stored spatial coordinates of a hyperboloid point.

    Since d(origin, x) = arcosh(sqrt(1 + c‖x_space‖²)) / sqrt(c) is strictly
    increasing in ‖x_space‖, this norm preserves the distance ranking from the
    hyperbolic origin exactly — parents (general) have small norm, children
    (specific) have large norm.

    NOTE: x must be in the codebase convention (spatial coords only, D-dim),
    NOT a full (D+1)-dim Lorentz vector.

    Args:
        x: Tensor of shape (..., D)
    Returns:
        Tensor of shape (...,)
    """
    return torch.norm(x + 1e-15, dim=-1)