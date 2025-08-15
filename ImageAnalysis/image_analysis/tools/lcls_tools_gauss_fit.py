from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from pathlib import Path

if TYPE_CHECKING:
    from numpy.typing import NDArray


import numpy as np
from lcls_tools.common.image.fit import ImageProjectionFit

def gauss_fit(img: np.ndarray) -> dict[str, float]:

    result = ImageProjectionFit().fit_image(img)

    # Add prefixes
    prefix_x = f'gaussian_fit_x_'
    prefix_y = f'gaussian_fit_y_'

    prefixed_dict_x = {f'{prefix_x}{k}': v for k, v in result.x_projection_fit_parameters.items()}
    prefixed_dict_y = {f'{prefix_y}{k}': v for k, v in result.y_projection_fit_parameters.items()}

    # Combine into one dictionary
    combined_dict = {**prefixed_dict_x, **prefixed_dict_y}

    return combined_dict
