from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any, Union, Sequence
from pathlib import Path
if TYPE_CHECKING:
    from numpy.typing import NDArray

import numpy as np
from scipy.optimize import curve_fit

from warnings import warn

from ..types import Array2D, QuantityArray2D
from ..base import ImageAnalyzer
from ..utils import ROI, read_imaq_image, NotAPath

from phasicsdensity.phasics_density_analysis import PhasicsImageAnalyzer, ureg

class U_PhasicsFileCopyImageAnalyzer(ImageAnalyzer):

    def __init__(self, 
                 roi: ROI = ROI(top=None, bottom=317, left=118, right=1600), 
                 medium: str = 'plasma',
                 background_path: Path = NotAPath(),
                 on_no_background: str = 'warn',
                ):
        """
        Parameters
        ----------
        roi : ROI
            Region of interest, as top, bottom (where top < bottom), left, right.
        medium : str
            One of 'plasma', 'gas/He', 'gas/N', for calculating density from Abel-
            inverted wavefront.
        background_path : Path
            A file or folder containing interferograms to use as background.
        on_no_background : str
            What to do if no background is set explicitly and no background path is
            given. 
                'raise': raise ValueError
                'warn': return wavefront with no background subtraction and issue warning
                'ignore': return wavefrtont with no background subtraction and don't
                          issue warning.

        """

        self.roi = roi
        self.medium = medium
        self.background: Optional[QuantityArray2D] = None
        self.on_no_background: str = on_no_background

        # for loading backgrounds on the fly. 
        self.background_path: Path = background_path
        self.background_cache: dict[tuple[Path, ROI], Array2D] = {}

    def calculate_wavefront(self, image: Array2D):
        """ Convenience function to invoke PhasicsImageAnalyzer.
        
        For example to generate backgrounds.
        
        Expects uncropped image, i.e. applies same cropbox as analyze_image() does.
        
        Parameters
        ----------
        image : Array2D
            A U_PhasicsFileCopy image

        Returns
        -------
        wavefront : Array2D with units [length] (optical path difference)

        """
        pia = PhasicsImageAnalyzer(reconstruction_method='velghe')
        return pia.calculate_wavefront(self.roi.crop(image))

    def set_background(self, wavefront: QuantityArray2D):
        """ Set background wavefront, as produced by 
            UPhasicsFileCopyImageAnalyzer.calculate_wavefront()
        """
        self.background = wavefront

    def calculate_background_from_path(self):
        def _calculate_background_from_filepath(filepath: Path) -> QuantityArray2D:
            return self.calculate_wavefront(self.roi.crop(read_imaq_image(filepath)))

        if self.background_path.is_file():
            return _calculate_background_from_filepath(self.background_path)
        
        elif self.background_path.is_dir():
            backgrounds = [_calculate_background_from_filepath(filepath)
                           for filepath in self.background_path.iterdir()
                           if filepath.suffix.lower() in ['.png', '.tif']
                          ]
            return sum(backgrounds) / len(backgrounds)

        else: 
            raise FileNotFoundError(f"Background path not found: {self.background_path}")

    def _get_background_wavefront(self):
        # get background
        # first check if it's explicitly set
        if self.background is not None:
            return self.background
        
        # next check for background_path
        elif self.background_path is not None:
            if (self.background_path, self.roi) not in self.background_cache:
                self.background_cache[(self.background_path, self.roi)] = self.calculate_background_from_path()
            return self.background_cache[(self.background_path, self.roi)]

        else:
            if self.on_no_background == 'ignore':
                pass

            elif self.on_no_background == 'raise':
                raise ValueError("No background wavefront. Use set_background(wavefront), where wavefront is "
                                 "obtained with calculate_wavefront(phasics_image)"
                                )
            elif self.on_no_background == 'warn':
                warn("No background wavefront. Returning wavefront with no background subtraction.")

            else:
                raise ValueError(f"Unknown value for on_no_background: {self.on_no_background}. Should be one of 'raise', 'warn', or 'ignore'")

    def analyze_image(self, image: Array2D) -> dict[str, float|NDArray]:
        """ Calculate metrics from U_PhasicsFileCopy
        """
        pia = PhasicsImageAnalyzer(reconstruction_method='velghe')
        wavefront = pia.calculate_wavefront(self.roi.crop(image))

        # subtract background
        wavefront -= self._get_background_wavefront()

        # set baseline wavefront to 0, using top and bottom rows as reference
        wavefront -= (wavefront[[0, -1], :]).mean()

        # 2d array of electron density with units [length]^-3
        # center row ((num_rows - 1) // 2) represents cylinder axis.
        density = pia.calculate_density(wavefront)
        # 1d array of electron density with units [length]^-3
        center_density_lineout = density[(density.shape[0] - 1) // 2, :]


        # ## fit density lineout

        def density_profile(x,  A1, x1, w,   A2, x2, sigma,   x3, x4):
            """ Density profile function that models the gas density lineout

            Piecewise function consisting of:
            for x < x3:         A1 / (1 + ((x - x1)/w)^2)
            for x >= x4:        A2 * exp(-(x - x2)^2/(2 * sigma^2))
            for x3 <= x < x4:   a line connecting the values of the lorentzian at x3 
                                and the gaussian at x4 
        
            """

            def lorentzian(x): 
                return A1 / (1 + np.square((x - x1) / w))
            def gaussian(x):
                return A2 * np.exp(-np.square(x - x2) / (2 * sigma**2))
                
            def midsection(x):
                return lorentzian(x3) + (gaussian(x4) - lorentzian(x3)) / (x4 - x3) * (x - x3)
                
            return (  lorentzian(x) * (x < x3) 
                    + midsection(x) * (x3 <= x) * (x < x4) 
                    + gaussian(x) * (x4 <= x)
                )

        @ureg.wraps(('cm^-3', 'px', 'px', 
                     'cm^-3', 'px', 'px',
                     'px', 'px'
                    ), 
                    ('cm^-3', None)
                   )
        def fit_density_lineout(lineout, p0=None):
            """ Fits density lineout from BELLA HTU. 
            
            Uses density_profile function, which is a Lorentzian + linear downslope + 
            Gaussian downramp. 
            
            Assumes calibration 4 * 4.64 um / px, as of 2023-01-26. This assumption is
            made in the explicit number of pixels past the peak where we look for the
            downramp.
            
            Returns
            -------
            parameters in units of the given lineout (A1, A2) and pixels. (x1, x2, x3,
            x4, w, sigma)
            
            """
            
            x = np.arange(len(lineout))    

            def estimate_parameters():
                x_max = np.argmax(lineout)
                max_density = lineout[x_max]
                # estimate lorentzian width
                for w in range(1, x_max):
                    if lineout[x_max - w] < 0.5 * max_density:
                        break
                else:
                    raise ValueError("Could not estimate lorentzian width")
                
                # set the split for the downramp as x_max + 6. 
                # NOTE this is specific to this calibration and this setup! Not 
                # generally a great way to do this.
                x_max_downramp = x_max + 6 + np.argmax(lineout[x_max + 6:])
                max_density_downramp = lineout[x_max_downramp]
                for sigma in range(1, len(lineout) - x_max_downramp):
                    if lineout[x_max_downramp + sigma] < np.exp(-0.5):
                        break
                else:
                    raise ValueError("could not estimate gaussian width")
                
                return [max_density, x_max, w,
                        max_density_downramp, x_max_downramp, sigma,
                        x_max, x_max + 6,
                    ]
                
            if p0 is None:
                p0 = estimate_parameters()

            (A1, x1, w,   A2, x2, sigma,   x3, x4), pcov = curve_fit(density_profile, x, lineout, p0) 
            
            return A1, x1, w,   A2, x2, sigma,   x3, x4

        density_lineout_fit_result = dict(zip(['A1', 'x1', 'w',  'A2', 'x2', 'sigma',  'x3', 'x4'], 
                                              fit_density_lineout(center_density_lineout)
                                         ))

        return {'wavefront_nm': wavefront.m_as('nm'), 
                'density_map_cm-3': density.m_as('cm^-3'),
                'density_lineout_cm-3': center_density_lineout.m_as('cm^-3'),
                'peak_density_cm-3': center_density_lineout.max().m_as('cm^-3'),

                'density_lineout_fit_A1_cm-3': density_lineout_fit_result['A1'].m_as('cm^-3'),
                'density_lineout_fit_x1_mm': (density_lineout_fit_result['x1'] * pia.CAMERA_RESOLUTION/ureg.px).m_as('mm'),
                'density_lineout_fit_w_mm': (density_lineout_fit_result['w'] * pia.CAMERA_RESOLUTION/ureg.px).m_as('mm'),
                'density_lineout_fit_A2_cm-3': density_lineout_fit_result['A1'].m_as('cm^-3'),
                'density_lineout_fit_x2_mm': (density_lineout_fit_result['x2'] * pia.CAMERA_RESOLUTION/ureg.px).m_as('mm'),
                'density_lineout_fit_sigma_mm': (density_lineout_fit_result['sigma'] * pia.CAMERA_RESOLUTION/ureg.px).m_as('mm'),
                'density_lineout_fit_x3_mm': (density_lineout_fit_result['x3'] * pia.CAMERA_RESOLUTION/ureg.px).m_as('mm'),
                'density_lineout_fit_x4_mm': (density_lineout_fit_result['x4'] * pia.CAMERA_RESOLUTION/ureg.px).m_as('mm'),
               }
