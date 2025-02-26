from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from pathlib import Path

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..types import Array2D, QuantityArray2D

try:
    import image_analysis.third_party_sdks.wavekit_43.wavekit_py as wkpy
except Exception as e:
    print("could not import wkpy, probably running mac")
    raise

import logging
import pandas as pd

import numpy as np
from scipy.optimize import curve_fit

from warnings import warn

from ..base import ImageAnalyzer
from ..utils import ROI, read_imaq_image, NotAPath

# from image_analysis.algorithms.qwlsi import QWLSIImageAnalyzer
from .. import ureg, Q_, Quantity

class HASO_himg_has_processor(ImageAnalyzer):

    def __init__(self, 
                 roi: ROI = ROI(top=None, bottom=317, left=118, right=1600), 
                 medium: str = 'plasma',
                 background_path: Path = Path('NUL'),
                 on_no_background: str = 'warn',

                 laser_wavelength: Quantity = Q_(800, 'nanometer'),
                 camera_resolution: Quantity = Q_(7.40, 'micrometer'),
                 image_resolution: Quantity = Q_(4.74, 'micrometer'),

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

        laser_wavelength : [length] Quantity
            of imaging laser
        camera_resolution : [length] Quantity
            pixel size of camera
        image_resolution : [length] Quantity
            the real-world length represented by a pixel, which can be different from camera_resolution if there 
            are optics between object and grating/camera.

        """

        self.roi = roi
        self.medium = medium
        self.background: Optional[QuantityArray2D] = None
        self.on_no_background: str = on_no_background

        # for loading backgrounds on the fly. 
        self.background_path: Path = background_path
        self.background_cache: dict[tuple[Path, ROI], Array2D] = {}

        self.laser_wavelength = laser_wavelength
        self.image_resolution = image_resolution

        self.camera_resolution = camera_resolution

        super().__init__()

        config_file_path = 'scan_analysis/third_party_sdks/wavekit_43/WFS_HASO4_LIFT_680_8244_gain_enabled.dat'

        self.instantiate_wavekit_resources(config_file_path=config_file_path)

        self._log_info(
            f"Initialized HasoAnalysis for device '{device_name}' with scan directory '{self.scan_directory}'")

    def _log_info(self, message: str, *args, **kwargs):
        """Log an info message if logging is enabled."""
        if self.flag_logging:
            logging.info(message, *args, **kwargs)

    def _log_warning(self, message: str, *args, **kwargs):
        """Log a warning message if logging is enabled."""
        if self.flag_logging:
            logging.warning(message, *args, **kwargs)

    def instantiate_wavekit_resources(self, config_file_path: Path):
        """
        attempt to instantiate necessary wavekit resources

        Args:
            config_file_path (Path): Path to the config file.

        """

        self._log_info(f"instantiating wavekit resources: HasoEngine etc.")

        try:
            # Create the necessary Wavekit objects.
            self.hasoengine = wkpy.HasoEngine(config_file_path=config_file_path)
            self.hasoengine.set_lift_enabled(True, 800)
            self.hasoengine.set_lift_option(True, 800)

            # Set preferences with an arbitrary subpupil and denoising strength.
            start_subpupil = wkpy.uint2D(87, 64)
            denoising_strength = 0.0
            self.hasoengine.set_preferences(start_subpupil, denoising_strength, False)

            self.compute_phase_set = wkpy.ComputePhaseSet(type_phase=wkpy.E_COMPUTEPHASESET.ZONAL)
            self.compute_phase_set.set_zonal_prefs(100, 500, 1e-6)

            self.post_processor = wkpy.SlopesPostProcessor()

        except Exception as e:
            self._log_warning(
                "Not able to create necessary Wavekit objects, likely a result of Wavekit not being installed or missing/incorrect license file")
            raise

    def analyze_image(self, file_path: Path) -> dict[str, float | NDArray]:
        """
        Create phase map from a .himg or .has file.

        Parameters:
            file_path: Path to the image file.

        Returns:
            A dictionary containing results (e.g., phase map and/or related parameters).

        Raises:
            ValueError: If the file type is not supported.
        """

        ext = file_path.suffix.lower()
        if ext == ".himg":
            self.raw_slopes = self.create_slopes_object(file_path)
        elif ext == ".has":
            result = self.compute_phase_from_slopes(file_path)
        else:
            msg = f"Unsupported file extension '{ext}'. Supported file types are .himg and .has."
            logging.error(msg)
            raise ValueError(msg)

        return result


    def create_slopes_object(self, image_file_path: Path) -> HasoSlopes:
        """
        Compute and save the slopes file (.has) from the provided image file.

        Args:
            image_file_path (Path): Path to the .himg file.

        Returns:
            Path: The path to the created slopes file (.has).
        """
        self._log_info(f"Creating slopes file for image: {image_file_path}")
        image_file_str = str(image_file_path)

        try:
            # Create the necessary Wavekit objects.
            image = wkpy.Image(image_file_path=image_file_str)
        except Exception as e:
            self._log_warning(
                "Not able to create necessary Wavekit objects, likely a result of Wavekit not being installed or missing/incorrect license file")
            return None

        # Compute slopes
        learn_from_trimmer = False
        _, hasoslopes = self.hasoengine.compute_slopes(image, learn_from_trimmer)

        return hasoslopes

    def load_slopes_from_file(self, slopes_file_path: Path) -> HasoSlopes:
        base_name = slopes_file_path.stem


    def compute_phase_from_slopes(self, slopes_file_path: Path) -> pd.DataFrame:
        """
        Compute phase data from the provided slopes file (.has) and save the result as a TSV.

        Args:
            slopes_file_path (Path): Path to the slopes (.has) file.

        Returns:
            DataFrame: The computed phase data.
        """
        self._log_info(f"Computing phase data from slopes file: {slopes_file_path}")
        base_name = slopes_file_path.stem
        tsv_file_path = self.path_dict['save'] / f"{base_name}.tsv"

        hasodata = wkpy.HasoData(has_file_path=str(slopes_file_path))

        phase = wkpy.Compute.phase_zonal(self.compute_phase_set, hasodata)

        phase_values = phase.get_data()[0]
        df = pd.DataFrame(phase_values)
        df.to_csv(tsv_file_path, sep="\t", index=False, header=False)
        self._log_info(f"Phase data saved to TSV file: {tsv_file_path}")
        return df

    def post_process_slopes(self, hasoslopes_to_modify: HasoSlopes,
                            background_path: Optional[Path] = None) -> HasoSlopes:

        hasoslopes = self.reference_subtract(hasoslopes_to_modify, background_path)
        hasoslopes = self.post_processor.apply_filter(hasoslopes, True, True, True, False, False, False)

        return hasoslopes

    def reference_subtract(hasoslopes_to_modify: HasoSlopes, background_path: Optional[Path] = None) -> HasoSlopes:
        if background_path:
            bkg_data = wkpy.HasoSlopes(has_file_path=str(background_path))
            bkg_subtracted = self.post_processor.apply_substractor(hasoslopes_to_modify, bkg_data)
            return bkg_subtracted
        else:
            return hasoslopes_to_modify

    def get_phase_from_himg(self, image_file_path: Path, use_raw_slopes: Bool = True) -> pd.DataFrame:
        """
        Process a .himg file by computing the slopes file and then the phase data.

        Args:
            image_file_path (Path): Path to the .himg file.

        Returns:
            DataFrame: The computed phase data.
        """
        self._log_info(f"Starting phase analysis for image file: {image_file_path}")
        self.create_slopes_file(image_file_path)

        if use_raw_slopes:
            df = self.compute_phase_from_slopes(self.slopes_file_path_raw)
        else:
            df = self.compute_phase_from_slopes(self.slopes_file_path_postprocessed)

        self._log_info(f"Completed phase analysis for image file: {image_file_path}")
        return df

    def analyze_image(self, image: Array2D) -> dict[str, float|NDArray]:
        """ Calculate metrics from U_PhasicsFileCopy
        """
        
        self._initialize_qwlsi_image_analyzer()
        
        wavefront = self.qwlsi_image_analyzer.calculate_wavefront(self.roi.crop(image))

        # subtract background
        wavefront -= self._get_background_wavefront()

        # set baseline wavefront to 0, using top and bottom rows as reference
        wavefront -= (wavefront[[0, -1], :]).mean()

        # 2d array of electron density with units [length]^-3
        # center row ((num_rows - 1) // 2) represents cylinder axis.
        density = self.qwlsi_image_analyzer.calculate_density(wavefront, wavelength=self.laser_wavelength, image_resolution=self.image_resolution)
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
                    if lineout[x_max_downramp + sigma] < np.exp(-0.5) * max_density_downramp:
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

        try:
            density_lineout_fit_result = dict(zip(['A1', 'x1', 'w',  'A2', 'x2', 'sigma',  'x3', 'x4'], 
                                                fit_density_lineout(center_density_lineout)
                                            ))
        except ValueError as err:
            warn(f"Error during fit_density_lineout: {err}")
            density_lineout_fit_result = {}

        # Compile analysis results
        analysis_results = {'wavefront_nm': wavefront.m_as('nm'), 
                            'density_map_cm-3': density.m_as('cm^-3'),
                            'density_lineout_cm-3': center_density_lineout.m_as('cm^-3'),
                            'peak_density_cm-3': center_density_lineout.max().m_as('cm^-3'),
                           }
        
        if density_lineout_fit_result:
            analysis_results.update({
                'density_lineout_fit_A1_cm-3': density_lineout_fit_result['A1'].m_as('cm^-3'),
                'density_lineout_fit_x1_mm': (density_lineout_fit_result['x1'] * self.qwlsi_image_analyzer.CAMERA_RESOLUTION/ureg.px).m_as('mm'),
                'density_lineout_fit_w_mm': (density_lineout_fit_result['w'] * self.qwlsi_image_analyzer.CAMERA_RESOLUTION/ureg.px).m_as('mm'),
                'density_lineout_fit_A2_cm-3': density_lineout_fit_result['A1'].m_as('cm^-3'),
                'density_lineout_fit_x2_mm': (density_lineout_fit_result['x2'] * self.qwlsi_image_analyzer.CAMERA_RESOLUTION/ureg.px).m_as('mm'),
                'density_lineout_fit_sigma_mm': (density_lineout_fit_result['sigma'] * self.qwlsi_image_analyzer.CAMERA_RESOLUTION/ureg.px).m_as('mm'),
                'density_lineout_fit_x3_mm': (density_lineout_fit_result['x3'] * self.qwlsi_image_analyzer.CAMERA_RESOLUTION/ureg.px).m_as('mm'),
                'density_lineout_fit_x4_mm': (density_lineout_fit_result['x4'] * self.qwlsi_image_analyzer.CAMERA_RESOLUTION/ureg.px).m_as('mm'),
            })
        else:
            analysis_results.update({
                'density_lineout_fit_A1_cm-3': 0.0,
                'density_lineout_fit_x1_mm': 0.0,
                'density_lineout_fit_w_mm': 0.0,
                'density_lineout_fit_A2_cm-3': 0.0,
                'density_lineout_fit_x2_mm': 0.0,
                'density_lineout_fit_sigma_mm': 0.0,
                'density_lineout_fit_x3_mm': 0.0,
                'density_lineout_fit_x4_mm': 0.0,
            })

        return analysis_results

    def _get_background_wavefront(self):
        # get background
        # first check if it's explicitly set
        if self.background is not None:
            return self.background

        # next check for background_path
        elif (self.background_path and self.background_path != Path('nul')):
            if (self.background_path, self.roi) not in self.background_cache:
                self.background_cache[(self.background_path, self.roi)] = self.calculate_background_from_path()
            return self.background_cache[(self.background_path, self.roi)]

        else:
            if self.on_no_background == 'ignore':
                return 0

            elif self.on_no_background == 'raise':
                raise ValueError("No background wavefront. Use set_background(wavefront), where wavefront is "
                                 "obtained with calculate_wavefront(interferogram)"
                                )
            elif self.on_no_background == 'warn':
                warn("No background wavefront. Returning wavefront with no background subtraction.")
                return 0

            else:
                raise ValueError(f"Unknown value for on_no_background: {self.on_no_background}. Should be one of 'raise', 'warn', or 'ignore'")

    def _initialize_qwlsi_image_analyzer(self):
        self.qwlsi_image_analyzer = QWLSIImageAnalyzer(
            reconstruction_method='velghe',
            camera_resolution=self.camera_resolution,
            grating_camera_distance=self.grating_camera_distance,
            grating_period=self.grating_period,
            camera_tilt=self.camera_tilt,
        )