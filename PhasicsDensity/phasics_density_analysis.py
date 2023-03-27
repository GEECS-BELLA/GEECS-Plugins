# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:44:15 2023

@author: ReiniervanMourik
"""

#%% init
from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Annotated

from itertools import product

import numpy as np

from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import lsqr

from skimage.restoration import unwrap_phase

from pint import UnitRegistry
if TYPE_CHECKING:
    from pint import Quantity
    SpatialFrequencyQuantity = Annotated[Quantity, '[length]**-1]']
    LengthQuantity = Annotated[Quantity, '[length]']

ureg = UnitRegistry()
Q_ = ureg.Quantity


#%% PhasicsImageAnalyzer class

class PhasicsImageAnalyzer:
    """ An engine that can analyze a Phasics image and return its phase map.

    General usage, with `img` an image from the HTU Gasjet Phasics camera 
       pia = PhasicsImageAnalyzer()
       phase_map = pia.calculate_phase_map(pia.crop_image(img))

              
    Methods
    -------
    calculate_phase_map(img)
        runs all steps of the phase map reconstruction from a Phasics image img.

    """

    CAMERA_RESOLUTION = Q_(4 * 4.74, 'micrometer')
    GRATING_CAMERA_DISTANCE = Q_(1.0, 'centimeter')  # this is a wild guess
    
    def __init__(self,
                 reconstruction_method = 'baffou',
                 diffraction_spot_crop_radius: Optional[Quantity] = None
                ):
        """ 
        Parameters
        ----------
        reconstruction_method : 'baffou' or 'velghe'
            which method to use for the step of combining the various spot FTs
            into a final phase map.
            'baffou' recovers phase map gradients in different directions and then
            integrates them
            'velghe' (not yet implemented) solves for the FT of the phase map

        diffraction_spot_crop_radius : Quantity with [length]^-1 units
            radius of disc around spot center to use for each spot's FT.
            If None, find the maximum radius that causes no overlap.

        """

        self.reconstruction_method = reconstruction_method
        self.diffraction_spot_crop_radius = diffraction_spot_crop_radius
    
    def crop_image(self, img: np.ndarray) -> np.ndarray:
        """ Crops for HTU Gasjet Phasics images
        
            Cuts out the jet blade. 
        """
        return img[:600, 600:]
    
    
    class Center:
        def __init__(self, parent: PhasicsImageAnalyzer, row: int, column: int):
            self.parent = parent
            self.row = row
            self.column = column
        
        @property
        def nu_x(self) -> SpatialFrequencyQuantity:
            """ x coordinate of center, in [length]^-1
            """
            return self.parent.freq_x[self.column]
 
        @property
        def nu_y(self) -> SpatialFrequencyQuantity:
            return self.parent.freq_y[self.row]
    
    def new_center(self, row: int, column: int):
        return self.Center(self, row, column)
    
    def _fourier_transform(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Takes the fourier transform of an image and shifts it.

        Returns
        -------
        IMG : np.ndarray
            Fourier-transformed image, with freq = 0,0 in the middle
        freq_x : np.ndarray
            array of frequencies in the axis=1 direction
        freq_y : np.ndarray
            array of frequencies in the axis=0 direction
        
        """
        fftshift = ureg.wraps('=A', ('=A', None))(np.fft.fftshift)
        self.IMG = np.fft.fftshift(np.fft.fft2(self.img))
        self.freq_x = fftshift(np.fft.fftfreq(self.shape[1], d=self.CAMERA_RESOLUTION))
        self.freq_y = fftshift(np.fft.fftfreq(self.shape[0], d=self.CAMERA_RESOLUTION))
        
        return self.IMG, self.freq_x, self.freq_y
    
    
    def _locate_diffraction_spots(self) -> list[tuple[int, int]]:
        """ Find centers of diffraction spot in Fourier-transformed QWLSI image
        
        Right now, just returns the centers for the HTU GasJet image cropped to [:600, 600:]
        
        TODO: generalize this.

        Parameters
        ----------
        IMG : np.ndarray
            Fourier-transformed image.

        Returns
        -------
        list[tuple[int, int]]
            list of row/column indices to diffraction spot centers in image

        """
        self.diffraction_spot_centers = [self.new_center(504, 589), 
                                         self.new_center(375, 715), 
                                         self.new_center(246, 841), 
                                         self.new_center(171, 626)
                                        ]
        
        return self.diffraction_spot_centers
    
    def _set_diffraction_spot_crop_radius(self) -> SpatialFrequencyQuantity:
        """ Calculate minimum distance between any two centers
        """
        self.diffraction_spot_crop_radius = np.sqrt( 
            min((center2.nu_x - center1.nu_x)**2 + (center2.nu_y - center1.nu_y)**2
                for center1, center2 in product(self.diffraction_spot_centers, self.diffraction_spot_centers)
                if center1 is not center2
               )
        ) / 2
        
        return self.diffraction_spot_crop_radius
  


    
    def _crop_and_center_diffraction_spots(self) -> list[np.ndarray]:

        # if diffraction spot crop radius not given, infer it
        if self.diffraction_spot_crop_radius is None:
            self._set_diffraction_spot_crop_radius()
              
        def _crop_and_center_diffraction_spot(center: self.Center) -> np.ndarray:
            """ Crop an area of freq space around diffraction spot center, and translate it
                to the middle of the image.
            
            Parameters
            ----------
            IMG : np.ndarray
                Fourier-transformed QWLSI image.
            center : tuple[int, int]
                row, column of diffraction spot center
    
            Returns
            -------
            G_i : np.ndarray
                image of same size as IMG, with diffraction spot in center and all
                information not related to this spot cropped away.
    
            """
            
            # np.roll won't work on a Quantity array, so create a unit-aware version
            roll_ua = ureg.wraps('=A', ('=A', None, None))(np.roll)
            
            NU_X, NU_Y = np.meshgrid(self.freq_x, self.freq_y)
            IMG_cropped = self.IMG * ((np.square(NU_X - center.nu_x) + np.square(NU_Y - center.nu_y)) < self.diffraction_spot_crop_radius**2)
            IMG_recentered = roll_ua(IMG_cropped, (self.shape[0]//2 - center.row, self.shape[1]//2 - center.column), (0, 1))
            return IMG_recentered


        self.diffraction_spot_IMGs = [_crop_and_center_diffraction_spot(center)
                                      for center in self.diffraction_spot_centers
                                     ]
        
        return self.diffraction_spot_IMGs
    
    
    def _reconstruct_phase_gradient_maps_from_cropped_centered_diffraction_FTs(self) -> list[np.ndarray]:
        """ Calculate the angle (argument) of the inverse FT of a diffraction 
            spot image.
            
            For Baffou reconstruction method, the phase gradient in a specific
            direction is in the argument of the inverse FT. 
        
        """

        self.phase_gradient_maps = [unwrap_phase(np.angle(np.fft.ifft2(np.fft.ifftshift(IMG.m))))
                                    for IMG in self.diffraction_spot_IMGs
                                   ] 

        return self.phase_gradient_maps


    def _rotate_phase_gradient_maps(self) -> list[np.ndarray]:
        """ Rotates the phase gradient maps 
        
        

        Returns
        -------
        None.

        """
    
    
    def _integrate_gradient_maps(self) -> np.ndarray:
        """ Calculates the phase map from gradients in different directions.
        
        More precisely, this finds the phase map whose gradients in different directions
        most closely match the given gradient maps. It constructs a matrix representing 
        finite differences in each gradient direction for each pixel and solves the least
        squares matrix equation.

        Returns
        -------
        phase map : np.ndarray

        """
        
        # Construct sparse matrix specifying the gradients in each direction for each pixel
        # for the equation A.x = b. 
        # Each row of A, and its corresponding value in b, represents the gradient in one particular
        # direction for one particular pixel of the phase map. The row is constructed by taking a 2D
        # image of zeros except for a few finite difference coefficients around a specific pixel, then
        # flattening it. 

        m = 0
        data = []
        row_ind = []
        col_ind = []

        b = []

        def to_flattened_index(i, j):
            return i * self.shape[1] + j

        for center, phase_gradient_map in zip(self.diffraction_spot_centers, self.phase_gradient_maps):
            # finite_difference_coefficients = np.array(
            #     [[ 0.0,               -center.nu_y / 2,       0.0       ],
            #      [ -center.nu_x / 2,         0.0       center.nu_x / 2  ],
            #      [ 0.0                 center.nu_y / 2,       0.0       ]
            #     ]
            # ) * 2*np.pi * self.GRATING_CAMERA_DISTANCE

            finite_difference_coefficients = [((-1, 0), (-2*np.pi * self.GRATING_CAMERA_DISTANCE * center.nu_y / 2).m_as('')),
                                              ((0, -1), (-2*np.pi * self.GRATING_CAMERA_DISTANCE * center.nu_x / 2).m_as('')),
                                              ((0, 1),  ( 2*np.pi * self.GRATING_CAMERA_DISTANCE * center.nu_x / 2).m_as('')),
                                              ((1, 0),  ( 2*np.pi * self.GRATING_CAMERA_DISTANCE * center.nu_y / 2).m_as('')),
                                             ]

            for i in range(1, self.shape[0] - 1):
                for j in range(1, self.shape[1] - 1):
                    for (di, dj), coeff in finite_difference_coefficients:
                        data.append(coeff)
                        row_ind.append(m)
                        col_ind.append(to_flattened_index(i + di, j + dj))
                    b.append(phase_gradient_map[i, j])
                    m += 1

        # The least squares loss is invariant under adding a constant to the entire phase map, so add a row to the list of 
        # equations requiring that the mean of the whole map is 0. 

        for j in range(self.shape[0] * self.shape[1]):
            data.append(1.0)
            row_ind.append(m)
            col_ind.append(j)
        b.append(0.0)

        A = csr_matrix((data, (row_ind, col_ind)), shape=(len(self.phase_gradient_maps) * (self.shape[0] - 2) * (self.shape[1] - 2) + 1, 
                                                          self.shape[0] * self.shape[1]
                                                         )
                      )

        # solve the linear equation in the least squares sense.
        self.phase_map = lsqr(A, b)[0].reshape(self.shape)

        return self.phase_map

    
    def _reconstruct_phase_map_FT_from_gradient_FTs(self):
        """ Fit phase map whose FT best corresponds to its gradient FTs
        
        From the method in 
            Velghe, Sabrina, Jérôme Primot, Nicolas Guérineau, Mathieu Cohen, 
            and Benoit Wattellier. "Wave-Front Reconstruction from Multidirectional 
            Phase Derivatives Generated by Multilateral Shearing Interferometers." 
            Optics Letters 30, no. 3 (February 1, 2005): 245. 
            https://doi.org/10.1364/OL.30.000245.

        This method solves for FT(W), where W is the phase map, by minimizing
        the error between each gradient phase map FT G_j and the gradient of W
        in the Fourier domain, 2*pi*i*u_j*W, where u_j is the conjugate of 
        the spatial coordinate. 
        
        The phase map is then simply the inverse FT of the best fit.

        """
        
        # Calculate the u_j. These are calculated as  nu . v, where nu is the
        # coordinates in the fourier domain, and v is the vector of the 
        # gradient in the spatial domain.
        NU_X, NU_Y = np.meshgrid(self.freq_x, self.freq_y)
        U = [  NU_X * 2 * np.pi * self.GRATING_CAMERA_DISTANCE * center.nu_x 
             + NU_Y * 2 * np.pi * self.GRATING_CAMERA_DISTANCE * center.nu_y
             for center in self.diffraction_spot_centers
            ]

        # calculate FTs of phase gradient maps
        G = [np.fft.fftshift(np.fft.fft2(pgm)) for pgm in self.phase_gradient_maps]

        # Solve for FT(W)_e, the estimate of the FT of the phase map.
        W_ft = (-1j/(2*np.pi) * sum(u * g for u, g in zip(U, G)) 
                              / sum(np.square(u) for u in U)
               )

        # W_ft is a Quantity array, which complicates the inversion. 
        # For now, just take the magnitude.
        # TODO: account for units.
        W_ft = W_ft.m
        # W_ft has axes -Ny..Ny x -Ny..Ny. For the ifft2, shift it back to 
        # 0..2*Ny x 0..2*Ny
        W_ft = np.fft.ifftshift(W_ft)
        # after shifting, the value corresponding to freq_x = freq_y = 0 should
        # be at 0, 0, and its value should be NaN because of the division by 
        # zero in estimating W_ft. 
        assert np.isnan(W_ft[0, 0])
        # setting it to 0 ensures that the mean of the phase map is 0. 
        W_ft[0, 0] = 0.0
        
        W = np.fft.ifft2(W_ft)
    
        return W.real
    

    
    def calculate_phase_map(self, img: np.ndarray) -> np.ndarray:
        """ Analyze cropped Phasics quadriwave shearing image
        
        Takes a cropped image and runs the full algorithm on it to obtain the
        reconstructed phase map. 

        Parameters
        ----------
        img : np.ndarray
            Phasics image, already cropped to region of interest.

        Returns
        -------
        phase map : np.ndarray
            reconstructed phase in radians.

        """


        # take 2D Fourier transform of image, shifted so that freq = 0, 0 is 
        # in the middle of the image.
        self.img = img
        self.shape = img.shape
        self._fourier_transform()

        # locate the 4 diffraction spots
        self._locate_diffraction_spots()

        # get cropped and centered FTs of each diffraction spot
        self._crop_and_center_diffraction_spots()

        self._reconstruct_phase_gradient_maps_from_cropped_centered_diffraction_FTs()

        # reconstruct phase map from diffraction spot FTs
        if self.reconstruction_method == 'baffou':
            W = self._integrate_gradient_maps()

        elif self.reconstruction_method == 'velghe':
            W = self._reconstruct_phase_map_FT_from_gradient_FTs()

        else:
            raise ValueError(f"Unknown reconstruction method: {self.reconstruction_method}")

        return W

