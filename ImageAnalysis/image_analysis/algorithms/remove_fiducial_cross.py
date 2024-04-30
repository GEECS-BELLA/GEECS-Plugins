from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Union
from warnings import warn

import cv2 as cv
import numpy as np
from scipy.optimize import minimize

from tau_inpainter.algorithms.total_curvature import TotalCurvatureInpainter
import networkx as nx

DEG = np.pi/180

def tapered_plateau(x, x1, x2, taper=1):
    """ A function that is 1 between x1 and x2, and tapers off quickly to 0 outside of this range.
    """
    def tapered_plateau_one_side(x, x1):
        return 1 / (1 + np.exp(-(x - x1) / (0.25 * taper)))
    
    return tapered_plateau_one_side(x, x1) * (1 - tapered_plateau_one_side(x, x2))

def gaussian(x, x1, fwhm):
    return np.exp(-np.square(x - x1) * 4 * np.log(2) / fwhm**2)

# order=True is needed for comparison operators to work
# frozen=True is needed to make the class hashable
@dataclass(order=True, frozen=True)
class FiducialCross:
    """ A dataclass to store information about a fiducial cross.

    Attributes
    ----------
    x : float
        The x-coordinate of the center of the cross.
    y : float
        The y-coordinate of the center of the cross.
    length : float
        The length of the cross, i.e. twice the length of each of the 4 arms.
    thickness : float
        The thickness of the cross arms, as a full-width half-maximum.
    angle : float
        The angle that the cross makes with reference to a plus sign arrangement.
    """

    x: float
    y: float
    length: float
    thickness: float
    angle: float

    def coordinates_in_cross_frame(self, shape: tuple[int, int], scale: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """ Get the coordinates of the cross in a frame centered at the cross center, 
        aligned with the cross arms.
        
        Parameters
        ----------
        shape : tuple[int, int]
            The shape of the image. The coordinates will be generated for this shape.
        scale : bool, optional
            TODO: implement this feature
            If True, the coordinates will be scaled to the length of the cross arms.
        """

        if scale:
            warn("The scale parameter is not implemented yet.")

        X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        Rinv = np.array([[np.cos(-self.angle), -np.sin(-self.angle)],
                         [np.sin(-self.angle),  np.cos(-self.angle)]
                        ]
                    )

        # Xrot and Yrot give coordinates of a system centered at cross center, 
        # aligned with the cross arms. i.e. the cross is located along the x and 
        # y axes of this system, centered where Xrot and Yrot are (0, 0).
        Xrot, Yrot = np.dot(Rinv, np.stack([X - self.x, Y - self.y], axis=1))

        return Xrot, Yrot

    def image(self, shape: tuple[int, int]) -> np.ndarray:
        """ Generate an image of the cross.

        Parameters
        ----------
        shape : tuple[int, int]
            The shape of the image to generate.

        Returns
        -------
        np.ndarray
            An image of the cross.
        """
        Xrot, Yrot = self.coordinates_in_cross_frame(shape, scale=False)

        cross = (1 - 
                   (1 - gaussian(Xrot, 0.0, self.thickness) * tapered_plateau(Yrot, -self.length/2, self.length/2))
                 * (1 - gaussian(Yrot, 0.0, self.thickness) * tapered_plateau(Xrot, -self.length/2, self.length/2))
                )

        return cross

@dataclass
class HoughLine:
    """ A line detected by cv.HoughLines

    The cv.HoughLines algorithm returns detected lines in the form of two parameters 
    rho and theta, as the line that connects to and is perpendicular to the vector 
    from the origin of length rho and angle theta. 

    Attributes
    ----------
    rho : float
        distance from origin to Hough line
    theta : float
        angle of perpendicular line from origin to Hough line, in radians between 
        0 and pi
    """
    rho: float
    theta: float

    def normalize(self) -> HoughLine:
        """ Returns polar coordinates of this line with theta in range 0..pi
        
        This may give negative rho
        """
        theta_norm = np.mod(self.theta, 2*np.pi)
        if theta_norm >= np.pi:
            return HoughLine(-self.rho, theta_norm - np.pi)
        else:
            return HoughLine(self.rho, theta_norm)

@dataclass
class HoughLinesPair:
    """ Two Hough lines that are parallel and close to each other
    """

    line1: HoughLine
    line2: HoughLine
    
    @property
    def midline(self) -> HoughLine:
        """ "Average" of the two lines in this pair
        
        TODO: better implementation of "average" HoughLine, which could be the 
        line that bisects the angle formed by two non-parallel lines, or the line 
        down the center of parallel lines
        """

        # get the x,y coordinates of the points at (rho, theta) polar coordinates
        line1_x, line1_y = (self.line1.rho * np.cos(self.line1.theta), self.line1.rho * np.sin(self.line1.theta))
        line2_x, line2_y = (self.line2.rho * np.cos(self.line2.theta), self.line2.rho * np.sin(self.line2.theta))
        midline_x, midline_y = (line1_x + line2_x) / 2, (line1_y + line2_y) / 2

        return HoughLine(np.sqrt(midline_x**2 + midline_y**2), np.arctan2(midline_y, midline_x)).normalize()

@dataclass
class HoughLinesQuartet:
    """ Two pairs of parallel Hough lines that are perpendicular to each other
    """

    parallel_pair1: HoughLinesPair
    parallel_pair2: HoughLinesPair

class FiducialCrossRemover:
    """ An algorithm to detect fiducial crosses in images and inpaint them.

    """

    def __init__(self, 
                 canny_min_threshold: int = 20,
                 canny_max_threshold: int = 150,
                 hough_threshold: int = 50,
                 hough_rho_resolution: float = 1.0,
                 hough_theta_resolution: float = 2*DEG,
                 parallel_hough_lines_max_distance: float = 7.0,
                 parallel_hough_lines_max_angle: float = 1*DEG,
                 perpendicular_hough_lines_max_angle: float = 1*DEG,
                ):
        """ 
        Canny parameters: see https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
        Hough parameter: see https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
        
        Parameters
        ----------
        canny_min_threshold : int, optional
        canny_max_threshold : int, optional
        hough_threshold : int, optional
        parallel_hough_lines_max_distance : float, optional
            The maximum distance between two parallel lines to be considered a pair.
        parallel_hough_lines_max_angle : float, optional
            The maximum angle between two parallel lines to be considered a pair.
        perpendicular_hough_lines_max_angle : float, optional
            The maximum angle between two perpendicular pairs of parallel lines.
        """

        self.canny_min_threshold = canny_min_threshold
        self.canny_max_threshold = canny_max_threshold
        self.hough_threshold = hough_threshold
        self.hough_rho_resolution = hough_rho_resolution
        self.hough_theta_resolution = hough_theta_resolution
        self.parallel_hough_lines_max_distance = parallel_hough_lines_max_distance
        self.parallel_hough_lines_max_angle = parallel_hough_lines_max_angle
        self.perpendicular_hough_lines_max_angle = perpendicular_hough_lines_max_angle

    def detect_crosses(self, 
                       image: np.ndarray,
                       approximate_locations: list[tuple[float, float]] = None,
                       approximate_length: float = None,
                      ) -> list[FiducialCross]:
        """ Detect fiducial crosses in the image.

        Parameters
        ----------
        image : np.ndarray
        approximate_locations : list[tuple[int, int]], optional
            TODO: implement this feature
            A list of approximate locations of crosses in the image, in the form 
            of (x, y) tuples. The algorithm is able to detect crosses without 
            initial locations, and it will return as many as it confidently 
            detects, but giving approximate locations can help prevent false 
            positives.
        approximate_length : float, optional
            The approximate length of the crosses. This can help the algorithm 
            detect crosses more accurately.
        
        Returns
        -------
        list[FiducialCross]
            A list of detected fiducial crosses.
        """

        if approximate_locations is not None:
            warn("The approximate_locations parameter is not implemented yet.")

        # Detect lines in the image
        canny =  cv.Canny(image, self.canny_min_threshold, self.canny_max_threshold)
        houghlines: Union[np.ndarray, None] = cv.HoughLines(canny, self.hough_rho_resolution, self.hough_theta_resolution, self.hough_threshold)
        # if none found, HoughLines returns None instead of empty array
        if houghlines is None: 
            houghlines = np.ndarray((0, 1, 2), dtype=float)
        
        # cv.HoughLines returns a N x 1 x 2 array. Convert this into list of HoughLine objects
        lines: list[HoughLine] = [HoughLine(rho, theta) 
                                  for rho, theta 
                                  in houghlines[:, 0, :]
                                 ]

        def lines_are_parallel(line1, line2):
            """ Check if hough lines are parallel
            
            Check if rhos and thetas are close numerically. Take into account that 
            parallel lines' rho1 and rho2 may be opposite sign with thetas separated 
            by π if they are close to theta = 0
            """
            
            def rho_and_theta_close(rho1, theta1, rho2, theta2) -> bool:
                """ Check that rhos and thetas are close
                """
                return (    np.abs(line2.rho - line1.rho) <= self.parallel_hough_lines_max_distance
                        and np.abs(line1.theta - line2.theta) <= self.parallel_hough_lines_max_angle
                       )
            
            return (   rho_and_theta_close(line1.rho, line1.theta, line2.rho, line2.theta)
                    or rho_and_theta_close(-line1.rho, np.mod(line1.theta + np.pi, 2*np.pi), line2.rho, line2.theta)
                   ) 
            
    
        # find lines that are parallel and close to each other
        parallel_pairs = [HoughLinesPair(line1, line2)
                          for line1, line2 in product(lines, lines)
                          if lines_are_parallel(line1, line2)
                             and line1.rho > line2.rho  # prevent line1 matching itself, and prevent duplicate (line1, line2) pair
                         ]

        def compute_intersection(line1: HoughLine, line2: HoughLine) -> tuple[float, float]:
            x = (line2.rho * np.sin(line1.theta) - line1.rho * np.sin(line2.theta)) / np.sin(line1.theta - line2.theta)
            y = (line1.rho * np.cos(line2.theta) - line2.rho * np.cos(line1.theta)) / np.sin(line1.theta - line2.theta)
            return x, y

        def in_image(x, y):
            return (0 <= x < image.shape[1]) and (0 <= y < image.shape[0])

        # find pairs of parallel line pairs that are perpendicular to each other
        # and intersect within the boundaries of the image
        perpendicular_quartets = [HoughLinesQuartet(parallel_pair1, parallel_pair2)
                                  for parallel_pair1, parallel_pair2 in product(parallel_pairs, parallel_pairs)
                                  if (0.0 <= np.abs((parallel_pair1.midline.theta - parallel_pair2.midline.theta) - 90*DEG) < self.perpendicular_hough_lines_max_angle)
                                     and in_image(*compute_intersection(parallel_pair1.midline, parallel_pair2.midline))
                                 ]

        # TODO: estimate length and thickness of crosses
        def cross_from_perpendicular_quartet(perpendicular_quartet: HoughLinesQuartet) -> FiducialCross:
            x, y = compute_intersection(perpendicular_quartet.parallel_pair1.midline, perpendicular_quartet.parallel_pair2.midline)
            angle = min(perpendicular_quartet.parallel_pair1.midline.theta, perpendicular_quartet.parallel_pair2.midline.theta)
            thickness = abs(perpendicular_quartet.parallel_pair1.line2.rho - perpendicular_quartet.parallel_pair1.line1.rho)
            length = self._estimate_cross_length(image, x, y, angle, thickness, length0 = approximate_length)

            return FiducialCross(x, y, 
                                 length=length,
                                 thickness=thickness,  # TODO: replace by estimated thickness
                                 angle=angle
                                )

        crosses =  [cross_from_perpendicular_quartet(perpendicular_quartet)
                    for perpendicular_quartet in perpendicular_quartets
                   ]

        crosses = self._deduplicate_crosses(crosses)

        return crosses        

    # def _fit_plateau(self, x: np.ndarray, y: np.ndarray, x_mid_0: float, width_0: float) -> tuple[float, float]:
    #     """ Fit a plateau function to the data by cross-correlation.

    #     Parameters
    #     ----------
    #     x : np.ndarray
    #         The x-values of the data.
    #     y : np.ndarray
    #         The y-values of the data.
    #     x_mid_0 : float
    #         The initial guess for the plateau center.
    #     width_0 : float
    #         The initial guess for the plateau width.

    #     Returns
    #     -------
    #     x_mid_fit, width_fit : tuple[float, float]
    #         The fitted plateau center and width.
    #     """
    #     y_mean_subtracted = y - y.mean()
    #     y_std = y.std()
    #     def crosscorr(params):
    #         x_mid, width = params
    #         y_plateau = tapered_plateau(x, x_mid - width/2, x_mid + width/2)

    #         cov = np.mean(y_mean_subtracted * (y_plateau - y_plateau.mean()))
    #         return cov / (y_std * y_plateau.std())

    #     sol = minimize(lambda p: -crosscorr(p), [x_mid_0, width_0])
    #     x_mid_fit, width_fit = sol.x
    #     return x_mid_fit, width_fit

    def _fit_plateau_width(self, x: np.ndarray, y: np.ndarray, width_0: float, x_mid: float = 0.0) -> float:
            """ Fit a plateau function with given center to the data by cross-correlation.

            Parameters
            ----------
            x : np.ndarray
                The x-values of the data.
            y : np.ndarray
                The y-values of the data.
            width_0 : float
                The initial guess for the plateau width.
            x_mid : float, optional
                Known center of the plateau, default 0.0 (i.e. x array is pre-shifted)

            Returns
            -------
            float
                The fitted plateau width.
            """
            def mean(y_):
                return np.trapz(y_, x) / (x[-1] - x[0])
            def std(y_):
                return np.trapz(np.square(y_ - mean(y_)), x) / (x[-1] - x[0])
            def cov(y1, y2):
                return np.trapz((y1 - mean(y1)) * (y2 - mean(y2)), x) / (x[-1] - x[0])

            y_std = std(y)

            def crosscorr(params):
                (width,) = params
                y_plateau = tapered_plateau(x, x_mid - width/2, x_mid + width/2)
                return cov(y, y_plateau) / (y_std * std(y_plateau))

            sol = minimize(lambda p: -crosscorr(p), [width_0], bounds=[(0, None)])
            (width_fit,) = sol.x
            return width_fit


    def _estimate_cross_length(self, image: np.ndarray, x: float, y: float, angle: float, width: float = 1.0, length0: float = None) -> float:
        """Estimates the length of the cross in the given image.

        Length here means end-to-end of the two lines that make up the cross.

        Parameters
        ----------
        image : np.ndarray
            The input image.
        x : float
            The x-coordinate of the cross center.
        y : float
            The y-coordinate of the cross center.
        angle : float
            The angle 
        width : float, optional
            The thickness of the cross arms. The data for fitting is taken from 
            a region of this width around each cross arm. Default 1.0
        length0 : float, optional
            Initial guess for the length of the cross.

        Returns
        -------
        float
            The estimated length of the cross.
        """
        
        # Get the coordinates of the cross in a frame centered at the cross center, 
        # aligned with the cross arms.
        Xrot, Yrot = FiducialCross(x, y, 1.0, width, angle).coordinates_in_cross_frame(image.shape, scale=False)

        xs = []; ys = []
        # get values close to the horizontal arms. Put the distance from center
        # along the horizontal arms into x values for fitting.
        s = (np.abs(Yrot) < width / 2)
        xs.append(Xrot[s])
        ys.append(image[s])

        # get values close to the vertical arms. Put the distance from center
        # along the vertical arms into x values for fitting.
        s = (np.abs(Xrot) < width / 2)
        xs.append(Yrot[s])
        ys.append(image[s])

        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        s = np.argsort(xs)
        xs = xs[s]
        ys = ys[s]

        # note plateau function width corresponds to cross length
        if length0 is None:
            length0 = (xs[-1] - xs[0]) / 2

        # Find the plateau width by fitting a plateau function to the image data
        width = self._fit_plateau_width(xs, ys, length0)

        return width

    def _deduplicate_crosses(self, crosses: list[FiducialCross]):
        """ Consolidate crosses at similar location and angle 
        
        Generate graph with crosses as vertices and similarity as links, then 
        find the disjoint groups and produce one average cross.

        """
        # Create a graph with crosses at vertices
        G = nx.Graph()
        G.add_nodes_from(crosses)

        # connect vertices representing similar crosses
        def crosses_are_similar(cross1: FiducialCross, cross2: FiducialCross) -> bool:
            return (
                    ((cross1.x - cross2.x)**2 + (cross1.y - cross2.y)**2 <= (1.1 * np.sqrt(2) * self.hough_rho_resolution)**2)
                and (np.abs(cross1.angle - cross2.angle) <= 2 * self.hough_theta_resolution)
            )

        G.add_edges_from((cross1, cross2) for cross1, cross2 in product(crosses, crosses) 
                         if cross1 < cross2 and crosses_are_similar(cross1, cross2)
                        )

        # Compute average cross for each group
        def compute_average_cross(crosses: set[FiducialCross]) -> FiducialCross:

            def mean_angle(angles: np.ndarray) -> float:
                return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
            
            return FiducialCross(
                x = np.mean([cross.x for cross in crosses]),
                y = np.mean([cross.y for cross in crosses]),
                length = np.mean([cross.length for cross in crosses]),
                thickness = np.mean([cross.thickness for cross in crosses]),
                angle = mean_angle(np.array([cross.angle for cross in crosses]))
            )

        average_crosses = []
        for connected_group in nx.connected_components(G):
            average_cross = compute_average_cross(connected_group)
            average_crosses.append(average_cross)

        return average_crosses

    def inpaint_crosses(self, 
                        image: np.ndarray,
                        crosses: list[FiducialCross],
                       ) -> np.ndarray:
        """ Inpaint fiducial crosses in the image.

        Parameters
        ----------
        image : np.ndarray
        crosses : list[FiducialCross]
            A list of fiducial crosses to inpaint.
        
        Returns
        -------
        np.ndarray
            The inpainted image.
        """

        inpainter = TotalCurvatureInpainter(image.shape)
        
        mask = np.zeros(image.shape, dtype=bool)
        for cross in crosses:
            expanded_cross = FiducialCross(cross.x, cross.y, 1.10 * cross.length, 1.10 * cross.thickness, cross.angle)
            mask |= (expanded_cross.image(image.shape) > 0.01)

        return inpainter.inpaint(image, mask)


    def detect_and_inpaint_crosses(self, 
                                   image: np.ndarray,
                                   approximate_locations: list[tuple[float, float]] = None,
                                   approximate_length: float = None,
                                  ) -> np.ndarray:
        """ Detect and inpaint fiducial crosses in the image.

        Parameters
        ----------
        image : np.ndarray
        approximate_locations : list[tuple[int, int]], optional
            A list of approximate locations of crosses in the image, in the form 
            of (x, y) tuples. The algorithm is able to detect crosses without 
            initial locations, and it will return as many as it confidently 
            detects, but giving approximate locations can help prevent false 
            positives.
        approximate_length : float, optional
            The approximate length of the crosses. This can help the algorithm 
            detect crosses more accurately.
        
        Returns
        -------
        np.ndarray
            The inpainted image.
        """

        crosses = self.detect_crosses(image, approximate_locations, approximate_length)
        return self.inpaint_crosses(image, crosses)
