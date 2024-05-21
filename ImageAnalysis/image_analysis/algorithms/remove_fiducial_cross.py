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

@dataclass
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
        The angle that the cross makes with reference to a plus sign arrangement, 
        between -pi/4 and pi/4
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

# The HoughLine classes need to be ordered and frozen so that HoughLinesQuartet 
# is hashable and sortable for deduplication
@dataclass(order=True, frozen=True)
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

@dataclass(order=True, frozen=True)
class HoughLinesPair:
    """ Two Hough lines that are parallel and close to each other
    """

    line1: HoughLine
    line2: HoughLine
    
    @property
    def midline(self) -> HoughLine:
        """ "Average" of the two lines in this pair
        """

        # exactly parallel
        line1_norm, line2_norm = self.line1.normalize(), self.line2.normalize()
        if line1_norm.theta == line2_norm.theta:
            return HoughLine((line1_norm.rho + line2_norm.rho) / 2, line1_norm.theta)

        # otherwise, find the angle that bisects the intersection angle.
        # first get the angle between them in the range -pi..pi
        intersection_angle = np.mod(self.line2.theta - self.line1.theta + np.pi, 2*np.pi) - np.pi
        if abs(intersection_angle) == np.pi/2:
            raise ValueError("Midline of perpendicular lines is ambiguous")
        
        if abs(intersection_angle) > np.pi/2:
            # flip representation of one of the Houghlines to get instersection 
            # angle between -pi/2..pi/2
            return HoughLinesPair(self.line1, HoughLine(-self.line2.rho, self.line2.theta - np.pi)).midline

        theta = (self.line1.theta + self.line2.theta) / 2

        # rho is the length of the projection of the intersection point onto the 
        # vector from origin with angle theta.
        x, y = compute_houghline_intersection(self.line1, self.line2)
        rho = x * np.cos(theta) + y * np.sin(theta)

        return HoughLine(rho, theta).normalize()

@dataclass(order=True, frozen=True)
class HoughLinesQuartet:
    """ Two pairs of parallel Hough lines that are perpendicular to each other
    """

    parallel_pair1: HoughLinesPair
    parallel_pair2: HoughLinesPair

    @property
    def center(self) -> tuple[float, float]:
        return compute_houghline_intersection(self.parallel_pair1.midline, self.parallel_pair2.midline)

    @property
    def angle(self) -> float:
        """ Calculate the angle this perpendicular quartet makes with the x-axis

        TODO: distinguish this from theta

        Returns the angle almost always in the range -pi/4 to pi/4

        """
        # Find the angle of whichever of the four arms is closest to the positive
        # x-axis. Note midline.theta is in the range 0..pi
        return min([self.parallel_pair1.midline.theta, 
                    self.parallel_pair2.midline.theta, 
                    self.parallel_pair1.midline.theta - np.pi, 
                    self.parallel_pair2.midline.theta - np.pi
                   ],
                   key=np.abs
        )

def compute_houghline_intersection(line1: HoughLine, line2: HoughLine) -> tuple[float, float]:
    x = (line2.rho * np.sin(line1.theta) - line1.rho * np.sin(line2.theta)) / np.sin(line1.theta - line2.theta)
    y = (line1.rho * np.cos(line2.theta) - line2.rho * np.cos(line1.theta)) / np.sin(line1.theta - line2.theta)
    return x, y


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
                       approximate_location_threshold_distance = 50.0,
                       approximate_length: float = None,
                      ) -> list[FiducialCross]:
        """ Detect fiducial crosses in the image.

        Parameters
        ----------
        image : np.ndarray
        approximate_locations : list[tuple[int, int]], optional
            A list of approximate locations of crosses in the image, in the form 
            of (x, y) tuples. The algorithm is able to detect crosses without 
            initial locations, and it will return as many as it confidently 
            detects, but giving approximate locations can help prevent false 
            positives.
        approximate_location_threshold_distance : float, optional
            Detected crosses too farther from any approximate location than this
            parameter are discarded. Default 50.0
        approximate_length : float, optional
            The approximate length of the crosses. This can help the algorithm 
            detect crosses more accurately.
        
        Returns
        -------
        list[FiducialCross]
            A list of detected fiducial crosses.
        """

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

        def houghline_close_to_approximate_locations(line: HoughLine) -> bool:
            """ Check if HoughLine is close to any of the given approximate locations
            """
            def houghline_close_to_approximate_location(approximate_location: tuple[float, float]) -> bool:
                """ Check if HoughLine is close to the given approximate location

                That is, calculate the distance from the approximate location to 
                its projection onto the line. 

                We can make use of the line's parametrization, theta, which gives
                the line that is perpendicular to the Hough line and passes through
                the origin. The projection of the approximate location onto this 
                line, (x, y) . (cos(theta), sin(theta)), gives a length that is 
                equal to rho plus the perpendicular distance from approximate 
                location to the Hough line.

                Thus the distance we're after is |(x, y) . (cos(theta), sin(theta)) - rho|

                Parameters
                ----------
                approximate_location : tuple[float, float]
                    a tuple of x, y coordinates

                """
                x, y = approximate_location
                return np.abs(x * np.cos(line.theta) + y * np.sin(line.theta) - line.rho) <= approximate_location_threshold_distance

            return any(houghline_close_to_approximate_location(approximate_location) for approximate_location in approximate_locations)

        # filter out lines that are close to approximate locations
        if approximate_locations is not None:
            lines = [line for line in lines if houghline_close_to_approximate_locations(line)]

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
                             and line1 < line2  # prevent line1 matching itself, and prevent duplicate (line1, line2) pair
                         ]

        def in_image(x, y):
            return (0 <= x < image.shape[1]) and (0 <= y < image.shape[0])

        # find pairs of parallel line pairs that are perpendicular to each other
        # and intersect within the boundaries of the image
        perpendicular_quartets = [HoughLinesQuartet(parallel_pair1, parallel_pair2)
                                  for parallel_pair1, parallel_pair2 in product(parallel_pairs, parallel_pairs)
                                  if (0.0 <= np.abs((parallel_pair1.midline.theta - parallel_pair2.midline.theta) - 90*DEG) < self.perpendicular_hough_lines_max_angle)
                                     and in_image(*compute_houghline_intersection(parallel_pair1.midline, parallel_pair2.midline))
                                 ]

        # consolidate perpendicular quartets at similar locations and angles
        perpendicular_quartets = self._deduplicate_perpendicular_quartets(perpendicular_quartets)

        # TODO: estimate thickness of crosses
        def cross_from_perpendicular_quartet(perpendicular_quartet: HoughLinesQuartet) -> FiducialCross:
            x, y = compute_houghline_intersection(perpendicular_quartet.parallel_pair1.midline, perpendicular_quartet.parallel_pair2.midline)
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
                return np.sqrt(np.trapz(np.square(y_ - mean(y_)), x) / (x[-1] - x[0]))
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

    def _deduplicate_perpendicular_quartets(self, perpendicular_quartets: list[HoughLinesQuartet]):
        """ Consolidate perpendicular quartets at similar location and angle 
        
        Generate graph with quartets as vertices and similarity as links, then 
        find the disjoint groups and produce one average quartet.

        """
        # Create a graph with quartets at vertices
        G = nx.Graph()
        G.add_nodes_from(perpendicular_quartets)

        # connect vertices representing similar quartets
        def quartets_are_similar(quartet1: HoughLinesQuartet, quartet2: HoughLinesQuartet) -> bool:
            x1, y1 = quartet1.center
            x2, y2 = quartet2.center

            return (
                (x1 - x2)**2 + (y1 - y2)**2 <= (1.1 * np.sqrt(2) * self.hough_rho_resolution)**2
                and np.abs(quartet1.angle - quartet2.angle) <= 2 * self.hough_theta_resolution
            )

        G.add_edges_from((quartet1, quartet2) for quartet1, quartet2 in product(perpendicular_quartets, perpendicular_quartets) 
                         if quartet1 < quartet2 and quartets_are_similar(quartet1, quartet2)
                        )

        # Compute average quartet for each group
        def compute_average_quartet(quartets: set[HoughLinesQuartet]) -> HoughLinesQuartet:

            def mean_angle(angles: np.ndarray) -> float:
                return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

            return min(quartets, key=lambda q: np.abs(q.angle - mean_angle([q.angle for q in quartets])))

        return [compute_average_quartet(connected_group)
                for connected_group in nx.connected_components(G)
               ] 

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
                                   approximate_location_threshold_distance: float = 50.0,
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
        approximate_location_threshold_distance : float, optional
            Detected crosses too farther from any approximate location than this
            parameter are discarded. Default 50.0
        approximate_length : float, optional
            The approximate length of the crosses. This can help the algorithm 
            detect crosses more accurately.
        
        Returns
        -------
        np.ndarray
            The inpainted image.
        """

        crosses = self.detect_crosses(image, 
                                      approximate_locations=approximate_locations, 
                                      approximate_location_threshold_distance=approximate_location_threshold_distance, 
                                      approximate_length=approximate_length
                                     )
        return self.inpaint_crosses(image, crosses)
