"""
Cheetah-based Bayesian optimization generator.

This module provides a specialized Bayesian optimization generator that uses
the Cheetah particle accelerator simulation framework as a physics-informed
prior. The generator is designed for optimizing electron beam transport
parameters using realistic beam dynamics simulations.

The Cheetah prior provides physics-based guidance to the optimization process,
potentially improving convergence and reducing the number of experimental
evaluations needed to find optimal parameters.

Functions
---------
get_cheetah_generator(vocs)
    Create Cheetah-based Bayesian optimization generator.

Classes
-------
CheetahPrior
    PyTorch module implementing physics-informed prior using Cheetah simulations.

Examples
--------
Creating a Cheetah-based generator:

>>> from xopt import VOCS
>>> vocs = VOCS(
...     variables={"EMQ1H": [-10, 10], "EMQ2V": [-10, 10], "EMQ3H": [-10, 10]},
...     objectives={"f": "MINIMIZE"}
... )
>>> generator = get_cheetah_generator(vocs)

Notes
-----
This generator requires:
- Cheetah particle accelerator simulation package
- PyTorch for neural network components
- Access to HTU lattice configuration files
- Proper system path configuration for simulation scripts

The generator uses Expected Improvement acquisition function with a Cheetah-based
mean function that simulates beam transport through the accelerator lattice.
"""

import sys

# optimization/generators/generator_cheetah.py
from scipy.constants import m_e, e, c
import torch


def get_cheetah_generator(vocs):
    """
    Create Cheetah-based Bayesian optimization generator.

    Constructs a Bayesian optimization generator that uses the Cheetah particle
    accelerator simulation framework to provide physics-informed priors. This
    generator is specifically designed for optimizing electron beam transport
    parameters in the HTU (High-repetition-rate Terawatt Undulator) beamline.

    Parameters
    ----------
    vocs : VOCS
        Variables, Objectives, and Constraints Specification defining the
        optimization problem. Should include quadrupole magnet parameters
        (EMQ1H, EMQ2V, EMQ3H) as variables.

    Returns
    -------
    ExpectedImprovementGenerator
        Bayesian optimization generator with Cheetah-based physics prior.

    Raises
    ------
    ImportError
        If Cheetah package or HTU lattice modules cannot be imported.

    Examples
    --------
    >>> from xopt import VOCS
    >>> vocs = VOCS(
    ...     variables={
    ...         "EMQ1H": [-10, 10],
    ...         "EMQ2V": [-10, 10],
    ...         "EMQ3H": [-10, 10]
    ...     },
    ...     objectives={"f": "MINIMIZE"}
    ... )
    >>> generator = get_cheetah_generator(vocs)

    Notes
    -----
    This function requires:
    - Cheetah accelerator simulation package
    - HTU lattice configuration and utility functions
    - PyTorch for neural network components

    The generator uses Expected Improvement acquisition with a CheetahPrior
    mean function that simulates beam transport through the accelerator
    lattice to provide physics-based guidance.

    The hardcoded path suggests this is configured for a specific system
    and may need adjustment for different installations.
    """
    try:
        from cheetah import ParticleBeam, Segment

        sys.path.append(
            r"C:\Users\loasis.LOASIS\Documents\GitHub\2025_pilot_digital_twin_LDRD\simulation_scripts"
        )
        from htu_lattice import get_lattice, current_to_k

    except ImportError as e:
        raise ImportError(
            "The 'cheetah' dependency is required for 'bayes_cheetah'. "
            "Please install it separately."
        ) from e

    class CheetahPrior(torch.nn.Module):
        """
        Physics-informed prior using Cheetah beam dynamics simulations.

        This PyTorch module implements a mean function for Gaussian Process
        regression that uses Cheetah particle accelerator simulations to
        provide physics-based predictions of beam size at the diagnostic
        screen based on quadrupole magnet settings.

        The prior simulates electron beam transport through the HTU beamline
        and computes the expected beam size at the UC_ALineEBeam3 diagnostic
        location, providing informed guidance for the optimization process.

        Methods
        -------
        forward(X)
            Compute beam size prediction from quadrupole parameters.

        Notes
        -----
        The simulation uses:
        - 100 MeV beam energy
        - Twiss parameters: beta=2mm, alpha=0, emittance=1.5μm
        - 5000 simulation particles
        - Fixed random seed for reproducibility

        The objective function is the sum of squares of x and y beam sizes
        in millimeters, matching the experimental objective function.
        """

        def forward(self, X):
            """
            Compute beam size prediction from quadrupole magnet parameters.

            Simulates electron beam transport through the HTU beamline using
            Cheetah and computes the expected beam size at the diagnostic
            screen based on the input quadrupole magnet settings.

            Parameters
            ----------
            X : torch.Tensor
                Input tensor with shape (..., 3) containing quadrupole magnet
                current values [EMQ1H, EMQ2V, EMQ3H] in Amperes.

            Returns
            -------
            torch.Tensor
                Predicted beam size metric: (sigma_x * 1e3)² + (sigma_y * 1e3)²
                where sigma values are in meters and the result is in mm².

            Notes
            -----
            The simulation process:
            1. Converts currents to quadrupole strengths using current_to_k
            2. Sets up HTU lattice segment to UC_ALineEBeam3 screen
            3. Creates initial beam with specified Twiss parameters
            4. Tracks beam through the lattice
            5. Extracts final beam sizes and computes objective

            The quadrupole types and beam energy are hardcoded based on
            the HTU beamline configuration.
            """
            segment = Segment(get_lattice("cheetah", to_element="UC_ALineEBeam3"))
            segment.EMQ1H.k1 = current_to_k(
                X[..., 0].to(torch.float32), "EMQD-113-394", 100e6
            )
            segment.EMQ2V.k1 = current_to_k(
                X[..., 1].to(torch.float32), "EMQD-113-949", 100e6
            )
            segment.EMQ3H.k1 = current_to_k(
                X[..., 2].to(torch.float32), "EMQD-113-394", 100e6
            )
            torch.manual_seed(0)
            beam_energy = torch.tensor(100e6)
            gamma = beam_energy * e / (m_e * c**2)

            incoming = ParticleBeam.from_twiss(
                beta_x=torch.tensor(0.002),
                alpha_x=torch.tensor(0.0),
                emittance_x=torch.tensor(1.5e-6) / gamma,
                beta_y=torch.tensor(0.002),
                alpha_y=torch.tensor(0.0),
                emittance_y=torch.tensor(1.5e-6) / gamma,
                sigma_tau=torch.tensor(1e-6),
                sigma_p=torch.tensor(2.5e-2),
                energy=beam_energy,
                total_charge=torch.tensor(25.0e-12),
                num_particles=5000,
            )
            segment.track(incoming=incoming)
            beam = segment.UC_ALineEBeam3.get_read_beam()
            matching = (1e3 * beam.sigma_x) ** 2 + (1e3 * beam.sigma_y) ** 2
            return matching

    from xopt.generators.bayesian import ExpectedImprovementGenerator
    from xopt.generators.bayesian.models.standard import StandardModelConstructor

    return ExpectedImprovementGenerator(
        vocs=vocs,
        gp_constructor=StandardModelConstructor(mean_modules={"f": CheetahPrior()}),
    )
