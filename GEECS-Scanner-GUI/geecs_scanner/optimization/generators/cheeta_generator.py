# optimization/generators/generator_cheetah.py
from scipy.constants import m_e, e, c
import torch

def get_cheetah_generator(vocs):
    try:
        from cheetah import ParameterBeam, ParticleBeam, Segment
        sys.path.append(r'C:\Users\loasis.LOASIS\Documents\GitHub\2025_pilot_digital_twin_LDRD\simulation_scripts')
        from htu_lattice import get_lattice, current_to_k

    except ImportError as e:
        raise ImportError("The 'cheetah' dependency is required for 'bayes_cheetah'. "
                          "Please install it separately.") from e

    class CheetahPrior(torch.nn.Module):
        def forward(self, X):
            segment = Segment(get_lattice("cheetah", to_element='UC_ALineEBeam3'))
            segment.EMQ1H.k1 = current_to_k(X[..., 0].to(torch.float32), "EMQD-113-394", 100e6)
            segment.EMQ2V.k1 = current_to_k(X[..., 1].to(torch.float32), "EMQD-113-949", 100e6)
            segment.EMQ3H.k1 = current_to_k(X[..., 2].to(torch.float32), "EMQD-113-394", 100e6)
            torch.manual_seed(0)
            beam_energy = torch.tensor(100e6)
            gamma = beam_energy * e / (m_e * c ** 2)

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
                num_particles=5000
            )
            segment.track(incoming=incoming)
            beam = segment.UC_ALineEBeam3.get_read_beam()
            matching = (1e3 * beam.sigma_x) ** 2 + (1e3 * beam.sigma_y) ** 2
            return matching

    from xopt.generators.bayesian import ExpectedImprovementGenerator
    from xopt.generators.bayesian.models.standard import StandardModelConstructor

    return ExpectedImprovementGenerator(
        vocs=vocs,
        gp_constructor=StandardModelConstructor(
            mean_modules={"f": CheetahPrior()}
        )
    )
