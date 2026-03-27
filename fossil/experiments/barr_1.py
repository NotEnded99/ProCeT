# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
import timeit

from lbp_neural_cbf.cbf.network import BarrierNN
from fossil import control, domains, learner, translator, verifier
from fossil import certificate
from fossil import main
from fossil.consts import *


class Barr1(control.DynamicalModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [y + 2 * x * y, -x - y**2 + 2 * x**2]

    def f_smt(self, v):
        x, y = v
        return [y + 2 * x * y, -x - y**2 + 2 * x**2]


class UnsafeDomain(domains.Set):
    dimension = 2

    def generate_domain(self, v):
        x, y = v
        return x + y**2 <= 0

    def generate_data(self, batch_size):
        points = []
        limits = [[-2, -2], [0, 2]]
        while len(points) < batch_size:
            dom = domains.square_init_data(limits, batch_size)
            idx = torch.nonzero(dom[:, 0] + dom[:, 1] ** 2 <= 0)
            points += dom[idx][:, 0, :]
        return torch.stack(points[:batch_size])


def test_lnn(args):
    XD = domains.Rectangle([-2, -2], [2, 2])
    XI = domains.Rectangle([0, 1], [1, 2])
    XU = UnsafeDomain()

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    data = {
        certificate.XD: XD._generate_data(500),
        certificate.XI: XI._generate_data(500),
        certificate.XU: XU._generate_data(500),
    }

    system = Barr1
    activations = [ActivationType.TANH, ActivationType.TANH, ActivationType.TANH]
    hidden_neurons = [128, 256, 128]
    opts = CegisConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        VERBOSE=2,
    )

    # Prep domains
    x = verifier.get_verifier_type(opts.VERIFIER).new_vars(
        opts.N_VARS
    )
    _domains = {
        label: (
            domain.generate_boundary(x)
            if label in certificate.BORDERS
            else domain.generate_domain(x)
        )
        for label, domain in opts.DOMAINS.items()
    }
    
    # Create certificate
    custom_certificate = opts.CUSTOM_CERTIFICATE
    certificate_type = certificate.get_certificate(
        opts.CERTIFICATE, custom_certificate
    )
    _certificate = certificate_type(_domains, opts)

    # Prepare dynamics
    system = opts.SYSTEM
    ctrler = None
    f = None
    if opts.CTRLAYER:
        ctrl_activ = opts.CTRLACTIVATION
        ctrler = control.GeneralController(
            inputs=opts.N_VARS,
            output=opts.CTRLAYER[-1],
            layers=opts.CTRLAYER[:-1],
            activations=ctrl_activ,
        )
        f = system(ctrler)
    else:
        f = system()
    xdot = f(x)
    opts = replace(opts, SYSTEM=f)

    # Create learner
    learner_type = learner.get_learner(
        opts.TIME_DOMAIN, opts.CTRLAYER
    )
    learner_instance = learner_type(
        opts.N_VARS,
        _certificate.learn,
        *opts.N_HIDDEN_NEURONS,
        activation=opts.ACTIVATION,
        bias=_certificate.bias,
        config=opts,
    )

    # Load pretrained model
    model_path = "data/barr1_cbf.pth"
    device = torch.device("cpu")
    network = BarrierNN(opts.N_VARS, opts.N_HIDDEN_NEURONS)
    network.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

    k = 0
    for layer in network.network:
        if isinstance(layer, torch.nn.Linear):
            learner_instance.layers[k].weight.data.copy_(layer.weight.data)
            learner_instance.layers[k].bias.data.copy_(layer.bias.data)
            k += 1
        elif not isinstance(layer, torch.nn.Tanh):
            raise ValueError("Unexpected layer type in loaded network.")

    # Create translator
    translator_type = translator.get_translator_type(
        opts.TIME_DOMAIN, opts.VERIFIER
    )
    _translator = translator.get_translator(
        translator_type,
        x,
        xdot,
        opts.ROUNDING,
        config=opts,
    )

    V, Vdot = _translator.get_symbolic_formula(learner_instance, x, xdot, lf=opts.FACTORS)

    # Create verifier
    verifier_type = verifier.get_verifier_type(opts.VERIFIER)
    verifier_instance = verifier.get_verifier(
        verifier_type,
        opts.N_VARS,
        _certificate.get_constraints,
        x,
        opts,
    )
    verifier_instance._solver_timeout = 3600  # seconds

    # Verify
    print(verifier_instance.verify(V, Vdot))


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
