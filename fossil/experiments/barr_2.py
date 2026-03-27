# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable


from dataclasses import replace
from lbp_neural_cbf.cbf.network import BarrierNN
import torch
from fossil import domains, learner, verifier, certificate, translator
from fossil import main, control
from fossil.consts import *

class Barr2(control.DynamicalModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [torch.exp(-x) + y - 1, -(torch.sin(x) ** 2)]

    def f_smt(self, v):
        sin = self.fncs["sin"]
        exp = self.fncs["exp"]
        x, y = v
        return [exp(-x) + y - 1, -(sin(x) ** 2)]


def test_lnn(args):
    batch_size = 500

    system = Barr2

    XD = domains.Rectangle([-2, -2], [2, 2])
    XI = domains.Sphere([-0.5, 0.5], 0.4)
    XU = domains.Sphere([0.7, -0.7], 0.3)

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    data = {
        certificate.XD: XD._generate_data(batch_size),
        certificate.XI: XI._generate_data(batch_size),
        certificate.XU: XU._generate_data(batch_size),
    }

    ###############################
    #
    ###############################

    activations = [ActivationType.TANH, ActivationType.TANH]
    hidden_neurons = [64, 64]
    opts = CegisConfig(
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
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
    model_path = "data/barr2_cbf.pth"
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
    verifier_instance.get_timer().reset()
    print(verifier_instance.verify(V, Vdot))
    print("Verifier times: {}".format(verifier_instance.get_timer()))


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
