import torch
from torchdeq import get_deq

import numpy as np
import matplotlib.pyplot as plt

deq = get_deq(f_solver='broyden', f_max_iter=20, f_tol=1e-6)

# The third equilibrium function
h = lambda z: torch.exp(-z)
z0 = torch.tensor(0.5)
z_out, info = deq(h, z0)

h_abs_trace = info['abs_trace']
h_abs_trace = h_abs_trace.mean(dim=0)[1:]


print(z_out, h_abs_trace)

def bind_newton_func(a):
    # newton fixed-point function
    return lambda x: 0.5 * (x + (a / x))

f_2 = bind_newton_func(4.0)

(z, info) = deq(f_2, z0)
z_abs_trace = info['abs_trace']



print(z, z_abs_trace)

