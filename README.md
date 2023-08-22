
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
<!--- [![Bindings](https://img.shields.io/github/workflow/status/cda-tum/ddsim/Deploy%20to%20PyPI?style=flat-square&logo=github&label=python)]()
 [![Documentation](https://img.shields.io/readthedocs/ddsim?logo=readthedocs&style=flat-square)]() 
 [![codecov](https://img.shields.io/codecov/c/github/cda-tum/)]() -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/qmap/main/docs/source/_static/mqt_light.png" width="60%">
    <img src="https://raw.githubusercontent.com/cda-tum/qmap/main/docs/source/_static/mqt_dark.png" width="60%">
  </picture>
</p>


# MQT Qudits - Compilation of Entangling Gates for High-Dimensional Quantum Systems

A tool for the compilation of arbitrary d-dimensional two-qudit entangling unitaries into error-efficient sequences of local operations and entangling primitives supported by the quantum architecture by [Chair for Design Automation](https://www.cda.cit.tum.de/).


If you have any questions, feel free to contact us via [quantum.cda@xcit.tum.de](mailto:iic-quantum@jku.at) or by creating an [issue](https://github.com/cda-tum/qudit-compilation/issues) on GitHub.

## Getting Started

The compiler demands only for the resolution of dependencies, to solve run in terminal.
```
pip install -r requirements.txt
```

The following code gives an example on the usage:

```python3
from src import global_vars
from src.decomposer.cex import Cex
from src.decomposer.entangled_qr import EntangledQR
from src.gates.entangling_gates import csum

global_vars.dimension_fix = 4 # dimension of the single qudit

# global_vars.CEX_SEQUENCE = [...] # It is also possible to input a customized sequence for substituting the CEX operation, as a list of numpy arrays

target = csum(global_vars.dimension_fix) # the target gate for which we want to find a decomposition

decomposer = EntangledQR(target)

sequence, num_crots, num_pswaps = decomposer.entangling_qr()

outcome = decomposer.basic_verify(target, sequence) # outcome must be true

```

In case you want to use the second compilation step described in the paper:
```python3
import numpy as np

from src.layered.compile_pkg.ansatz.instantiate import create_ms_instance
from src.layered.compile_pkg.opt.distance_measures import fidelity_on_unitares
from src.layered.compile_pkg.search import binary_search_compile
from src import global_vars
from src.decomposer.cex import Cex
from src.utils.qudit_circ_utils import gate_expand_to_circuit
from src.utils.rotation_utils import matmul

# First we set all the necessary global variables for the solver, the duration of an opt. cycle is inside the function "run"
global_vars.OBJ_FIDELITY = 1e-4
global_vars.SINGLE_DIM = 2
global_vars.TARGET_GATE = Cex().cex_101(global_vars.SINGLE_DIM)
global_vars.MAX_NUM_LAYERS = (2 * global_vars.SINGLE_DIM ** 2)

best_layer, best_error, best_xi = binary_search_compile(global_vars.SINGLE_DIM, global_vars.MAX_NUM_LAYERS, "MS")

# the function creates a list of numpy arrays as resulting sequence
decomposed_target = create_ms_instance(best_xi, global_vars.SINGLE_DIM)

# a simple verification
unitary = gate_expand_to_circuit(np.identity(global_vars.SINGLE_DIM, dtype=complex), n=2, target=0, dim=global_vars.SINGLE_DIM)

for rot in decomposed_target:
    unitary = matmul(unitary, rot)

print((1 - fidelity_on_unitares(unitary, global_vars.TARGET_GATE)) < 1e-4) # outcome should be true




```


## System Requirements and Building

The implementation is compatible with a minimimum version of Python 3.8.

Building (and running) is continuously tested under Linux, macOS, and Windows using the [latest available system versions for GitHub Actions](https://github.com/actions/virtual-environments).

## References

No References.

K. Mato, M. Ringbauer, S. Hillmich and R. Wille, "Compilation of Entangling Gates for High-Dimensional Quantum Systems," 2023 28th Asia and South Pacific Design Automation Conference (ASP-DAC), Tokyo, Japan, 2023, pp. 202-208.
