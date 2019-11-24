import numpy as np

import qiskit as q
import qiskit.tools.jupyter
from qiskit.pulse import pulse_lib as _  # only want to call pulse_lib as q.pulse.pulse_lib


def get_line_maxcut_qaoa_circuit(N, beta=np.pi/3, gamma=np.pi/3):
    circ = q.QuantumCircuit(N)
    for i in range(N):
        circ.h(i)

    for i in range(N - 1):
        circ.zz_interaction(gamma, i, i + 1)
    
    for i in range(N):
        circ.rx(beta, i)

    return circ


def get_CH4_trotter_simulation_circuit(num_trotter_steps=1, superposition_start=True):
    """This circuit only applies phasing, so has no effect on |00> state.
    Use |++> instead, unless performing process tomography and phase effects can be measured.
    
    CH4_sto-3g_BK_grnd_AS1.txt
    Hamiltonian is -25.749885689311967 I0 +  1.8095454182113748 Z0 +
    1.8095454182113748 Z0 Z1 + 0.8752795188418716Z1
    """
    N = 2
    circ = q.QuantumCircuit(N)
    
    if superposition_start:
        for i in range(N):
            circ.h(i)
    
    for _ in range(num_trotter_steps):
        circ.rz(1.8095454182113748 / num_trotter_steps, 1)

        circ.zz_interaction(1.8095454182113748 / num_trotter_steps, 0, 1)

        circ.rz(0.8752795188418716 / num_trotter_steps, 1)

    return circ


def get_H2O_trotter_simulation_circuit(num_trotter_steps=1, superposition_start=True):
    """This circuit only applies phasing, so has no effect on |00> state.
    Use |++> instead, unless performing process tomography and phase effects can be measured.

    H2O_6-31g_BK_104_AS1.txt
    Hamiltonian is -93.72260987098602 I0 + 8.649346371694238 Z0
    8.649346371694238 Z0 Z1 + 1.1849141858184884 Z1
    """
    N = 2
    circ = q.QuantumCircuit(N)
    
    if superposition_start:
        for i in range(N):
            circ.h(i)

    
    for _ in range(num_trotter_steps):
        circ.rz(8.649346371694238 / num_trotter_steps, 1)

        circ.zz_interaction(8.649346371694238 / num_trotter_steps, 0, 1)

        circ.rz(1.1849141858184884 / num_trotter_steps, 1)
    
    return circ


def get_H2_trotter_simulation_circuit(num_trotter_steps=1, superposition_start=True):
    """This circuit only applies phasing, so has no effect on |00> state.
    Use |++> instead, unless performing process tomography and phase effects can be measured.

    H2_6-31g_JW_0.7_AS1.txt
    Hamiltonian is 12.990317523564187 I0 + -1.3462263309526756 Z0
    0.1663920290097622 Z0 Z1 + -1.3462263309526756 Z1
    """
    N = 2
    circ = q.QuantumCircuit(N)
    
    if superposition_start:
        for i in range(N):
            circ.h(i)
    
    for _ in range(num_trotter_steps):
        circ.rz(-1.3462263309526756 / num_trotter_steps, 1)
        circ.zz_interaction(0.1663920290097622 / num_trotter_steps, 0, 1)
        circ.rz(-1.3462263309526756 / num_trotter_steps, 1)
    
    return circ


def get_LiH_trotter_simulation_circuit(num_trotter_steps=1, superposition_start=True):
    """This circuit only applies phasing, so has no effect on |00> state.
    Use |++> instead, unless performing process tomography and phase effects can be measured.

    LiH_sto-3g_BK_1.45_AS1.txt
    Hamiltonian is 2.839189402302054 I0 - 0.767730045854766  Z0
    -0.767730045854766 Z0 Z1 + 0.4144660434569498 Z1
    """
    N = 2
    circ = q.QuantumCircuit(N)
    
    if superposition_start:
        for i in range(N):
            circ.h(i)
    
    for _ in range(num_trotter_steps):
        circ.rz(-0.767730045854766 / num_trotter_steps, 1)
        circ.zz_interaction(-0.767730045854766 / num_trotter_steps, 0, 1)
        circ.rz(0.4144660434569498 / num_trotter_steps, 1)
    
    return circ


def get_LiH_VQE(alpha, beta):
    """LiH VQE UCCSD circuit from https://arxiv.org/pdf/1803.10238.pdf.
    Per eq. 8, and relabeling qubits {2, 4, 6} to {1, 0, 2}, we have
    U_UCCSD = exp(-i*alpha* Y_0 X_1) * exp(-i*beta* X_1 Y_2)
    applied to |111>.
    """
    N = 3
    circ = q.QuantumCircuit(N)
    circ.x(0); circ.x(1); circ.x(2);  # start with |111>

    #  exp(-i*beta* X_1 Y_2)
    circ.h(1)
    circ.sdg(2); circ.h(2)
    circ.cx(1, 2); circ.rz(beta, 2);  circ.cx(1, 2);
    circ.h(2); circ.s(2)

    # exp(-i*alpha* Y_0 X_1)
    circ.sdg(0); circ.h(0);
    circ.cx(0, 1); circ.rz(alpha, 1);  circ.cx(0, 1);
    circ.h(0); circ.s(0);
    circ.h(1)

    return circ


def get_H2_VQE(theta):
    """H2 VQE UCCSD from https://arxiv.org/pdf/1512.06860.pdf Fig 1.

    I am reversing the indices from the figure."""
    N = 2
    circ = q.QuantumCircuit(N)
    circ.ry(np.pi / 2, 0)
    circ.rx(np.pi / 2, 1)
    circ.cx(0, 1); circ.rz(theta, 1); circ.cx(0, 1)
    circ.ry(-np.pi / 2, 0)
    circ.rx(np.pi / 2, 1)
    return circ


def main():
    print(get_line_maxcut_qaoa_circuit(4))
    print(get_H2O_trotter_simulation_circuit())
    print(get_H2_trotter_simulation_circuit())
    print(get_LiH_trotter_simulation_circuit())
    print(get_CH4_trotter_simulation_circuit())
    print(get_LiH_VQE(0.3, 0.4))
    print(get_H2_VQE(0.3))


if __name__ == "__main__":
    main()
