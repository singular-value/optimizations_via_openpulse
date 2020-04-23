Note that most of our experiments were run in November 2019. The pulse library for qiskit (as well as the backend support) has changed since then.
However, it should be not be too difficult to migrate to the new APIs, for example: `cmd_def` -> `InstructionScheduleMap`.
The most recent batch of experiments on Armonk were run in January 2020, when the pulse API was closer to the current form.


Our compiler structural modifications are in qiskit-terra/ while most of the "business logic" is in pulse_compiler_helper_fns.py.

Our custom simulator is in qiskit-aer/

The Experiments/ directory has iPython notebooks demonstrating our results; Some representative notebooks:
- ZZ_Interaction_EXPERIMENT.ipynb has our ZZ-interaction fidelity measurements (Fig 10)
- Almaden Qutrit... has our results for qutrit control (Fig 11)
- Randomized_Benchmarking_Armonk.ipynb has our RB-style experiments (Fig 13)
- various tomography notebooks are included

Also, the Figures/ subdirectory has notebooks used to generate Figures 2 and 4

