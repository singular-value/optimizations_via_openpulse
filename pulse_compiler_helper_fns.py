import numpy as np
import qiskit as q


def update_basis_gates_and_cmd_def(decomposed_circuit, backend, system, cmd_def):
    """While Parametrized Schedules are not supported, simply update basis gates and cmd_def."""
    basis_gates = backend.configuration().basis_gates
    for instruction, qargs, cargs in decomposed_circuit.data:
        if instruction.name.startswith('direct_rx'):
            if instruction.name not in basis_gates:
                basis_gates.append(instruction.name)
            
            if not cmd_def.has(instruction.name, qubits=[qargs[0].index]):
                theta = float(instruction.name[len('direct_rx_'):])
                schedule = get_direct_rx_schedule(theta, qargs[0].index, cmd_def, system)
                cmd_def.add(instruction.name, qubits=[qargs[0].index], schedule=schedule)


def rescale_samples(samples, scale_factor, method='rescale_height'):
    assert scale_factor <= 1, 'only tested for scaling down pulses'

    if method == 'rescale_height':
        return _rescale_height(samples, scale_factor)
    elif method == 'rescale_width':
        return _rescale_width(samples, scale_factor)
    elif method == 'rescale_height_and_width':
        return _rescale_height_and_width(samples, scale_factor)


def _rescale_height(samples, scale_factor):
    return samples * scale_factor


def _rescale_width(samples, scale_factor):
    assert False, 'still debugging implementation'


def _rescale_height_and_width(samples, scale_factor):
    assert False, 'still debugging implementation'

    print('original real area under curve is %s' % sum(map(np.real, samples)))

    rescaled_length = int(0.5 + len(samples) * np.sqrt(scale_factor))
    rescaled_samples = [0] * rescaled_length
    width_scale_factor = rescaled_length / len(samples)
    height_scale_factor = scale_factor / width_scale_factor
    samples = samples * scale_factor
    
    for i in range(len(samples)):
        # split samples[i...i+1] into rescaled_samples[i/scale_factor...(i+1)/scale_factor]
        if int(i * width_scale_factor) == int((i + 1) * width_scale_factor):
            rescaled_samples[int(i * width_scale_factor)] += samples[i]
        else:
            fraction = int(1 + i * width_scale_factor) - int(i * width_scale_factor)
            rescaled_samples[int(i * width_scale_factor)] += samples[i] * fraction
            rescaled_samples[int(i * width_scale_factor)] += samples[i] * (1 - fraction)
    print('final real area under curve is %s' % sum(map(np.real, rescaled_samples)))
    return rescaled_samples


def get_direct_rx_schedule(theta, qubit_index, cmd_def, system):
    x_instructions = cmd_def.get('x', qubits=[qubit_index]).instructions
    assert len(x_instructions) == 1
    x_samples = x_instructions[0][1].command.samples
    area_under_curve = sum(map(np.real, x_samples))
    
    if theta > np.pi:
        theta -= 2 * np.pi
    
    direct_rx_samples = rescale_samples(x_samples, (theta / np.pi))
    direct_rx_samplepulse = q.pulse.SamplePulse(direct_rx_samples)
    direct_rx_command = direct_rx_samplepulse(system.drives[qubit_index])

    return direct_rx_command


def get_cr_schedule(theta, control, target, cmd_def, system):
    """Returns schedule for a cross-resonance pulse between control and target.
    Does a RX(-theta) on target if control is |0> and a RX(theta) on target if
    control is |1>.
    Crashes if the backend does not support CR between control and target
    (either because no connectivity, or because the CR is between target and control)
    """
    cx_instructions = cmd_def.get('cx', qubits=[control, target]).instructions
    xc_instructions = cmd_def.get('cx', qubits=[target, control]).instructions
    assert len(cx_instructions) < len(xc_instructions), 'CR pulse is on flipped indices'
    
    cr_control_inst = [inst for (_, inst) in cx_instructions if 'CR90p' in inst.name and inst.channels[0].name.startswith('u')]
    cr_drive_inst = [inst for (_, inst) in cx_instructions if 'CR90p' in inst.name and inst.channels[0].name.startswith('d')]
    
    assert len(cr_drive_inst) == 1 and len(cr_control_inst) == 1
    cr_control_inst = cr_control_inst[0]  # driving of control qubit at target's frequency
    cr_drive_inst = cr_drive_inst[0]  # active cancellation tone
    
    if theta > 2 * np.pi:
        theta -= 2 * np.pi

    full_area_under_curve = sum(map(np.real, cr_control_inst.command.samples))
    target_area_under_curve = full_area_under_curve * (theta / (np.pi / 2))

    # CR pulse samples have gaussian rise, flattop, and then gaussian fall.
    # we want to find the start and end indices of the flattop
    flat_start = 0
    while cr_drive_inst.command.samples[flat_start] != cr_drive_inst.command.samples[flat_start + 1]:
        flat_start += 1
    assert cr_control_inst.command.samples[flat_start] == cr_control_inst.command.samples[flat_start + 1]

    flat_end = flat_start + 1
    while cr_drive_inst.command.samples[flat_end] == cr_drive_inst.command.samples[flat_end + 1]:
        flat_end += 1
    assert cr_control_inst.command.samples[flat_end] == cr_control_inst.command.samples[flat_end - 1]

    area_under_curve = sum(map(np.real, cr_control_inst.command.samples[:flat_start]))
    area_under_curve += sum(map(np.real, cr_control_inst.command.samples[flat_end+1:]))
    flat_duration = (target_area_under_curve - area_under_curve) / np.real(cr_control_inst.command.samples[flat_start])
    flat_duration = max(0, int(flat_duration + 0.5))
    duration = len(cr_drive_inst.command.samples[:flat_start]) + flat_duration + len(cr_drive_inst.command.samples[flat_end+1:])
    if duration % 16 <= 8 and flat_duration > 8:
        flat_duration -= duration % 16
    else:
        flat_duration += 16 - (duration % 16)

    cr_drive_samples = np.concatenate([
        cr_drive_inst.command.samples[:flat_start],
        [cr_drive_inst.command.samples[flat_start]] * flat_duration,
        cr_drive_inst.command.samples[flat_end+1:]
    ])

    cr_control_samples = np.concatenate([
        cr_control_inst.command.samples[:flat_start],
        [cr_control_inst.command.samples[flat_start]] * flat_duration,
        cr_control_inst.command.samples[flat_end+1:]
    ])

    assert len(cr_drive_samples) % 16 == 0
    assert len(cr_control_samples) % 16 == 0
    
    cr_p_schedule = q.pulse.SamplePulse(cr_drive_samples)(cr_drive_inst.channels[0]) | q.pulse.SamplePulse(
        cr_control_samples)(cr_control_inst.channels[0])
    cr_m_schedule = q.pulse.SamplePulse(-1*cr_drive_samples)(cr_drive_inst.channels[0]) | q.pulse.SamplePulse(
        -1*cr_control_samples)(cr_control_inst.channels[0])
    
    schedule = cr_p_schedule
    schedule |= cmd_def.get('x', qubits=[control]) << schedule.duration
    schedule |= cr_m_schedule << schedule.duration

    return schedule
