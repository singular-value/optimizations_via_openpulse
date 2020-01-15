import numpy as np
import qiskit as q


def update_basis_gates_and_circ_inst_map(decomposed_circuit, backend, circ_inst_map):
    """While Parametrized Schedules are not supported, simply update basis gates and circ_inst_map."""
    basis_gates = backend.configuration().basis_gates
    for instruction, qargs, cargs in decomposed_circuit.data:
        if instruction.name.startswith('direct_rx'):
            if instruction.name not in basis_gates:
                basis_gates.append(instruction.name)
            
            if not circ_inst_map.has(instruction.name, qubits=[qargs[0].index]):
                theta = float(instruction.name[len('direct_rx_'):])
                schedule = get_direct_rx_schedule(theta, qargs[0].index, circ_inst_map, backend)
                circ_inst_map.add(instruction.name, qubits=[qargs[0].index], schedule=schedule)

        elif instruction.name.startswith('cr'):
            if instruction.name not in basis_gates:
                basis_gates.append(instruction.name)

            if not circ_inst_map.has(instruction.name, qubits=[qargs[0].index, qargs[1].index]):
                theta = float(instruction.name[len('cr_'):])
                schedule = get_cr_schedule(theta, qargs[0].index, qargs[1].index, circ_inst_map, backend)
                circ_inst_map.add(instruction.name,
                            qubits=[qargs[0].index, qargs[1].index], schedule=schedule)

        elif instruction.name == 'open_cx':
            if instruction.name not in basis_gates:
                basis_gates.append(instruction.name)


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


def get_direct_rx_schedule(theta, qubit_index, circ_inst_map, backend):
    x_instructions = circ_inst_map.get('x', qubits=[qubit_index]).instructions
    assert len(x_instructions) == 1
    x_samples = x_instructions[0][1].command.samples
    area_under_curve = sum(map(np.real, x_samples))
    
    if theta > np.pi:
        theta -= 2 * np.pi
    
    direct_rx_samples = rescale_samples(x_samples, (theta / np.pi))
    direct_rx_samplepulse = q.pulse.SamplePulse(direct_rx_samples)
    direct_rx_command = direct_rx_samplepulse(backend.configuration().drive(qubit_index))
    return q.pulse.Schedule([0, direct_rx_command])

    return direct_rx_command


def get_cr_schedule(theta, control, target, circ_inst_map, backend):

    """Returns schedule for a cross-resonance pulse between control and target.
    Does a RX(-theta) on target if control is |0> and a RX(theta) on target if
    control is |1>.
    Crashes if the backend does not support CR between control and target
    (either because no connectivity, or because the CR is between target and control)
    """
    cx_instructions = circ_inst_map.get('cx', qubits=[control, target]).instructions
    xc_instructions = circ_inst_map.get('cx', qubits=[target, control]).instructions
    assert len(cx_instructions) < len(xc_instructions), 'CR pulse is on flipped indices'
    
    cr_control_inst = [inst for (_, inst) in cx_instructions if 'CR90p' in inst.name and inst.channels[0].name.startswith('u')]
    cr_drive_inst = [inst for (_, inst) in cx_instructions if 'CR90p' in inst.name and inst.channels[0].name.startswith('d')]
    
    assert len(cr_drive_inst) == 1 and len(cr_control_inst) == 1
    cr_control_inst = cr_control_inst[0]  # driving of control qubit at target's frequency
    cr_drive_inst = cr_drive_inst[0]  # active cancellation tone

    flip = False
    if theta < 0:
        flip = True
        theta = -1 * theta
    
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

    current_area_under_curve = sum(map(np.real, cr_control_samples))
    scaling_factor = target_area_under_curve / current_area_under_curve

    cr_drive_samples *= scaling_factor
    cr_control_samples *= scaling_factor
    
    cr_p_schedule = q.pulse.SamplePulse(cr_drive_samples)(cr_drive_inst.channels[0]) | q.pulse.SamplePulse(
        cr_control_samples)(cr_control_inst.channels[0])
    cr_m_schedule = q.pulse.SamplePulse(-1*cr_drive_samples)(cr_drive_inst.channels[0]) | q.pulse.SamplePulse(
        -1*cr_control_samples)(cr_control_inst.channels[0])

    if flip:
        schedule = cr_m_schedule
        schedule |= circ_inst_map.get('x', qubits=[control]) << schedule.duration
        schedule |= cr_p_schedule << schedule.duration
    else:
        schedule = cr_p_schedule
        schedule |= circ_inst_map.get('x', qubits=[control]) << schedule.duration
        schedule |= cr_m_schedule << schedule.duration

    return schedule


def kl_divergence(ideal_counts, actual_counts):
    """Return KL divergence between two frequency dictionaries."""
    ideal_total = sum(ideal_counts.values())
    actual_total = sum(actual_counts.values())
    kl_div = 0
    for k, v in ideal_counts.items():
        p = v / ideal_total
        q = actual_counts.get(k, 0) / actual_total
        if q != 0:
            kl_div += p * np.log(p / q)
    return kl_div


def cross_entropy(ideal_counts, actual_counts):
    """Return cross entropy between two frequency dictionaries."""
    ideal_total = sum(ideal_counts.values())
    actual_total = sum(actual_counts.values())
    cross_entropy = 0
    for k, v in ideal_counts.items():
        p = v / ideal_total
        q = actual_counts.get(k, 0) / actual_total
        if q != 0:
            cross_entropy += -p * np.log(q)
    return cross_entropy
