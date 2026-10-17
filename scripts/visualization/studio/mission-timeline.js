const sameTime = (a, b) => Math.abs(a - b) <= 1e-8;
const vector = (value, n) => Array.isArray(value) && value.length === n && value.every(Number.isFinite);

/** Validate actual SI contact poses and their source-bound display handoff. */
export function finiteContactRecord(lifecycle, approach = null) {
  const contact = lifecycle?.finite_contact;
  if (!contact) return null;
  const times = contact.time_s, ids = Object.keys(contact.chasers || {});
  if (contact.schema !== 'orbexa.finite-contact.studio.v1' || !times?.length ||
      times.some((t, i) => !Number.isFinite(t) || i && t <= times[i - 1]) ||
      contact.approach_canonical_sha256 !== lifecycle.approach_canonical_sha256 ||
      contact.lifecycle_canonical_sha256 !== lifecycle.lifecycle_canonical_sha256 ||
      !/^[a-f0-9]{64}$/.test(contact.lifecycle_canonical_sha256 || '') ||
      !/^[a-f0-9]{64}$/.test(contact.source_history_sha256 || '') || !ids.length ||
      JSON.stringify(Object.keys(contact.pads || {}).sort()) !== JSON.stringify([...ids].sort()) ||
      contact.display_rotation_eci_to_capture?.length !== 3 || !contact.display_rotation_eci_to_capture.every(row => vector(row, 3))) {
    throw new Error('Finite-contact replay has malformed clock, fleet, frame or lifecycle lineage.');
  }
  const frame = contact.display_rotation_eci_to_capture;
  const determinant = frame[0][0] * (frame[1][1] * frame[2][2] - frame[1][2] * frame[2][1]) -
    frame[0][1] * (frame[1][0] * frame[2][2] - frame[1][2] * frame[2][0]) +
    frame[0][2] * (frame[1][0] * frame[2][1] - frame[1][1] * frame[2][0]);
  if (Math.abs(determinant - 1) > 1e-8 || frame.some((a, i) => frame.some((b, j) => Math.abs(a.reduce((sum, value, k) => sum + value * b[k], 0) - Number(i === j)) > 1e-8))) throw new Error('Capture display axes are not a proper rigid coordinate rotation.');
  for (const body of [contact.target, ...ids.map(id => contact.chasers[id])]) {
    for (const [field, width] of [['position_eci_m', 3], ['velocity_eci_m_s', 3], ['quaternion_body_to_eci', 4], ['angular_velocity_body_rad_s', 3]]) {
      if (body?.[field]?.length !== times.length || !body[field].every(row => vector(row, width))) throw new Error('Finite contact lacks aligned recorded SI body poses.');
    }
    if (body.quaternion_body_to_eci.some(q => Math.abs(Math.hypot(...q) - 1) > 1e-8)) throw new Error('Finite-contact body quaternion is not unit length.');
    if (body.fuel_kg?.length !== times.length || body.fuel_kg.some(value => !Number.isFinite(value) || value < 0)) throw new Error('Finite contact lacks aligned physical fuel samples.');
  }
  for (const pad of Object.values(contact.pads)) {
    for (const field of ['tip_position_eci_m', 'extension_body_m', 'force_eci_n', 'moment_eci_nm']) {
      if (pad[field]?.length !== times.length || !pad[field].every(row => vector(row, 3))) throw new Error('Finite contact lacks aligned recorded pad tips.');
    }
    for (const field of ['gap_m', 'activation']) if (pad[field]?.length !== times.length || !pad[field].every(Number.isFinite)) throw new Error('Finite contact lacks aligned scalar pad records.');
    if (pad.activation.some(value => value < 0 || value > 1)) throw new Error('Finite-contact activation is outside its declared interval.');
    if (!vector(pad.mount_body_m, 3) || !vector(pad.latched_tip_body_m, 3)) throw new Error('Finite contact lacks recorded mount/latch tip offsets.');
  }
  if (!lifecycle.time_s?.length || !sameTime(times.at(-1), lifecycle.time_s[0])) throw new Error('Finite contact does not end at the actual rigid assembly epoch.');
  if (approach) {
    const binding = contact.approach_binding, expected = approach.tracks.map(track => track.id).sort();
    if (Boolean(lifecycle.synthetic) !== Boolean(approach.synthetic) || !binding ||
        binding.final_anomaly !== approach.q.at(-1) || !sameTime(binding.physical_time_s, approach.elapsed_s?.at(-1)) ||
        !sameTime(times[0], binding.physical_time_s) || JSON.stringify([...ids].sort()) !== JSON.stringify(expected)) {
      throw new Error('Finite-contact source is not the selected actual approach or qualification lineage.');
    }
    for (const track of approach.tracks) {
      const point = binding.positions_reference_m?.[track.id], actual = track.states.at(-1)?.slice(0, 3);
      if (!vector(point, 3) || !vector(actual, 3) || Math.hypot(...point.map((v, i) => v - actual[i])) > 5e-7) throw new Error('Finite-contact initial position differs from the selected approach.');
    }
    const rotation = approach.rotations?.at(-1), recorded = binding.target_rotation_reference;
    if (!rotation || !recorded || Math.hypot(...rotation.flat().map((v, i) => v - recorded.flat()[i])) > 1e-8) throw new Error('Finite-contact display axes differ from the selected approach rotation.');
  }
  return contact;
}

function appendPhysical(samples, lifecycle, contact) {
  const append = sample => {
    if (samples.length && sameTime(samples.at(-1).time, sample.time)) samples.pop();
    if (samples.length && sample.time <= samples.at(-1).time) throw new Error('Physical replay samples overlap or reverse their recorded clock.');
    samples.push(sample);
  };
  if (contact) contact.time_s.forEach((time, index) => append({ kind: 'contact', index, time, phase: 'finite_contact' }));
  lifecycle.time_s.forEach((time, index) => append({ kind: 'lifecycle', index, time, phase: lifecycle.phases[index] || 'attached_flight' }));
}

/** One clock over approach, actual compliant contact and attached flight. */
export function missionTimeline(approach, lifecycle = null) {
  const count = approach.q.length;
  const clock = approach.elapsed_s;
  const validClock = Array.isArray(clock) && clock.length === count &&
    clock.every((value, index) => Number.isFinite(value) && (!index || value > clock[index - 1]));
  const times = validClock ? clock : approach.q.map(value =>
    30 * (value - approach.q[0]) / (approach.q.at(-1) - approach.q[0] || 1));
  const samples = times.map((time, index) => ({ kind: 'approach', index, time, phase: approach.phases[index] || 'approach' }));
  if (lifecycle) {
    const attachedTimes = lifecycle.time_s;
    const contact = finiteContactRecord(lifecycle, approach);
    if (!validClock || !Array.isArray(attachedTimes) || !attachedTimes.length ||
        attachedTimes.some((value, index) => !Number.isFinite(value) || (index && value <= attachedTimes[index - 1])) ||
        attachedTimes[0] < times.at(-1) - 1e-8) {
      throw new Error('The attached-flight record cannot be placed on the recorded approach clock.');
    }
    appendPhysical(samples, lifecycle, contact);
  }
  const phases = [];
  samples.forEach((sample, index) => {
    if (!index || sample.phase !== samples[index - 1].phase) phases.push({ phase: sample.phase, index, time: sample.time });
  });
  return { samples, phases, physical: validClock, approachCount: count, start: times[0], end: samples.at(-1).time };
}

export function lifecycleTimeline(lifecycle) {
  const samples = [], contact = finiteContactRecord(lifecycle);
  if (!Array.isArray(lifecycle.time_s) || !lifecycle.time_s.length ||
      lifecycle.time_s.some((time, i) => !Number.isFinite(time) || i && time <= lifecycle.time_s[i - 1])) throw new Error('Physical lifecycle clock is malformed.');
  appendPhysical(samples, lifecycle, contact);
  return samples;
}

export function sampleAtTime(samples, time) {
  let left = 0, right = samples.length;
  while (left < right) {
    const middle = Math.floor((left + right) / 2);
    if (samples[middle].time <= time) left = middle + 1;
    else right = middle;
  }
  return Math.max(0, left - 1);
}

export function phaseTitle(phase, hasRelease = false) {
  return ({ carrier_departure: 'Payload release', transfer: 'Transfer', deployment: hasRelease ? 'Proximity approach' : 'Deployment', staging: 'Staging',
    rendezvous: 'Approach', docking: 'Final approach', finite_contact: 'Finite contact', attachment: 'Attachment', detumbling: 'Detumble',
    coast_to_apoapsis: 'Coast', thrust_alignment: 'Align for burn', finite_burn: 'Disposal burn',
    coast_to_entry_interface: 'Entry interface' })[phase] || phase.replaceAll('_', ' ');
}

export function releaseContextVisible(data, index) {
  const phase = data.phases[index];
  return phase === 'carrier_departure' || phase === 'transfer' && Array.isArray(data.elapsed_s) &&
    Number.isFinite(data.elapsed_s[index]) && data.elapsed_s[index] - data.elapsed_s[0] <= 5;
}
