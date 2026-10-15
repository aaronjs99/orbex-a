/** One clock over saved approach and attached-flight samples. No state interpolation. */
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
    if (!validClock || !Array.isArray(attachedTimes) || !attachedTimes.length ||
        attachedTimes.some((value, index) => !Number.isFinite(value) || (index && value <= attachedTimes[index - 1])) ||
        attachedTimes[0] < times.at(-1)) {
      throw new Error('The attached-flight record cannot be placed on the recorded approach clock.');
    }
    attachedTimes.forEach((time, index) => samples.push({ kind: 'lifecycle', index, time, phase: lifecycle.phases[index] || 'attached_flight' }));
  }
  const phases = [];
  samples.forEach((sample, index) => {
    if (!index || sample.phase !== samples[index - 1].phase) phases.push({ phase: sample.phase, index, time: sample.time });
  });
  return { samples, phases, physical: validClock, approachCount: count, start: times[0], end: samples.at(-1).time };
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
    rendezvous: 'Approach', docking: 'Final approach', attachment: 'Attachment', detumbling: 'Detumble',
    coast_to_apoapsis: 'Coast', thrust_alignment: 'Align for burn', finite_burn: 'Disposal burn',
    coast_to_entry_interface: 'Entry interface' })[phase] || phase.replaceAll('_', ' ');
}

export function releaseContextVisible(data, index) {
  const phase = data.phases[index];
  return phase === 'carrier_departure' || phase === 'transfer' && Array.isArray(data.elapsed_s) &&
    Number.isFinite(data.elapsed_s[index]) && data.elapsed_s[index] - data.elapsed_s[0] <= 5;
}
