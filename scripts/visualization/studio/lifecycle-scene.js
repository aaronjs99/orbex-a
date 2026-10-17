import * as THREE from 'three';
import { representativeSpacecraft, compositeSpacecraft } from './spacecraft.js';
import { finiteContactRecord, lifecycleTimeline } from './mission-timeline.js';

const rotation = value => new THREE.Matrix4().set(value[0][0], value[0][1], value[0][2], 0,
  value[1][0], value[1][1], value[1][2], 0, value[2][0], value[2][1], value[2][2], 0, 0, 0, 0, 1);

/** Display coordinate change only; never interpolate or propagate a state. */
export function lifecyclePose(data, kind, index) {
  // The dataset is validated once when installed, not once per mesh/sample.
  const contact = data?.finite_contact;
  if (!contact) throw new Error('Shared physical replay requires actual finite-contact records.');
  const display = new THREE.Quaternion().setFromRotationMatrix(rotation(contact.display_rotation_eci_to_capture));
  const vector = value => new THREE.Vector3(...value);
  const turn = value => vector(value).applyQuaternion(display);
  const compose = value => display.clone().multiply(new THREE.Quaternion(...value));
  const bodies = {}, pads = {};
  let targetQuaternion, targetWorld;
  if (kind === 'contact') {
    if (!Number.isInteger(index) || index < 0 || index >= contact.time_s.length) throw new Error('Contact replay node is outside its saved trace.');
    targetWorld = vector(contact.target.position_eci_m[index]);
    targetQuaternion = compose(contact.target.quaternion_body_to_eci[index]);
    for (const [id, body] of Object.entries(contact.chasers)) {
      const world = vector(body.position_eci_m[index]), physicalQ = new THREE.Quaternion(...body.quaternion_body_to_eci[index]);
      bodies[id] = { position: world.clone().sub(targetWorld).applyQuaternion(display).toArray(), quaternion: compose(body.quaternion_body_to_eci[index]).toArray() };
      pads[id] = { tip: turn(vector(contact.pads[id].tip_position_eci_m[index]).sub(targetWorld).toArray()).toArray(),
        mount: world.clone().add(vector(contact.pads[id].mount_body_m).applyQuaternion(physicalQ)).sub(targetWorld).applyQuaternion(display).toArray(),
        gap: contact.pads[id].gap_m[index], activation: contact.pads[id].activation[index],
        extension: Math.hypot(...contact.pads[id].extension_body_m[index]),
        force: Math.hypot(...contact.pads[id].force_eci_n[index]), moment: Math.hypot(...contact.pads[id].moment_eci_nm[index]) };
    }
  } else if (kind === 'lifecycle') {
    if (!Number.isInteger(index) || index < 0 || index >= data.time_s.length) throw new Error('Assembly replay node is outside its saved trace.');
    const assemblyQ = new THREE.Quaternion(...data.quaternions[index]);
    const target = data.components.find(value => value.name === 'target');
    if (!target) throw new Error('Assembly replay lacks its actual target element.');
    targetWorld = vector(data.positions_eci_m[index]).add(vector(target.position_body_m).sub(vector(data.center_of_mass_body_m[index])).applyQuaternion(assemblyQ));
    targetQuaternion = display.clone().multiply(assemblyQ).multiply(new THREE.Quaternion().setFromRotationMatrix(rotation(target.rotation_element_to_assembly)));
    for (const component of data.components.filter(value => value.name !== 'target')) {
      const id = component.name, elementQ = new THREE.Quaternion().setFromRotationMatrix(rotation(component.rotation_element_to_assembly));
      const physicalQ = assemblyQ.clone().multiply(elementQ);
      const position = vector(component.position_body_m).sub(vector(target.position_body_m)).applyQuaternion(assemblyQ).applyQuaternion(display);
      bodies[id] = { position: position.toArray(), quaternion: display.clone().multiply(physicalQ).toArray() };
      pads[id] = { mount: position.clone().add(vector(contact.pads[id].mount_body_m).applyQuaternion(physicalQ).applyQuaternion(display)).toArray(),
        tip: position.clone().add(vector(contact.pads[id].latched_tip_body_m).applyQuaternion(physicalQ).applyQuaternion(display)).toArray() };
    }
  } else throw new Error('Unknown physical replay sample kind.');
  return { target: { position: [0, 0, 0], quaternion: targetQuaternion.toArray(), position_eci_m: targetWorld.toArray() }, bodies, pads };
}

/** Contact and attached bodies share the approach renderer/camera/controls. */
export class LifecycleScene {
  constructor(host, data, approach = null) {
    finiteContactRecord(data, approach);
    this.host = host; this.data = data; this.active = false;
    this.ids = approach ? approach.tracks.map(track => track.id) : Object.keys(data.finite_contact.chasers);
    this.previousVehicles = host.vehicles;
    this.previousSetCamera = host.setCamera;
    host.setCamera = mode => this.active ? this.setCamera(mode) : this.previousSetCamera?.call(host, mode);
    this.group = new THREE.Group(); this.group.visible = false; host.scene.add(this.group);
    this.target = new THREE.Group(); this.group.add(this.target);
    if (data.target.shape === 'composite_spacecraft') {
      this.target.add(compositeSpacecraft(data.target.geometry));
      this.targetEnvelope = compositeSpacecraft(data.target.geometry, { envelope: true }); this.target.add(this.targetEnvelope);
    }
    else if (data.target.shape === 'cylinder') this.target.add(representativeSpacecraft(data.target.radius, data.target.height || 2 * data.target.half_length));
    else throw new Error('Unsupported physical target geometry.');
    this.buses = []; this.tips = {}; this.stems = {}; this.carriedArms = {}; this.paths = [];
    for (const [number, id] of this.ids.entries()) {
      const radius = data.body_radii_m?.[id] ?? data.body_radius_m;
      if (!(radius > 0)) throw new Error('Physical replay lacks a recorded positive body radius.');
      const color = approach?.tracks[number].color || '#5bd4bd', size = 2 * radius / Math.sqrt(3);
      const bus = new THREE.Mesh(new THREE.BoxGeometry(size, size, size), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: .1, metalness: .65, roughness: .36 }));
      bus.userData.track = number; this.group.add(bus); this.buses.push(bus);
      const envelope = new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 12), new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: .18 }));
      bus.add(envelope);
      const tip = new THREE.Mesh(new THREE.SphereGeometry(.014, 12, 8), new THREE.MeshBasicMaterial({ color: '#f5b567' }));
      this.group.add(tip); this.tips[id] = tip;
      const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3));
      const stem = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: '#f5b567' }));
      this.group.add(stem); this.stems[id] = stem;
      const carriedGeometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3));
      const carried = new THREE.Line(carriedGeometry, new THREE.LineBasicMaterial({ color: '#b8c5cf' }));
      this.group.add(carried); this.carriedArms[id] = carried;
    }
    const samples = lifecycleTimeline(data), points = Object.fromEntries(this.ids.map(id => [id, []]));
    for (const sample of samples) {
      const pose = lifecyclePose(data, sample.kind, sample.index);
      for (const id of this.ids) points[id].push(new THREE.Vector3(...pose.bodies[id].position));
    }
    for (const [number, id] of this.ids.entries()) {
      const path = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points[id]), new THREE.LineBasicMaterial({ color: approach?.tracks[number].color || '#5bd4bd', transparent: true, opacity: .3 }));
      this.group.add(path); this.paths.push(path);
    }
    this.update(samples[0], 0, {});
  }

  setVisible(active) {
    this.active = active; this.group.visible = active;
    if (this.host.model) this.host.model.visible = !active;
    this.host.vehicles = active ? this.buses : this.previousVehicles;
  }

  update(sample, selected, options) {
    this.sample = sample; this.selected = selected; this.options = options;
    const pose = lifecyclePose(this.data, sample.kind, sample.index);
    this.pose = pose; this.target.quaternion.set(...pose.target.quaternion);
    if (this.targetEnvelope) this.targetEnvelope.visible = Boolean(options.envelope);
    for (const [number, id] of this.ids.entries()) {
      this.buses[number].position.set(...pose.bodies[id].position);
      this.buses[number].quaternion.set(...pose.bodies[id].quaternion);
      this.buses[number].material.emissiveIntensity = selected === number ? .28 : .06;
      this.buses[number].children[0].visible = Boolean(options.envelope);
      this.tips[id].position.set(...pose.pads[id].tip);
      const positions = this.stems[id].geometry.attributes.position;
      positions.array.set([...pose.pads[id].mount, ...pose.pads[id].tip]); positions.needsUpdate = true;
      this.stems[id].geometry.computeBoundingSphere();
      const carried = this.carriedArms[id].geometry.attributes.position;
      carried.array.set([...pose.bodies[id].position, ...pose.pads[id].mount]); carried.needsUpdate = true;
      this.carriedArms[id].geometry.computeBoundingSphere();
      this.paths[number].visible = Boolean(options.paths);
    }
    if (this.active && ['follow', 'docking'].includes(this.host.mode)) {
      const next = this.host.mode === 'docking' ? this.tips[this.ids[selected]].position : this.buses[selected].position;
      const offset = this.host.camera.position.clone().sub(this.host.controls.target);
      this.host.controls.target.copy(next); this.host.camera.position.copy(next).add(offset);
    }
    return pose;
  }

  setCamera(mode) {
    this.host.mode = mode;
    const camera = this.host.camera, controls = this.host.controls;
    camera.up.set(0, mode === 'top' ? 1 : 0, mode === 'top' ? 0 : 1);
    const bounds = new THREE.Box3().setFromObject(this.group);
    const close = ['follow', 'docking'].includes(mode);
    const center = close ? (mode === 'docking' ? this.tips[this.ids[this.selected]].position : this.buses[this.selected].position).clone() : bounds.getCenter(new THREE.Vector3());
    const direction = mode === 'top' ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(-2.1, -2.5, 1.8).normalize();
    const extent = close ? Math.max(.8, 5 * this.data.body_radius_m) : Math.max(1, bounds.getSize(new THREE.Vector3()).length() / 2);
    const distance = close ? extent * 2 : extent / (Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * Math.min(1, camera.aspect)) * 1.2;
    controls.target.copy(center); camera.position.copy(center).addScaledVector(direction, distance);
    controls.maxDistance = Math.max(150, distance * 3); camera.far = Math.max(1000, distance * 5); camera.updateProjectionMatrix(); controls.update();
  }

  dispose() {
    this.setVisible(false); this.host.setCamera = this.previousSetCamera;
    this.group.traverse(object => { object.geometry?.dispose(); if (Array.isArray(object.material)) object.material.forEach(value => value.dispose()); else object.material?.dispose(); });
    this.group.removeFromParent();
  }
}
