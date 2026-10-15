import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { representativeSpacecraft, compositeSpacecraft, starField } from './spacecraft.js';

/** Recorded geometry. Finite chasers use their declared spherical body radius. */
export class MissionScene {
  constructor(container, onSelect) {
    this.container = container;
    this.onSelect = onSelect;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor('#010306');
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.setAttribute('aria-label', 'Recorded chaser and target geometry');
    container.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.add(starField());
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.005, 1000);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.minDistance = 0.1;
    this.controls.maxDistance = 150;
    this.scene.add(new THREE.AmbientLight('#b4cadd', 2.0));
    const key = new THREE.DirectionalLight('#e0efff', 3.5);
    key.position.set(-3, -5, 7);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight('#4d8eaf', 1.8);
    fill.position.set(4, 2, 1);
    this.scene.add(fill);
    this.model = new THREE.Group();
    this.scene.add(this.model);
    this.raycaster = new THREE.Raycaster();
    this.renderer.domElement.addEventListener('pointerdown', event => { this.pointerDown = [event.clientX, event.clientY]; });
    this.renderer.domElement.addEventListener('pointerup', event => {
      if (!this.pointerDown || Math.hypot(event.clientX - this.pointerDown[0], event.clientY - this.pointerDown[1]) > 5) return;
      const rect = this.renderer.domElement.getBoundingClientRect();
      this.raycaster.setFromCamera(new THREE.Vector2(2 * (event.clientX - rect.left) / rect.width - 1, 1 - 2 * (event.clientY - rect.top) / rect.height), this.camera);
      const hit = this.raycaster.intersectObjects(this.vehicles || [], false)[0];
      if (hit) this.onSelect(hit.object.userData.track);
    });
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.mode = 'overview';
    this.options = { paths: true, tubes: true };
  }

  clear() {
    this.model.traverse(object => {
      object.geometry?.dispose();
      if (Array.isArray(object.material)) object.material.forEach(material => material.dispose());
      else object.material?.dispose();
    });
    this.model.clear();
  }

  load(data) {
    this.clear();
    this.data = data;
    this.index = 0;
    this.selected = 0;
    const bounds = new THREE.Box3();
    const radius = data.chaser_body_radius_m || 0;
    for (const track of data.tracks) for (const row of track.states) {
      const position = new THREE.Vector3(...row.slice(0, 3));
      bounds.expandByPoint(position.clone().addScalar(-radius));
      bounds.expandByPoint(position.clone().addScalar(radius));
    }
    if(data.target.shape==='composite_spacecraft'){
      const rational = value => {
        const [n, d] = String(value).split('/');
        return Number(n) / (d === undefined ? 1 : Number(d));
      };
      const vertices = data.target.geometry.components.flatMap(component =>
        component.exact_vertices_body_m.map(row => row.map(rational)));
      for (const rotation of data.rotations || [[[1,0,0],[0,1,0],[0,0,1]]]) {
        const matrix = new THREE.Matrix3().set(...rotation.flat());
        for (const vertex of vertices) bounds.expandByPoint(new THREE.Vector3(...vertex).applyMatrix3(matrix));
      }
    } else {
      bounds.expandByPoint(new THREE.Vector3(-1, -1, -1));
      bounds.expandByPoint(new THREE.Vector3(1, 1, 1));
    }
    this.bounds = bounds;
    this.center = bounds.getCenter(new THREE.Vector3());
    this.extent = Math.max(...bounds.getSize(new THREE.Vector3()).toArray()) * 0.55;
    const grid = new THREE.GridHelper(Math.max(4, Math.ceil(this.extent * 2)), 20, '#2a4054', '#1b2b3b');
    this.grid = grid;
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -Math.max(1, (data.target.half_length || 0.3) * 2.5);
    grid.material.transparent = true;
    grid.material.opacity = 0.65;
    this.model.add(grid);
    const axes = new THREE.AxesHelper(Math.max(1, this.extent * 0.3));
    this.axes = axes;
    axes.material.transparent = true;
    axes.material.opacity = 0.5;
    this.model.add(axes);
    this.target = new THREE.Group();
    this.target.matrixAutoUpdate = false;
    this.model.add(this.target);
    this.envelope=this.envelopeEdges=null;
    if(data.target.shape==='composite_spacecraft'){
      this.target.add(compositeSpacecraft(data.target.geometry));
      this.envelope=compositeSpacecraft(data.target.geometry,{envelope:true});
      this.envelopeEdges=new THREE.Group();
      this.target.add(this.envelope,this.envelopeEdges);
    }
    if (data.target.shape === 'cylinder' && data.target.radius > 0 && data.rotations) {
      const radius = data.target.radius;
      const height = data.target.height || 2 * data.target.half_length;
      const envelopeRadius = data.collision_envelope?.radius ?? radius;
      const envelopeHeight = data.collision_envelope ? 2 * data.collision_envelope.half_length : height;
      const cylinder = new THREE.Mesh(new THREE.CylinderGeometry(envelopeRadius, envelopeRadius, envelopeHeight, 64), new THREE.MeshStandardMaterial({ color: '#91a5b7', transparent: true, opacity: 0.06, depthWrite: false, roughness: 0.44, metalness: 0.58 }));
      this.envelope = cylinder;
      cylinder.rotation.x = Math.PI / 2;
      this.target.add(cylinder);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(cylinder.geometry, 25), new THREE.LineBasicMaterial({ color: '#d6e3ed', transparent: true, opacity: 0.38 }));
      edges.rotation.copy(cylinder.rotation);
      this.target.add(edges);
      edges.material.opacity = 0.18;
      this.envelopeEdges = edges;
      this.target.add(representativeSpacecraft(radius, height));
      for (const sign of [-1, 1]) {
        const rim = new THREE.Mesh(new THREE.TorusGeometry(radius, radius * 0.015, 6, 64), new THREE.MeshStandardMaterial({ color: '#c4d4df', metalness: 0.5, roughness: 0.45 }));
        rim.position.z = sign * height / 2;
        rim.visible = false;
      }
    }
    this.vehicles = [];
    this.traces = [];
    this.boxes = [];
    this.centers = [];
    this.goals = [];
    for (const [trackIndex, track] of data.tracks.entries()) {
      const path = new THREE.Line(new THREE.BufferGeometry().setFromPoints(track.states.map(row => new THREE.Vector3(...row.slice(0, 3)))), new THREE.LineBasicMaterial({ color: track.color, transparent: true, opacity: 0.56 }));
      this.model.add(path);
      this.traces.push(path);
      const glyph = new THREE.Mesh(data.chaser_body_radius_m > 0 ? new THREE.SphereGeometry(data.chaser_body_radius_m,24,16) : new THREE.OctahedronGeometry(0.045), new THREE.MeshStandardMaterial({ color: track.color, emissive: track.color, emissiveIntensity: 0.23, roughness: 0.45,metalness:.5 }));
      if (data.chaser_body_radius_m > 0) {
        const envelope = new THREE.Mesh(new THREE.SphereGeometry(data.chaser_body_radius_m, 20, 12), new THREE.MeshBasicMaterial({color:track.color,wireframe:true,transparent:true,opacity:0.18}));
        // Child scale is undone in update: physical radius never changes when
        // the selected display glyph is highlighted.
        glyph.add(envelope);
      }
      glyph.userData.track = trackIndex;
      this.model.add(glyph);
      this.vehicles.push(glyph);
      const center = new THREE.Mesh(new THREE.SphereGeometry(0.012, 10, 8), new THREE.MeshBasicMaterial({ color: '#e9f0f7' }));
      this.model.add(center);
      this.centers.push(center);
      const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: track.color, transparent: true, opacity: 0.9 }));
      this.model.add(box);
      this.boxes.push(box);
      const goal = new THREE.Mesh(new THREE.OctahedronGeometry(0.026), new THREE.MeshBasicMaterial({ color: '#f5b567', wireframe: true }));
      if (track.goal_body && data.rotations) { goal.position.set(...track.goal_body); this.target.add(goal); }
      else goal.visible = false;
      this.goals.push(goal);
      if (track.surface_point_body_m && data.rotations) {
        const patch = new THREE.Mesh(new THREE.SphereGeometry(0.014, 12, 8), new THREE.MeshBasicMaterial({color:track.color}));
        patch.position.set(...track.surface_point_body_m);
        this.target.add(patch);
      }
    }
    this.update(0, 0, this.options);
    this.setCamera(this.mode);
  }

  setCamera(mode) {
    this.mode = mode;
    if (!this.data) return;
    this.camera.up.set(0, mode === 'top' ? 1 : 0, mode === 'top' ? 0 : 1);
    const center = mode === 'docking' && this.data.tracks[this.selected].goal_body && this.data.rotations ? new THREE.Vector3(...this.data.tracks[this.selected].goal_body).applyMatrix4(this.target.matrix) : mode === 'follow' ? new THREE.Vector3(...this.data.tracks[this.selected].states[this.index].slice(0, 3)) : mode === 'overview' ? this.center : new THREE.Vector3();
    const extent = mode === 'docking' || mode === 'follow' ? Math.max(1.0,this.data.chaser_body_radius_m*5 || 1.5) : this.extent;
    this.controls.target.copy(center);
    const offset = mode === 'top' ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(-2.1, -2.5, 1.8).normalize();
    let distance = extent * 3.8;
    if (mode === 'overview' || mode === 'top') {
      const right = new THREE.Vector3().crossVectors(this.camera.up, offset).normalize();
      const up = new THREE.Vector3().crossVectors(offset, right).normalize();
      const tangentY = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
      const tangentX = tangentY * this.camera.aspect;
      distance = 0;
      for (const x of [this.bounds.min.x, this.bounds.max.x])
        for (const y of [this.bounds.min.y, this.bounds.max.y])
          for (const z of [this.bounds.min.z, this.bounds.max.z]) {
            const delta = new THREE.Vector3(x, y, z).sub(center);
            distance = Math.max(distance, delta.dot(offset) + Math.abs(delta.dot(right)) / tangentX,
              delta.dot(offset) + Math.abs(delta.dot(up)) / tangentY);
          }
      distance = Math.max(1, distance * 1.18);
    }
    this.controls.maxDistance = Math.max(150, distance * 3);
    this.camera.far = Math.max(1000, distance * 5);
    this.camera.updateProjectionMatrix();
    this.camera.position.copy(center).addScaledVector(offset, distance);
    this.controls.update();
  }

  update(index, selected, options) {
    if (!this.data) return;
    this.index = index;
    this.selected = selected;
    this.options = options;
    this.grid.visible = this.axes.visible = Boolean(options.grid);
    if (this.envelope) this.envelope.visible = this.envelopeEdges.visible = Boolean(options.envelope);
    const rotation = this.data.rotations?.[index];
    if (rotation) {
      this.target.matrix.set(rotation[0][0], rotation[0][1], rotation[0][2], 0, rotation[1][0], rotation[1][1], rotation[1][2], 0, rotation[2][0], rotation[2][1], rotation[2][2], 0, 0, 0, 0, 1);
    }
    this.data.tracks.forEach((track, number) => {
      this.vehicles[number].position.set(...track.states[index].slice(0, 3));
      this.vehicles[number].scale.setScalar(this.data.chaser_body_radius_m > 0 ? 1 : number === selected ? 1.2 : 0.8);
      this.vehicles[number].material.emissiveIntensity=number===selected?.28:.06;
      this.vehicles[number].children.forEach(child => { child.scale.setScalar(1 / this.vehicles[number].scale.x); child.visible = Boolean(options.envelope); });
      this.traces[number].visible = options.paths;
      this.traces[number].material.opacity = number === selected ? 0.95 : 0.33;
      const tube = options.tubes && track.center && track.omega;
      this.boxes[number].visible = Boolean(tube);
      this.centers[number].visible = Boolean(tube);
      if (tube) {
        this.boxes[number].position.set(...track.center[index]);
        this.boxes[number].scale.set(...track.omega[index].map(width => 2 * width));
        this.centers[number].position.set(...track.center[index]);
      }
    });
    if (this.mode === 'follow' || this.mode === 'docking' && this.data.tracks[selected].goal_body && rotation) {
      const position = this.mode === 'docking' ? new THREE.Vector3(...this.data.tracks[selected].goal_body).applyMatrix4(this.target.matrix) : this.vehicles[selected].position;
      const offset = this.camera.position.clone().sub(this.controls.target);
      this.controls.target.copy(position);
      this.camera.position.copy(position).add(offset);
    }
  }

  resize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    if (this.data) this.setCamera(this.mode);
  }

  render() { this.controls.update(); this.renderer.render(this.scene, this.camera); }
  dispose() { this.resizeObserver.disconnect(); this.clear(); this.controls.dispose(); this.renderer.dispose(); this.renderer.domElement.remove(); }
}
