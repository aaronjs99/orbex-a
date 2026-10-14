import * as THREE from 'three';

/** Representative cylindrical spacecraft with the modeled contact surface.
 * Cladding and flush panels describe a concept, not an identified flight asset.
 */
export function representativeSpacecraft(radius, height) {
  const craft = new THREE.Group();
  const metal = new THREE.MeshStandardMaterial({ color: '#afbbc5', roughness: 0.43, metalness: 0.75 });
  const blanket = new THREE.MeshStandardMaterial({ color: '#aa8842', roughness: 0.62, metalness: 0.64 });
  const solar = new THREE.MeshStandardMaterial({ color: '#183756', roughness: 0.33, metalness: 0.55 });
  for (const material of [metal, blanket, solar]) {
    material.polygonOffset = true;
    material.polygonOffsetFactor = 1;
    material.polygonOffsetUnits = 1;
  }
  const facets = 128;
  const geometry = new THREE.CylinderGeometry(radius, radius, height, facets, 1, false);
  const caps = geometry.groups.slice(1);
  geometry.clearGroups();
  for (let facet = 0; facet < facets; facet++) {
    const column = facet % 32;
    geometry.addGroup(facet * 6, 6, column >= 7 && column < 25 ? 2 : 0);
  }
  for (const cap of caps) geometry.addGroup(cap.start, cap.count, 1);
  const hull = new THREE.Mesh(geometry, [blanket, metal, solar]);
  hull.rotation.x = Math.PI / 2;
  craft.add(hull);
  // Flush cladding/cell seams follow the same inscribed cylindrical surface.
  const seams = [];
  const vertex = (facet, z) => {
    const theta = facet * 2 * Math.PI / facets;
    return [radius * Math.sin(theta), -radius * Math.cos(theta), z];
  };
  for (let facet = 0; facet < facets; facet++) {
    const column = facet % 32;
    if ([0, 7, 13, 19, 25].includes(column)) seams.push(...vertex(facet, -height/2), ...vertex(facet, height/2));
    for (const fraction of [-.5, -.25, 0, .25, .5]) {
      if (column >= 7 && column < 25 || Math.abs(fraction) === .5)
        seams.push(...vertex(facet, height * fraction), ...vertex(facet + 1, height * fraction));
    }
  }
  craft.add(new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(seams, 3)),
    new THREE.LineBasicMaterial({ color: '#8397a7', transparent: true, opacity: .62 })));
  for (const sign of [-1, 1]) {
    const aperture = new THREE.Mesh(new THREE.CircleGeometry(radius * .22, 48),
      new THREE.MeshStandardMaterial({ color: '#263947', metalness: .65, roughness: .4, side: THREE.DoubleSide }));
    aperture.position.z = sign * height / 2;
    if (sign < 0) aperture.rotation.x = Math.PI;
    craft.add(aperture);
    const ring = new THREE.Mesh(new THREE.RingGeometry(radius * .72, radius * .76, 96),
      new THREE.MeshStandardMaterial({ color: '#c4c8bd', metalness: .6, roughness: .48, side: THREE.DoubleSide }));
    ring.position.z = sign * height / 2;
    craft.add(ring);
  }
  return craft;
}

export function starField() {
  // Decorative background, independent of the numerical mission and frame.
  let seed = 42817;
  const random = () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 4294967296; };
  const vertices = [];
  for (let index = 0; index < 750; index++) {
    const z = random() * 2 - 1, azimuth = random() * Math.PI * 2;
    const radial = Math.sqrt(1 - z * z);
    vertices.push(300 * radial * Math.cos(azimuth), 300 * radial * Math.sin(azimuth), 300 * z);
  }
  const stars = new THREE.Points(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3)), new THREE.PointsMaterial({ color: '#d7e4ef', size: 0.55, sizeAttenuation: true, transparent: true, opacity: 0.75, depthWrite: false }));
  stars.raycast = () => {};
  return stars;
}
