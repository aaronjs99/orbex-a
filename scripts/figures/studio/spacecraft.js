import * as THREE from 'three';

/** The same component declaration used by planning, continuous admission and capture. */
export function compositeSpacecraft(geometry, { envelope = false } = {}) {
  if (geometry?.schema !== 'orbexa.spacecraft-geometry.v1' || !geometry.components?.length)
    throw new Error('Composite replay requires its recorded component geometry.');
  const craft = new THREE.Group();
  const materials = {
    bus: new THREE.MeshStandardMaterial({color:'#c8a452',metalness:.67,roughness:.47}),
    solar_array: new THREE.MeshStandardMaterial({color:'#163953',metalness:.46,roughness:.33}),
    antenna: new THREE.MeshStandardMaterial({color:'#b9c4cb',metalness:.68,roughness:.4}),
    structure: new THREE.MeshStandardMaterial({color:'#8798a6',metalness:.72,roughness:.39}),
  };
  for(const component of geometry.components){
    const c=component.center_m,h=component.half_extents_m,m=component.rotation_component_to_body;
    if(![...c,...h,...m.flat()].every(Number.isFinite)||h.some(x=>x<=0))throw new Error('Invalid recorded spacecraft component.');
    if(envelope){
      const rational=value=>{const [n,d]=String(value).split('/');return Number(n)/(d===undefined?1:Number(d));};
      const points=component.exact_vertices_body_m.map(row=>row.map(rational)),vertices=[];
      for(let i=0;i<8;i++)for(const bit of [1,2,4])if((i&bit)===0)vertices.push(...points[i],...points[i|bit]);
      craft.add(new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(vertices,3)),
        new THREE.LineBasicMaterial({color:'#b8d5e9',transparent:true,opacity:.3})));
      continue;
    }
    const material=materials[component.role] || materials.structure;
    const mesh=new THREE.Mesh(new THREE.BoxGeometry(...h.map(v=>2*v)),material);
    mesh.name=component.id; mesh.position.set(...c);
    mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(m[0][0],m[0][1],m[0][2],0,m[1][0],m[1][1],m[1][2],0,m[2][0],m[2][1],m[2][2],0,0,0,0,1));
    craft.add(mesh);
    const edges=new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),new THREE.LineBasicMaterial({color:component.role==='solar_array'?'#6385a3':'#d4c392',transparent:true,opacity:.5}));
    edges.position.copy(mesh.position);edges.quaternion.copy(mesh.quaternion);craft.add(edges);
    if(component.role==='solar_array'){
      const cells=[];
      for(let i=1;i<16;i++){const x=-h[0]+2*h[0]*i/16;cells.push(x,-h[1],h[2],x,h[1],h[2]);}
      for(let i=1;i<8;i++){const y=-h[1]+2*h[1]*i/8;cells.push(-h[0],y,h[2],h[0],y,h[2]);}
      const lines=new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(cells,3)),new THREE.LineBasicMaterial({color:'#7698b7',transparent:true,opacity:.48}));
      lines.position.copy(mesh.position);lines.quaternion.copy(mesh.quaternion);craft.add(lines);
    }
  }
  return craft;
}

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
