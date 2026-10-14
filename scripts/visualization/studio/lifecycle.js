import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { representativeSpacecraft, starField } from './spacecraft.js';
import { SceneVideoExporter, downloadVideo } from './recorder.js';

const $ = id => document.getElementById(id);
const views = [];
let data, index = 0, playing = false, started = 0, startTime = 0, visible = true, exporting = false;
window.addEventListener('message',event=>{if(event.origin===location.origin&&event.source===window.parent&&event.data?.type==='orbexa.lifecycle.visibility'){visible=event.data.active;if(!visible)stop();}});

function view(container, distance) {
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setClearColor('#010306'); renderer.setPixelRatio(Math.min(devicePixelRatio || 1,2));
  container.append(renderer.domElement);
  const scene = new THREE.Scene(); scene.add(starField());
  const camera = new THREE.PerspectiveCamera(38,1,.001,1000); camera.up.set(0,0,1); camera.position.set(-distance,-distance,distance*.65);
  const controls = new OrbitControls(camera,renderer.domElement); controls.enableDamping = true;
  scene.add(new THREE.AmbientLight('#c1d3e4',1.9));
  const light = new THREE.DirectionalLight('#e8f3ff',3); light.position.set(-4,-6,8); scene.add(light);
  new ResizeObserver(() => {const w=container.clientWidth,h=container.clientHeight;if(w&&h){renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}}).observe(container);
  const result={renderer,scene,camera,controls}; views.push(result); return result;
}

function setup() {
  const near=view($('life-near'),2.5), orbit=view($('life-orbit'),2.8);
  const assembly = new THREE.Group(); near.scene.add(assembly); near.assembly=assembly;
  near.structure=new THREE.Group(); assembly.add(near.structure);
  near.structure.add(representativeSpacecraft(data.target.radius,data.target.height || 2*data.target.half_length));
  const envelope=new THREE.Mesh(new THREE.CylinderGeometry(data.target.radius,data.target.radius,data.target.height||2*data.target.half_length,48),new THREE.MeshBasicMaterial({color:'#a5b7ca',wireframe:true,transparent:true,opacity:.1}));envelope.rotation.x=Math.PI/2;near.structure.add(envelope);
  for(const component of data.components.filter(x=>x.name!=='target')){
    const size=(data.body_radii_m?.[component.name] ?? data.body_radius_m)*2/Math.sqrt(3);
    const bus=new THREE.Mesh(new THREE.BoxGeometry(size,size,size),new THREE.MeshStandardMaterial({color:'#5bd4bd',metalness:.65,roughness:.32}));
    bus.position.set(...component.position_body_m);
    const m=component.rotation_element_to_assembly;
    bus.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(m[0][0],m[0][1],m[0][2],0,m[1][0],m[1][1],m[1][2],0,m[2][0],m[2][1],m[2][2],0,0,0,0,1));
    near.structure.add(bus);
    const anchor=new THREE.Mesh(new THREE.SphereGeometry(.012,8,6),new THREE.MeshBasicMaterial({color:'#f5b567'}));
    const contact=data.surface_contacts?.[component.name];
    const surface=new THREE.Vector3(...(contact?.surface_point_body_m ?? component.position_body_m));
    if(!contact){const radial=Math.hypot(surface.x,surface.y);surface.x*=data.target.radius/radial;surface.y*=data.target.radius/radial;}
    anchor.position.copy(surface);near.structure.add(anchor);
    const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([surface,bus.position]),new THREE.LineBasicMaterial({color:'#f5b567'}));near.structure.add(line);
  }
  near.force=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(),.6,'#f5b567',.1,.04);assembly.add(near.force);
  const earth=new THREE.Mesh(new THREE.SphereGeometry(1,64,32),new THREE.MeshStandardMaterial({color:'#194a79',roughness:.83,metalness:0}));orbit.scene.add(earth);
  const atmosphere=new THREE.Mesh(new THREE.SphereGeometry(1.018,48,24),new THREE.MeshBasicMaterial({color:'#74b9ff',transparent:true,opacity:.08,side:THREE.BackSide}));orbit.scene.add(atmosphere);
  for(let latitude=-60;latitude<=60;latitude+=30){const angle=latitude*Math.PI/180,r=Math.cos(angle);const points=Array.from({length:129},(_,i)=>new THREE.Vector3(r*Math.cos(i*2*Math.PI/128),r*Math.sin(i*2*Math.PI/128),Math.sin(angle)));orbit.scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:'#668da7',transparent:true,opacity:.17})));}
  const points=data.positions_eci_m.map(p=>new THREE.Vector3(...p.map(v=>v/data.earth_radius_m)));
  orbit.scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:'#74b9ff',transparent:true,opacity:.65})));
  orbit.marker=new THREE.Mesh(new THREE.OctahedronGeometry(.018),new THREE.MeshBasicMaterial({color:'#f5b567'}));orbit.scene.add(orbit.marker);
  const interfaceRadius=1+data.entry_interface_altitude_m/data.earth_radius_m;
  const ring=Array.from({length:257},(_,i)=>new THREE.Vector3(interfaceRadius*Math.cos(i*2*Math.PI/256),interfaceRadius*Math.sin(i*2*Math.PI/256),0));
  orbit.scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(ring),new THREE.LineBasicMaterial({color:'#5bd4bd',transparent:true,opacity:.5})));
}

function paint(){
  const near=views[0],orbit=views[1];
  near.assembly.quaternion.set(...data.quaternions[index]);
  near.structure.position.set(...data.center_of_mass_body_m[index].map(x=>-x));
  const force=new THREE.Vector3(...data.applied_force_body_n[index]); near.force.visible=force.length()>1e-7;if(near.force.visible)near.force.setDirection(force.normalize());
  orbit.marker.position.set(...data.positions_eci_m[index].map(x=>x/data.earth_radius_m));
  $('life-seek').value=index;
  $('life-seek').setAttribute('aria-valuetext',`Recorded lifecycle sample ${index+1} of ${data.time_s.length}`);
  $('life-clock').textContent=`${(data.elapsed_s[index]/60).toFixed(1)} min · ${index+1}/${data.time_s.length}`;
  $('life-phase').textContent=data.phases[index].replaceAll('_',' ');
  $('life-spin').textContent=data.angular_speed_rad_s[index].toExponential(3)+' rad/s';
  $('life-fuel').textContent=data.fuel_kg[index].toFixed(3)+' kg';
  $('life-altitude').textContent=(data.altitude_m[index]/1000).toFixed(2)+' km';
  $('life-perigee').textContent=(data.perigee_altitude_m[index]/1000).toFixed(2)+' km';
  if(window.parent!==window)window.parent.postMessage({type:'orbexa.lifecycle.clock',elapsed:data.elapsed_s[index],index,count:data.time_s.length},location.origin);
}
function stop(){playing=false;$('life-play').textContent='Play';}
function seek(i){if(exporting)return;stop();index=Math.max(0,Math.min(data.time_s.length-1,Number(i)));paint();}
$('life-play').addEventListener('click',()=>{if(exporting)return;if(playing){stop();return;}if(index===data.time_s.length-1)index=0;playing=true;started=performance.now();startTime=data.elapsed_s[index];$('life-play').textContent='Pause';});
$('life-first').addEventListener('click',()=>seek(0));$('life-last').addEventListener('click',()=>seek(data.time_s.length-1));$('life-seek').addEventListener('input',()=>seek($('life-seek').value));
try{
  const path=new URLSearchParams(location.search).get('data');
  if(!path)throw new Error('No saved lifecycle was selected.');
  const url=new URL(path,location.href);if(url.origin!==location.origin)throw new Error('Lifecycle replay requires a local saved dataset.');
  const response=await fetch(url);if(!response.ok)throw new Error('Saved lifecycle data is unavailable.');
  data=await response.json();
  if(data.schema!=='orbexa.lifecycle.studio.v1'||!data.time_s?.length)throw new Error('Unsupported lifecycle replay dataset.');
  $('life-provenance').textContent=data.synthetic?'Synthetic physics fixture · no ADTMPC approach inferred':data.provenance;
  const contactCount=Object.keys(data.surface_contacts ?? {}).length;
  if(data.attachment_success){const summary=document.createElement('p');summary.className='sample-note';summary.textContent=`${contactCount} chasers attached at selected hull surface points · idealized chaser attachment interface · no target docking hardware`;document.querySelector('.section-title').after(summary);}
  const terminal=data.terminal_handoff?.terminal_execution;
  if(terminal&&Number.isFinite(terminal.physical_duration_seconds)){const summary=document.createElement('p');summary.className='sample-note';summary.textContent=`Attachment follows ${terminal.physical_duration_seconds.toFixed(2)} s of recorded certified terminal tracking · ${terminal.holds} admitted fleet holds · original approach preserved`;document.querySelector('.section-title').after(summary);}
  $('life-scope').textContent=data.scope;
  $('life-raw').href=path;$('life-seek').max=data.time_s.length-1;
  setup();paint();
  const actions=document.createElement('div');actions.className='replay-actions';const status=document.createElement('span');status.setAttribute('role','status');status.textContent='Export both camera views as a 20 second video.';const button=document.createElement('button');button.textContent='Export mission video';button.disabled=!SceneVideoExporter.supported();actions.append(status,button);document.querySelector('main').append(actions);
  button.addEventListener('click',async()=>{if(exporting)return;stop();exporting=true;button.disabled=true;const canvas=document.createElement('canvas');canvas.width=1280;canvas.height=640;const ctx=canvas.getContext('2d');const adapter={renderer:{domElement:canvas},render(){ctx.fillStyle='#010306';ctx.fillRect(0,0,1280,640);views.forEach((v,i)=>{v.renderer.render(v.scene,v.camera);const source=v.renderer.domElement;const scale=Math.min(630/source.width,620/source.height);const w=source.width*scale,h=source.height*scale;ctx.drawImage(source,i*640+(640-w)/2,(640-h)/2,w,h);});}};try{const file=await new SceneVideoExporter().record(adapter,data,data.synthetic?'Synthetic post-capture physics fixture':'Attachment, detumbling and orbital delivery',{onFrame:i=>{index=i;paint();},onProgress:p=>status.textContent=`Exporting recorded lifecycle · ${p}%`});downloadVideo(file.blob,`ORBEX-A-physical-lifecycle.${file.extension}`);status.textContent='Mission video created. Download requested.';}catch(error){status.textContent=error.message;}finally{exporting=false;button.disabled=false;}});
  function animate(now){if(visible&&!exporting){if(playing){const time=startTime+(now-started)*.3;while(index<data.time_s.length-1&&data.elapsed_s[index+1]<=time)index++;paint();if(index===data.time_s.length-1)stop();}for(const v of views){v.controls.update();v.renderer.render(v.scene,v.camera);}}requestAnimationFrame(animate);}requestAnimationFrame(animate);
}catch(error){$('life-error').hidden=false;$('life-error').textContent=error.message;for(const id of ['life-play','life-first','life-last','life-seek'])$(id).disabled=true;}
