import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { representativeSpacecraft, compositeSpacecraft, starField } from './spacecraft.js';
import { SceneVideoExporter, downloadVideo } from './recorder.js';
import { LifecycleScene } from './lifecycle-scene.js';
import { lifecycleTimeline, phaseTitle } from './mission-timeline.js';

const $ = id => document.getElementById(id);
const views = [];
const parameters = new URLSearchParams(location.search);
const session = parameters.get('session');
const controlled = window.parent !== window && parameters.get('controlled') === '1' && Boolean(session);
document.body.classList.toggle('parent-controlled', controlled);
let data, samples, index = 0, playing = false, started = 0, startTime = 0, visible = !controlled, exporting = false;
let receiveParentData;
window.addEventListener('message', event => {
  if (!controlled || event.origin !== location.origin || event.source !== window.parent || event.data?.session !== session) return;
  if (event.data.type === 'orbexa.lifecycle.load' && receiveParentData && event.data.data?.schema === 'orbexa.lifecycle.studio.v1') {
    receiveParentData(event.data.data);
    receiveParentData = null;
  }
  if (event.data.type === 'orbexa.lifecycle.visibility' && typeof event.data.active === 'boolean') {
    const becomingVisible = !visible && event.data.active;
    visible = event.data.active;
    if (!visible) stop();
    if (becomingVisible) views.forEach(item => item.resize());
  }
  if (event.data.type === 'orbexa.lifecycle.seek' && samples && Number.isInteger(event.data.index) && event.data.index >= 0 && event.data.index < samples.length) seek(event.data.index);
});

function view(container, distance) {
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setClearColor('#010306'); renderer.setPixelRatio(Math.min(devicePixelRatio || 1,2));
  container.append(renderer.domElement);
  const scene = new THREE.Scene(); scene.add(starField());
  const camera = new THREE.PerspectiveCamera(38,1,.001,1000); camera.up.set(0,0,1); camera.position.set(-distance,-distance,distance*.65);
  const controls = new OrbitControls(camera,renderer.domElement); controls.enableDamping = true;
  scene.add(new THREE.AmbientLight('#c1d3e4',1.9));
  const light = new THREE.DirectionalLight('#e8f3ff',3); light.position.set(-4,-6,8); scene.add(light);
  const resize = () => {const w=container.clientWidth,h=container.clientHeight;if(w&&h){renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}};
  new ResizeObserver(resize).observe(container);
  const result={renderer,scene,camera,controls,resize}; views.push(result); return result;
}

function setup() {
  const composite=data.target.shape==='composite_spacecraft';
  const near=view($('life-near'),composite ? (data.target.bounding_sphere_radius || 10)*1.9 : 2.5), orbit=view($('life-orbit'),2.8);
  if (data.finite_contact) {
    const host = { ...near, mode:'mission', model:new THREE.Group() };
    near.lifecycleScene = new LifecycleScene(host,data); near.lifecycleScene.setVisible(true); near.lifecycleScene.setCamera('mission');
  } else {
  const assembly = new THREE.Group(); near.scene.add(assembly); near.assembly=assembly;
  near.structure=new THREE.Group(); assembly.add(near.structure);
  if(composite)near.structure.add(compositeSpacecraft(data.target.geometry));
  else if(data.target.shape==='cylinder')near.structure.add(representativeSpacecraft(data.target.radius,data.target.height || 2*data.target.half_length));
  else throw new Error('Unsupported recorded target geometry.');
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
    if(!contact){if(composite)throw new Error('Composite attachment replay requires actual recorded surface contacts.');const radial=Math.hypot(surface.x,surface.y);surface.x*=data.target.radius/radial;surface.y*=data.target.radius/radial;}
    anchor.position.copy(surface);near.structure.add(anchor);
    const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([surface,bus.position]),new THREE.LineBasicMaterial({color:'#f5b567'}));near.structure.add(line);
  }
  near.force=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(),.6,'#f5b567',.1,.04);assembly.add(near.force);
  }
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
  const sample=samples[index],node=sample.index,contact=sample.kind==='contact';
  if (near.lifecycleScene) near.lifecycleScene.update(sample,0,{paths:true});
  else {
  near.assembly.quaternion.set(...data.quaternions[node]);
  near.structure.position.set(...data.center_of_mass_body_m[node].map(x=>-x));
  const force=new THREE.Vector3(...data.applied_force_body_n[node]); near.force.visible=force.length()>1e-7;if(near.force.visible)near.force.setDirection(force.normalize());
  }
  const position=contact?data.finite_contact.target.position_eci_m[node]:data.positions_eci_m[node];
  orbit.marker.position.set(...position.map(x=>x/data.earth_radius_m));
  $('life-seek').value=index;
  $('life-seek').setAttribute('aria-valuetext',`Recorded lifecycle sample ${index+1} of ${samples.length}`);
  $('life-clock').textContent=`t = ${sample.time.toFixed(2)} s · ${index+1}/${samples.length}`;
  $('life-phase').textContent=phaseTitle(sample.phase);
  const spin=contact?Math.hypot(...data.finite_contact.target.angular_velocity_body_rad_s[node]):data.angular_speed_rad_s[node];
  const fuel=contact?Object.values(data.finite_contact.chasers).reduce((sum,body)=>sum+body.fuel_kg[node],0):data.fuel_kg[node];
  $('life-spin').textContent=spin.toExponential(3)+' rad/s';
  $('life-fuel').textContent=fuel.toFixed(3)+' kg';
  $('life-altitude').textContent=((Math.hypot(...position)-data.earth_radius_m)/1000).toFixed(2)+' km';
  $('life-perigee').textContent=contact?'Not recorded during contact':(data.perigee_altitude_m[node]/1000).toFixed(2)+' km';
  $('life-contact-count').textContent=contact?`${Object.keys(data.finite_contact.chasers).length} buses + target · finite contact`:data.attachment_success?`${Object.keys(data.surface_contacts??{}).length} attachments · ${data.finite_contact?'finite latch':'idealized contact'}`:'Attachment incomplete';
  if(window.parent!==window)window.parent.postMessage({type:'orbexa.lifecycle.clock',session,time:sample.time,elapsed:sample.time-samples[0].time,index,count:samples.length},location.origin);
}
function stop(){playing=false;$('life-play').textContent='Play';}
function seek(i){if(exporting)return;stop();index=Math.max(0,Math.min(samples.length-1,Number(i)));paint();}
$('life-play').addEventListener('click',()=>{if(exporting)return;if(playing){stop();return;}if(index===samples.length-1)index=0;playing=true;started=performance.now();startTime=samples[index].time;$('life-play').textContent='Pause';});
$('life-first').addEventListener('click',()=>seek(0));$('life-last').addEventListener('click',()=>seek(samples.length-1));$('life-seek').addEventListener('input',()=>seek($('life-seek').value));
try{
  const path=parameters.get('data');
  if (controlled && parameters.get('source') === 'parent') {
    data = await new Promise(resolve => {
      receiveParentData = resolve;
      window.parent.postMessage({type:'orbexa.lifecycle.request',session},location.origin);
    });
  } else {
    if(!path)throw new Error('No saved lifecycle was selected.');
    const url=new URL(path,location.href);if(url.origin!==location.origin)throw new Error('Lifecycle replay requires a local saved dataset.');
    const response=await fetch(url);if(!response.ok)throw new Error('Saved lifecycle data is unavailable.');
    data=await response.json();
  }
  if(data.schema!=='orbexa.lifecycle.studio.v1'||!data.time_s?.length)throw new Error('Unsupported lifecycle replay dataset.');
  samples=lifecycleTimeline(data);
  $('life-provenance').textContent=data.synthetic?'Synthetic physics fixture · no ADTMPC approach inferred':data.provenance;
  const contactCount=Object.keys(data.surface_contacts ?? {}).length;
  $('life-contact-count').textContent=data.attachment_success?`${contactCount} attachments · idealized contact`:'Attachment incomplete';
  const terminal=data.terminal_handoff?.terminal_execution;
  if(terminal&&Number.isFinite(terminal.physical_duration_seconds))$('life-provenance').textContent+=` · Terminal handoff: ${terminal.physical_duration_seconds.toFixed(2)} s / ${terminal.holds} admitted holds.`;
  $('life-scope').textContent=data.scope;
  $('life-raw').href=path || URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'}));$('life-raw').download='ORBEX-A-lifecycle.json';$('life-seek').max=samples.length-1;
  setup();paint();
  if (controlled) window.parent.postMessage({type:'orbexa.lifecycle.ready',session,count:samples.length,firstTime:samples[0].time,lastTime:samples.at(-1).time},location.origin);
  const actions=document.createElement('div');actions.className='replay-actions';const status=document.createElement('span');status.setAttribute('role','status');const button=document.createElement('button');button.textContent='Export mission video';button.disabled=!SceneVideoExporter.supported();actions.append(status,button);document.querySelector('main').append(actions);
  button.addEventListener('click',async()=>{if(exporting)return;stop();exporting=true;button.disabled=true;const canvas=document.createElement('canvas');canvas.width=1280;canvas.height=640;const ctx=canvas.getContext('2d');const adapter={renderer:{domElement:canvas},render(){ctx.fillStyle='#010306';ctx.fillRect(0,0,1280,640);views.forEach((v,i)=>{v.renderer.render(v.scene,v.camera);const source=v.renderer.domElement;const scale=Math.min(630/source.width,620/source.height);const w=source.width*scale,h=source.height*scale;ctx.drawImage(source,i*640+(640-w)/2,(640-h)/2,w,h);});}};try{const file=await new SceneVideoExporter().record(adapter,{elapsed_s:samples.map(sample=>sample.time-samples[0].time),phases:samples.map(sample=>sample.phase)},data.synthetic?'Synthetic post-capture physics fixture':'Attachment, detumbling and orbital delivery',{onFrame:i=>{index=i;paint();},onProgress:p=>status.textContent=`Exporting recorded lifecycle · ${p}%`});downloadVideo(file.blob,`ORBEX-A-physical-lifecycle.${file.extension}`);status.textContent='Mission video created. Download requested.';}catch(error){status.textContent=error.message;}finally{exporting=false;button.disabled=false;}});
  function animate(now){if(visible&&!exporting){if(playing&&!controlled){const time=startTime+(now-started)/1000*30;while(index<samples.length-1&&samples[index+1].time<=time)index++;paint();if(index===samples.length-1)stop();}for(const v of views){v.controls.update();v.renderer.render(v.scene,v.camera);}}requestAnimationFrame(animate);}requestAnimationFrame(animate);
}catch(error){$('life-error').hidden=false;$('life-error').textContent=error.message;for(const id of ['life-play','life-first','life-last','life-seek'])$(id).disabled=true;if(controlled)window.parent.postMessage({type:'orbexa.lifecycle.error',session,message:error.message},location.origin);}
