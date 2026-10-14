"""Responsive, offline session gallery for available saved mission families."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import quote, urlsplit


_TITLES = {
    "docking_body_closeup": "Docking in the target frame", "docking_errors": "Error to the rotating goal",
    "actual_nominal_trajectories": "Actual and planned paths", "active_target_margin": "Active target clearance",
    "minimum_pairwise_spacing": "Closest recorded approaches", "multi_chaser_spacing": "Chaser separation",
    "observable_drag": "Drag combination and uncertainty", "smid_point_errors": "Adaptation at the emitted point",
    "parameter_estimates_vs_truth": "Parameter beliefs and simulated truth", "smid_fss_widths": "Marginal feasible-set widths",
    "tube_geometry": "Predicted tube geometry", "control_effort": "Applied controls",
    "target_attitude": "Target orientation", "target_angular_velocity": "Target orientation rates",
    "rendezvous_margin": "Rendezvous exclusion sphere", "docking_cylinder_margin": "Docking cylinder clearance",
    "tightened_rendezvous_margin": "Tube-tightened rendezvous margin", "tightened_docking_cylinder_margin": "Tube-tightened docking margin",
}


class MissionGallery:
    """Create a gallery from family metadata and files already on disk.

    Each entry supplies ``mission``, ``approximation``, ``backend``, ``success``,
    ``sample_count``, and ``relative_dir``. ``samplecount`` and ``dirrelative`` are
    accepted aliases. Optional ``artifacts`` entries are paths or dictionaries
    with ``path`` and ``label``. When omitted, the saved family manifest is used;
    a directory scan is only the fallback for an older manifest-less family.
    Relative artifact paths use this gallery's output directory as their root;
    ``data_href`` is an optional relative link to the raw saved data.

    The caller discovers all retained families before a partial render. This
    class neither constructs scenarios nor modifies those family artifacts.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir).resolve()

    def _inside(self, path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    def _entry(self, entry: Mapping) -> dict:
        relative = str(entry.get("relative_dir", entry.get("dirrelative", ""))).replace("\\", "/")
        if not relative or Path(relative).is_absolute() or ":" in relative:
            raise ValueError("Gallery families require a relative directory inside the session output")
        family = self.output_dir / relative
        if not self._inside(family, self.output_dir):
            raise ValueError("Gallery family directory escapes the session output")
        manifest = {}
        manifest_path = family / "manifest.json"
        if manifest_path.is_file() and self._inside(manifest_path, family):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
        labels = dict(manifest.get("artifact_labels", {}))
        skipped = dict(entry.get("skipped_artifacts", manifest.get("skipped_artifacts", {})))
        notes = list(entry.get("notes", []))
        for insights in (family / "insights_manifest.json", family / "extras" / "insights_manifest.json"):
            if insights.is_file() and self._inside(insights, family):
                try:
                    extra = json.loads(insights.read_text(encoding="utf-8"))
                    if "skipped_artifacts" not in entry:
                        skipped.update(extra.get("skipped_artifacts", {}))
                    labels.update(extra.get("artifacts", {}))
                    notes.extend(extra.get("notes", []))
                except (OSError, ValueError):
                    pass
        artifacts = entry.get("artifacts")
        if artifacts is None:
            artifacts = manifest.get("artifacts")
        if artifacts is None:
            artifacts = sorted(path for path in family.rglob("*") if path.is_file()) if family.is_dir() else []
        files = []
        seen = set()
        for artifact in artifacts:
            label = artifact.get("label") if isinstance(artifact, Mapping) else None
            value = artifact.get("path", "") if isinstance(artifact, Mapping) else artifact
            path = Path(str(value))
            if not path.is_absolute():
                candidate = family / path
                path = candidate if candidate.is_file() else self.output_dir / path
            if not path.is_file() or not self._inside(path, family) or path.name in skipped:
                continue
            url = quote(path.resolve().relative_to(self.output_dir).as_posix(), safe="/")
            if url in seen:
                continue
            seen.add(url)
            stem = path.stem
            if "margin" in stem or "spacing" in stem:
                category = "Safety"
            elif any(word in stem for word in ("parameter", "smid", "drag")):
                category = "Adaptation"
            elif "control" in stem or "target" in stem:
                category = "Control & target"
            else:
                category = "Motion"
            files.append({"url": url, "name": path.name, "stem": stem, "suffix": path.suffix.lower(), "category": category,
                          "title": _TITLES.get(stem, stem.replace("_", " ").capitalize()), "label": label or labels.get(path.name, "Recorded simulation artifact.")})
        sample_count = entry.get("sample_count", entry.get("samplecount"))
        data_href = str(entry.get("data_href") or "").replace("\\", "/")
        if data_href and (urlsplit(data_href).scheme or data_href.startswith("//")):
            raise ValueError("Gallery data_href must be a relative local link")
        return {
            "title": str(entry.get("title", "")),
            "mission": str(entry.get("mission", manifest.get("mission", "Mission"))),
            "approximation": str(entry.get("approximation", manifest.get("approximation", "Unspecified"))),
            "backend": str(entry.get("backend", entry.get("solver_backend", manifest.get("solver_backend", "Unspecified")))),
            "success": entry.get("success"), "sample_count": None if sample_count is None else int(sample_count),
            "chaser_count": entry.get("chaser_count"), "mission_complete": entry.get("mission_complete"),
            "data_href": quote(data_href, safe="/%."),
            "message": str(entry.get("message", "")), "files": files,
            "skipped": skipped, "notes": list(dict.fromkeys(str(note) for note in notes)),
        }

    def render(self, entries: Iterable[Mapping]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        families = [self._entry(entry) for entry in entries]
        payload = json.dumps(families, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        page = _PAGE.replace("__FAMILIES__", payload)
        path = self.output_dir / "index.html"
        path.write_text(page, encoding="utf-8")
        return path


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ORBEX-A · Mission gallery</title><style>
:root{color-scheme:dark;--bg:#091321;--panel:#112136;--ink:#eaf1fa;--muted:#b6c6d8;--line:#2a4059;--accent:#88c8ff;--gold:#ebc67c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}main{max-width:1460px;margin:auto;padding:34px clamp(18px,4vw,56px) 64px}h1{font-size:clamp(34px,5vw,60px);font-weight:650;line-height:1.08;letter-spacing:-.035em;margin:12px 0 20px}h2{font-size:28px;font-weight:600;letter-spacing:-.02em;margin:38px 0 16px}h3{font-size:18px;margin:0 0 6px}.eyebrow{color:var(--accent);letter-spacing:.13em;font-size:12px;text-transform:uppercase}.sub{color:var(--muted);max-width:820px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin:30px 0}.metric{border-top:2px solid var(--line);padding-top:14px}.metric strong{display:block;font-size:28px;font-weight:600}.metric span,.note{color:var(--muted);font-size:14px}.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:18px 0}.toolbar label{color:var(--muted)}button,select{font:inherit;border:1px solid var(--line);border-radius:8px;padding:9px 15px;background:var(--panel);color:var(--ink);cursor:pointer}select{max-width:100%}button[aria-pressed=true]{background:var(--ink);border-color:var(--ink);color:var(--bg)}a{color:var(--accent);text-underline-offset:3px}button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid var(--accent);outline-offset:3px}[hidden]{display:none!important}.family-header{display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:20px}.family-header h2{margin:0}.badge{font-size:13px;padding:5px 10px;border:1px solid var(--line);border-radius:20px;color:var(--gold)}.badge.passed{color:#aed6ec;border-color:#4a778e}.status{color:var(--muted);margin:9px 0 0}.viewer{width:100%;height:720px;border:0;background:white;border-radius:12px}.viewer-links{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:14px}.animations,.plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.artifact{margin:0;background:var(--panel);border-radius:12px;overflow:hidden}.artifact img,.artifact video{width:100%;display:block;background:white}.artifact figcaption{padding:18px 20px}.artifact p{color:var(--muted);margin:5px 0 10px;font-size:14px}.artifact button{padding:0;border:0;border-radius:0;width:100%;display:block;background:white}.empty{color:var(--muted);padding:30px;border:1px dashed var(--line);border-radius:12px}details{border-top:1px solid var(--line);margin-top:24px;padding-top:16px}summary{cursor:pointer;color:var(--muted)}details li{margin:7px 0;color:var(--muted)}footer{margin-top:44px;border-top:1px solid var(--line);padding-top:20px;color:var(--muted);font-size:14px}dialog{max-width:96vw;max-height:96vh;padding:16px;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:12px}dialog::backdrop{background:#000c}dialog img{max-width:90vw;max-height:78vh;display:block;background:white}dialog .toolbar{justify-content:space-between;margin-top:0}.file-links{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}
@media(max-width:800px){.animations,.plots{grid-template-columns:1fr}.metrics{gap:12px}.metric strong{font-size:22px}.viewer{height:590px}main{padding-top:24px}}@media(max-width:480px){.metrics{grid-template-columns:1fr}.family-header{gap:8px}.toolbar select{width:100%}.viewer{height:520px}}
</style></head><body><main>
<header><div class="eyebrow">ORBEX-A · Saved simulation evidence</div><h1>Rendezvous & docking</h1><p class="sub">Explore recorded motion, inspect sampled safety margins, and follow adaptation in the controller's saved beliefs. Every available view belongs to the selected mission family.</p></header>
<div class="metrics"><div class="metric"><strong id="family-count">0</strong><span>Available mission families</span></div><div class="metric"><strong id="success-count">0</strong><span>Saved mission criteria passed</span></div><div class="metric"><strong id="sample-count">—</strong><span>Recorded time samples across these families</span></div></div>
<div class="toolbar"><label for="family">Mission family</label><select id="family" aria-label="Choose a saved mission family"></select></div>
<section id="selected"><div class="family-header"><h2 id="family-title"></h2><span id="status-badge" class="badge"></span></div><p id="family-status" class="status" aria-live="polite"></p><div id="family-links" class="file-links"></div>
<p id="raw-data" class="note" hidden><a id="raw-data-link" target="_blank" rel="noopener">Open saved raw mission data</a></p>
<section id="animation-section"><h2>Recorded approach</h2><div id="animations" class="animations"></div></section>
<section id="explorer-section"><h2>Explore in 3D</h2><div class="toolbar"><label for="view">Interactive view</label><select id="view"></select></div><iframe id="explorer" class="viewer" title="Interactive saved mission view" loading="lazy"></iframe><div class="viewer-links"><p>Drag to rotate · scroll to zoom · replay controls use recorded samples</p><p><a id="fullview" target="_blank" rel="noopener">Open full-screen view</a></p></div></section>
<section><h2>Evidence plots</h2><div class="toolbar" id="categories" aria-label="Filter plots"></div><div id="plots" class="plots"></div><p id="no-plots" class="empty" hidden>No supported plots are available in this family.</p></section>
<details id="missing-section"><summary>Unavailable views</summary><ul id="missing"></ul></details><details id="notes-section"><summary>Saved-data notes</summary><ul id="notes"></ul></details></section>
<p id="no-families" class="empty" hidden>No saved mission families were supplied. Render a saved family to populate this gallery.</p>
<footer>These are simulation samples. Physical margins and tube-tightened margins describe different quantities; neither sampled plots nor local controller approximations establish hardware readiness or a general nonlinear safety proof. Velocity states use true anomaly rather than seconds.</footer>
</main><dialog id="lightbox"><div class="toolbar"><strong id="lightbox-title"></strong><button type="button" id="close-dialog">Close</button></div><img id="lightbox-img" alt=""><p><a id="lightbox-link" target="_blank" rel="noopener">Open full-resolution image</a></p></dialog>
<script>
'use strict';
const families=__FAMILIES__;
const el=id=>document.getElementById(id);
let selected=0,category='All',currentView='';
function link(url,text){const a=document.createElement('a');a.href=url;a.textContent=text;a.target='_blank';a.rel='noopener';return a}
function familyLabel(f){const title=value=>value.charAt(0).toUpperCase()+value.slice(1);const mission=f.chaser_count===1?'Single chaser':f.chaser_count>1?`${f.chaser_count} chasers`:title(f.mission);return f.title||`${mission} · ${title(f.approximation)} · ${f.backend.toUpperCase()}`}
function openImage(file){el('lightbox-title').textContent=file.title;el('lightbox-img').src=file.url;el('lightbox-img').alt=file.title;el('lightbox-link').href=file.url;el('lightbox').showModal()}
function imageCard(file,animated=false){const figure=document.createElement('figure');figure.className='artifact';const img=document.createElement(file.suffix==='.mp4'?'video':'img');img.src=file.url;if(img.tagName==='VIDEO'){img.controls=true;img.preload='metadata'}else{img.alt=file.title;img.loading='lazy'}if(animated){figure.append(img)}else{const button=document.createElement('button');button.type='button';button.setAttribute('aria-label',`Enlarge ${file.title}`);button.append(img);button.addEventListener('click',()=>openImage(file));figure.append(button)}const caption=document.createElement('figcaption'),h=document.createElement('h3'),p=document.createElement('p');h.textContent=file.title;p.textContent=file.label;caption.append(h,p,link(file.url,animated?'Open animation':'Open full size'));figure.append(caption);return figure}
function renderPlots(){const files=families[selected].files.filter(f=>['.png','.jpg','.jpeg','.webp','.svg'].includes(f.suffix));el('plots').replaceChildren();for(const file of files){if(category!=='All'&&file.category!==category)continue;el('plots').append(imageCard(file))}el('no-plots').hidden=el('plots').children.length>0}
function renderView(){const option=el('view').selectedOptions[0];if(!option)return;currentView=option.value;el('explorer').src=option.value;el('fullview').href=option.value;el('explorer').title=`${familyLabel(families[selected])}: ${option.textContent}`}
function listNotes(id,section,values){el(id).replaceChildren();for(const value of values){const item=document.createElement('li');item.textContent=value;el(id).append(item)}el(section).hidden=values.length===0}
function renderFamily(){const f=families[selected];el('raw-data').hidden=!f.data_href;if(f.data_href)el('raw-data-link').href=f.data_href;el('family-title').textContent=familyLabel(f);const passed=f.success===true;el('status-badge').textContent=passed?'Saved criteria passed':f.success===false?'Incomplete or criteria not met':'Outcome unavailable';el('status-badge').classList.toggle('passed',passed);el('family-status').textContent=`${f.sample_count===null?'Sample count unavailable':f.sample_count+' recorded states'}`+(f.message?' · '+f.message:'');el('family-links').replaceChildren();for(const [name,title] of [['index.html','Mission report'],['manifest.json','Artifact manifest'],['insights_manifest.json','Insight provenance'],['mission_data.json','Raw mission data']]){const file=f.files.find(item=>item.name===name);if(file)el('family-links').append(link(file.url,title))}const gifs=f.files.filter(file=>file.suffix==='.gif');const animations=gifs.length?gifs:f.files.filter(file=>file.suffix==='.mp4');el('animations').replaceChildren();for(const file of animations)el('animations').append(imageCard({...file,title:file.stem.includes('closeup')?'Docking in the target frame':'Recorded mission approach'},true));el('animation-section').hidden=animations.length===0;const labels={replay:'Animated recorded replay',trajectory:'Actual and planned paths',tube_trajectory:'Tubes and prediction horizons',diagnostics:'Interactive diagnostics'};const views=f.files.filter(file=>file.suffix==='.html'&&Object.keys(labels).some(name=>file.stem===name||name==='replay'&&file.stem.endsWith('_replay')));views.sort((a,b)=>(a.stem.includes('replay')?-1:0)-(b.stem.includes('replay')?-1:0));el('view').replaceChildren();for(const file of views){const option=document.createElement('option');option.value=file.url;option.textContent=labels[file.stem]||labels.replay;el('view').append(option)}el('explorer-section').hidden=views.length===0;if(views.length){if(views.some(file=>file.url===currentView))el('view').value=currentView;renderView()}else{el('explorer').removeAttribute('src')}el('categories').replaceChildren();const categories=['All',...new Set(f.files.filter(file=>['.png','.jpg','.jpeg','.webp','.svg'].includes(file.suffix)).map(file=>file.category))];if(!categories.includes(category))category='All';for(const name of categories){const button=document.createElement('button');button.type='button';button.textContent=name;button.dataset.category=name;button.setAttribute('aria-pressed',String(name===category));el('categories').append(button)}renderPlots();listNotes('missing','missing-section',Object.entries(f.skipped).map(([name,reason])=>`${name}: ${reason}`));listNotes('notes','notes-section',f.notes)}
el('family-count').textContent=String(families.length);const known=families.filter(f=>typeof f.success==='boolean');el('success-count').textContent=`${families.filter(f=>f.success===true).length} / ${known.length}`;const counts=families.map(f=>f.sample_count);el('sample-count').textContent=counts.length&&counts.every(n=>n!==null)?counts.reduce((a,b)=>a+b,0).toLocaleString():'Unavailable';for(const [index,family] of families.entries()){const option=document.createElement('option');option.value=String(index);option.textContent=familyLabel(family);el('family').append(option)}el('family').disabled=families.length<2;el('selected').hidden=families.length===0;el('no-families').hidden=families.length>0;if(families.length)renderFamily();el('family').addEventListener('change',()=>{selected=Number(el('family').value);renderFamily()});el('view').addEventListener('change',renderView);el('categories').addEventListener('click',event=>{const button=event.target.closest('button[data-category]');if(!button)return;category=button.dataset.category;el('categories').querySelectorAll('button').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));renderPlots()});el('close-dialog').addEventListener('click',()=>el('lightbox').close());el('lightbox').addEventListener('click',event=>{if(event.target===el('lightbox'))el('lightbox').close()});
</script></body></html>'''
