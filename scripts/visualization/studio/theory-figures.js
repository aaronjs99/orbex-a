/** Explanatory schematics. Dimensions are illustrative, not saved flight data. */
const shell=(name,body,width=820,height=180)=>`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${name}"><title>${name}</title>${body}</svg>`;
const label=(x,y,text,kind='')=>`<text x="${x}" y="${y}" class="${kind}">${text}</text>`;
const figures=new WeakMap();

export function renderTheoryFigure(container,title) {
  if (!figures.has(container)) {
    const state={title};
    state.observer=new ResizeObserver(()=>drawTheoryFigure(container,state.title));
    figures.set(container,state);
    state.observer.observe(container);
  }
  figures.get(container).title=title;
  drawTheoryFigure(container,title);
}

function drawTheoryFigure(container,title) {
  const solver=/solver/i.test(title);
  const held=/continuous|terminal tracking|hold|handoff/i.test(title);
  const geometry=/envisat|spacecraft|deployment|target/i.test(title);
  let name,body;
  if(solver){
    name='One saved nonlinear problem: residual before and after bound repair';
    const x=value=>132+(Math.log10(value)+14)*59;
    body=label(18,24,'Same saved eight-chaser NLP')+label(18,76,'Earlier backend')+label(18,125,'Hard bounds');
    for(const [v,y,text] of [[5.22233e-6,72,'≈ 5.22 × 10⁻⁶'],[2.08e-14,121,'≈ 2.08 × 10⁻¹⁴']]){
      body+=`<line x1="132" y1="${y}" x2="${x(v)}" y2="${y}" class="data"/><circle cx="${x(v)}" cy="${y}" r="4" class="point"/>`+label(x(v)+10,y+4,text);
    }
    body+=`<line x1="${x(1e-8)}" x2="${x(1e-8)}" y1="43" y2="140" class="threshold"/>`+label(x(1e-8)+8,43,'Acceptance 10⁻⁸');
    body+=label(132,165,'10⁻¹⁴')+label(660,165,'10⁻⁵  ·  residual (log scale)');
  }else if(geometry){
    name='Component geometry and selected structural patches';
    body=`<rect x="84" y="60" width="257" height="68" class="solid"/><rect x="341" y="85" width="62" height="12" class="solid"/><rect x="403" y="49" width="300" height="89" class="wing"/><rect x="108" y="36" width="177" height="14" class="solid"/>`;
    for(const x of [122,180,238,296])body+=`<circle cx="${x}" cy="128" r="4" class="point"/><circle cx="${x}" cy="154" r="9" class="bus"/><line x1="${x}" y1="145" x2="${x}" y2="132" class="contact"/>`;
    body+=label(84,25,'Bus & appendages')+label(478,25,'Solar wing')+label(430,165,'Every occupied component constrains every path');
  }else if(held){
    name='Checked nodes and continuous held-flow enclosure';
    body=`<path d="M55 116 Q170 15 300 86 T572 84 T770 87 L770 128 Q650 167 572 122 T300 124 Q170 65 55 153Z" class="band"/><path d="M55 135 Q170 40 300 105 T572 103 T770 107" class="data"/>`;
    for(const [x,y]of [[55,135],[300,105],[572,103],[770,107]])body+=`<rect x="${x-8}" y="${y-8}" width="16" height="16" class="node"/>`;
    body+=label(24,23,'Held feedback')+label(337,23,'Continuous flow enclosure')+label(627,163,'■ recorded nodes');
  }else{
    name='The COM tracks the reference inside its tube; the tip reaches the hull';
    body=`<rect x="44" y="43" width="190" height="116" class="solid"/><line x1="234" y1="30" x2="234" y2="164" class="surface"/><rect x="310" y="63" width="176" height="78" class="band"/><path d="M398 90 L410 102 L398 114 L386 102 Z" class="goal"/><circle cx="408" cy="107" r="14" class="bus"/><line x1="394" y1="107" x2="234" y2="107" class="contact"/>`;
    body+=label(44,25,'Target hull')+label(330,25,'Position tube')+label(525,75,'COM reference stays at the center')+label(525,108,'Tube fits the endpoint tolerance')+label(525,141,'Articulating tip closes the gap');
  }
  const width=container.clientWidth;
  if(width>0 && width<620){
    const diagramWidth=Math.max(280,width);
    if(solver){
      const x=value=>20+(Math.log10(value)+14)*(diagramWidth-60)/9;
      body=label(20,23,'Same saved eight-chaser NLP')+label(20,57,'Earlier backend')+label(20,123,'Hard bounds');
      for(const[v,y,text]of [[5.22233e-6,82,'≈ 5.22 × 10⁻⁶'],[2.08e-14,148,'≈ 2.08 × 10⁻¹⁴']]){
        body+=`<line x1="20" y1="${y}" x2="${x(v)}" y2="${y}" class="data"/><circle cx="${x(v)}" cy="${y}" r="4" class="point"/>`;
        body+=label(v>1e-8?x(v)-95:65,y-10,text);
      }
      body+=`<line x1="${x(1e-8)}" x2="${x(1e-8)}" y1="45" y2="164" class="threshold"/>`;
      body+=label(20,191,'Acceptance 10⁻⁸')+label(20,220,'Residual · log scale');
      container.innerHTML=shell(name,body,diagramWidth,237);
    }else if(geometry){
      const scale=(diagramWidth-30)/740;
      body=`<g transform="translate(0 30) scale(${scale})">${body.replace(/<text.*?<\/text>/g,'')}</g>`;
      body+=label(20,23,'Component geometry · schematic')+label(20,175,'Every occupied component')+label(20,197,'constrains every path');
      container.innerHTML=shell(name,body,diagramWidth,215);
    }else if(held){
      body=`<g transform="translate(0 30) scale(${(diagramWidth-15)/800})">${body.replace(/<text.*?<\/text>/g,'')}</g>`;
      body+=label(20,23,'Continuous flow · schematic')+label(20,145,'■ recorded nodes')+label(20,168,'Shaded region covers each hold');
      container.innerHTML=shell(name,body,diagramWidth,187);
    }else{
      const scale=(diagramWidth-30)/490;
      body=`<g transform="translate(0 12) scale(${scale})">${body.replace(/<text.*?<\/text>/g,'')}</g>`;
      body+=label(20,139,'COM reference stays at the center')+label(20,163,'Tube fits the endpoint tolerance')+label(20,187,'Articulating tip closes the gap');
      container.innerHTML=shell(name,body,diagramWidth,205);
    }
  }else container.innerHTML=shell(name,body);
}
