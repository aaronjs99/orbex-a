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
  if(/sensed.*tube|tube.*sensing/i.test(title)){
    const width=Math.max(280,container.clientWidth||820),small=width<620;
    const plotWidth=small?width-68:Math.min(350,width*.48),left=48,top=15,plotHeight=190;
    const x=v=>left+(v+1.2)/2.4*plotWidth,y=v=>top+(1.65-v)/3.3*plotHeight;
    const vertices=[[-1,.5],[-1,1.5],[1,-.5],[1,-1.5]];
    let body=`<path d="M${x(-1.2)} ${y(0)}H${x(1.2)} M${x(0)} ${y(-1.65)}V${y(1.65)}" class="surface"/>`;
    body+=`<polygon points="${vertices.map(v=>`${x(v[0])},${y(v[1])}`).join(' ')}" class="band"/>`;
    body+=`<path d="M${x(-1)} ${y(1)}L${x(1)} ${y(-1)}" class="data"/>`;
    body+=`<rect x="${x(-.1)}" y="${y(.2)}" width="${x(.5)-x(-.1)}" height="${y(-.6)-y(.2)}" fill="none" class="threshold"/>`;
    body+=`<circle cx="${x(.3)}" cy="${y(-.3)}" r="4" class="point"/>`;
    body+=label(left,top+plotHeight+24,'Position error / Ω')+
      `<text x="17" y="${top+plotHeight/2}" text-anchor="middle" transform="rotate(-90 17 ${top+plotHeight/2})">Velocity error / ΛΩ</text>`;
    const tx=small?left:plotWidth+left+35,ty=small?255:60;
    body+=label(tx,ty,'Relational tube · |eᵥ + Λeₚ| ≤ Φ')+
      label(tx,ty+24,'Dashed: navigation rectangle')+label(tx,ty+48,'Corners may lie outside the tube')+
      label(tx,ty+72,'The physical state belongs to both');
    const height=small?350:250;
    container.innerHTML=shell('Illustrative normalized position–velocity tube and navigation sets',body,width,height)
      .replace('<svg ','<svg style="max-height:none" ');return;
  }
  const pipeline=/nonlinear|navigation|sensing|coordinates/i.test(title);
  const backup=/fallback|refuge|continuation/i.test(title);
  const contact=/contact|attachment|disposal/i.test(title);
  const comparison=/study|comparison|experiment/i.test(title);
  if(pipeline||backup||contact||comparison){
    const steps=comparison?['Independent declared cases','Same priors, vehicle and costs','Adaptive / frozen × dynamic / fixed','Every outcome and deadline','Paired metrics + source hashes']:
      contact?['Exterior COM tube','Finite tip extension','Compliant contact','Settled finite bonds','Detumble + orbital burn']:
      backup?['Stored fleet refuge ready','ADTMPC candidate','Nonlinear sweep + successor + fuel','Admit candidate or execute refuge','Archive actual applied hold']:
      ['Nonlinear spacecraft plant','Delayed bounded sensor packets','Navigation sets + parameter membership','Nominal + tube + bandwidth NLP','Independent whole-fleet admission'];
    const width=Math.max(280,container.clientWidth||820),small=width<620;
    const cols=small?1:width<900?2:3,boxWidth=(width-16*(cols+1))/cols,boxHeight=38;
    const rows=Math.ceil(steps.length/cols),height=24+rows*62;
    let body='';
    steps.forEach((text,i)=>{
      const row=Math.floor(i/cols),col=row%2?cols-1-i%cols:i%cols,x=16+col*(boxWidth+16),y=10+row*62;
      body+=`<rect x="${x}" y="${y}" width="${boxWidth}" height="${boxHeight}" rx="6" class="band"/>`;
      const words=text.split(' '),mid=Math.ceil(words.length/2);
      if(text.length>30){
        body+=label(x+12,y+15,words.slice(0,mid).join(' '))+label(x+12,y+30,words.slice(mid).join(' '));
      }else body+=label(x+12,y+24,text);
      if(i<steps.length-1){
        const nextRow=Math.floor((i+1)/cols);
        if(nextRow!==row)body+=label(x+boxWidth/2-5,y+54,'↓');
        else body+=label(row%2?x-14:x+boxWidth+3,y+24,row%2?'←':'→');
      }
    });
    container.innerHTML=shell('Schematic · '+title,body,width,height);return;
  }
  const solver=/solver/i.test(title);
  const held=/continuous|terminal tracking|hold|handoff/i.test(title);
  const geometry=/envisat|spacecraft|deployment|target/i.test(title);
  let name,body;
  if(solver){
    name='One reconstructed chaser MPC subproblem: residual before and after bound repair';
    const x=value=>132+(Math.log10(value)+14)*59;
    body=label(18,24,'One reconstructed chaser MPC')+label(18,76,'Earlier backend')+label(18,125,'Hard bounds');
    for(const [v,y,text] of [[5.22233e-6,72,'≈ 5.22 × 10⁻⁶'],[2.08e-14,121,'≈ 2.08 × 10⁻¹⁴']]){
      body+=`<line x1="132" y1="${y}" x2="${x(v)}" y2="${y}" class="data"/><circle cx="${x(v)}" cy="${y}" r="4" class="point"/>`+label(x(v)+10,y+4,text);
    }
    body+=`<line x1="${x(1e-8)}" x2="${x(1e-8)}" y1="43" y2="140" class="threshold"/>`+label(x(1e-8)+8,43,'Acceptance 10⁻⁸');
    body+=label(132,165,'10⁻¹⁴')+label(660,165,'10⁻⁵  ·  residual (log scale)');
  }else if(geometry){
    name='Schematic of declared component geometry and nominal structural patches';
    body=`<rect x="84" y="60" width="257" height="68" class="solid"/><rect x="341" y="85" width="62" height="12" class="solid"/><rect x="403" y="49" width="300" height="89" class="wing"/><rect x="108" y="36" width="177" height="14" class="solid"/>`;
    for(const x of [122,180,238,296])body+=`<circle cx="${x}" cy="128" r="4" class="point"/><circle cx="${x}" cy="154" r="9" class="bus"/><line x1="${x}" y1="145" x2="${x}" y2="132" class="contact"/>`;
    body+=label(84,25,'Bus & appendages · schematic')+label(478,25,'Solar wing')+label(430,165,'All declared parts enter node and swept checks');
  }else if(held){
    name='Schematic of checked nodes and an admitted continuous held-flow enclosure';
    body=`<path d="M55 116 Q170 15 300 86 T572 84 T770 87 L770 128 Q650 167 572 122 T300 124 Q170 65 55 153Z" class="band"/><path d="M55 135 Q170 40 300 105 T572 103 T770 107" class="data"/>`;
    for(const [x,y]of [[55,135],[300,105],[572,103],[770,107]])body+=`<rect x="${x-8}" y="${y-8}" width="16" height="16" class="node"/>`;
    body+=label(24,23,'Held feedback')+label(337,23,'Enclosure under declared bounds')+label(627,163,'■ checked nodes');
  }else{
    name='Reference-centered terminal tube: the COM tracks an exterior goal and a carried tip reaches the nominal hull';
    body=`<rect x="44" y="43" width="190" height="116" class="solid"/><line x1="234" y1="30" x2="234" y2="164" class="surface"/><rect x="310" y="63" width="176" height="78" class="band"/><path d="M398 90 L410 102 L398 114 L386 102 Z" class="goal"/><circle cx="408" cy="107" r="14" class="bus"/><line x1="394" y1="107" x2="234" y2="107" class="contact"/>`;
    body+=label(44,25,'Target hull')+label(330,25,'Terminal tube')+label(525,75,'Terminal reference at tube center')+label(525,108,'Admitted tube meets pose tolerance')+label(525,141,'Carried tip closes nominal gap');
  }
  const width=container.clientWidth;
  if(width>0 && width<620){
    const diagramWidth=Math.max(280,width);
    if(solver){
      const x=value=>20+(Math.log10(value)+14)*(diagramWidth-60)/9;
      body=label(20,23,'One reconstructed chaser MPC')+label(20,57,'Earlier backend')+label(20,123,'Hard bounds');
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
      body+=label(20,23,'Component geometry · schematic')+label(20,175,'All declared components')+label(20,197,'enter node and swept checks');
      container.innerHTML=shell(name,body,diagramWidth,215);
    }else if(held){
      body=`<g transform="translate(0 30) scale(${(diagramWidth-15)/800})">${body.replace(/<text.*?<\/text>/g,'')}</g>`;
      body+=label(20,23,'Continuous flow · schematic')+label(20,145,'■ checked nodes')+label(20,168,'Admitted hold enclosure');
      container.innerHTML=shell(name,body,diagramWidth,187);
    }else{
      const scale=(diagramWidth-30)/490;
      body=`<g transform="translate(0 12) scale(${scale})">${body.replace(/<text.*?<\/text>/g,'')}</g>`;
      body+=label(20,139,'Terminal reference at tube center')+label(20,163,'Admitted tube meets pose tolerance')+label(20,187,'Carried tip closes nominal gap');
      container.innerHTML=shell(name,body,diagramWidth,205);
    }
  }else container.innerHTML=shell(name,body);
}
