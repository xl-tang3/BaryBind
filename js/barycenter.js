/* Deterministic scientific illustrations; no model inference or experimental embeddings. */
(() => {
'use strict';
const NS='http://www.w3.org/2000/svg';
const colors=['#86b5bb','#a0b79a','#c5a397','#aea5c2','#8ca6c3'];
const names=['TEXT','VIDEO','AUDIO','SUBTITLE','DEPTH'];
const centers=[[165,120],[470,142],[112,333],[326,453],[520,352]];
const warm='#e5d9b6';
function el(tag,attrs={},text){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
function add(p,tag,a,t){const n=el(tag,a,t);p.appendChild(n);return n;}
function point(k,j){const a=j*2.3999632297+k*.63,r=Math.sqrt((j+.5)/44);return [centers[k][0]+Math.cos(a)*r*(k%2?49:56),centers[k][1]+Math.sin(a)*r*(k%2?34:26)];}
function setup(svg,mode='normal'){
 const id=svg.id,defs=add(svg,'defs');let g=add(defs,'radialGradient',{id:id+'-glow'});add(g,'stop',{offset:'0%','stop-color':warm,'stop-opacity':'.13'});add(g,'stop',{offset:'100%','stop-color':warm,'stop-opacity':'0'});
 let pattern=add(defs,'pattern',{id:id+'-grid',width:40,height:40,patternUnits:'userSpaceOnUse'});add(pattern,'path',{d:'M40 0H0V40',fill:'none',stroke:'#3c494e','stroke-width':'.6',opacity:'.23'});
 add(svg,'rect',{x:25,y:20,width:590,height:510,fill:`url(#${id}-grid)`});add(svg,'ellipse',{cx:320,cy:282,rx:250,ry:230,fill:`url(#${id}-glow)`});
 const orbit=add(svg,'g',{class:mode==='hero'?'orbital':'',opacity:'.65'});
 [170,226].forEach((r,i)=>add(orbit,'ellipse',{cx:320,cy:284,rx:r,ry:r*.74,fill:'none',stroke:'#435155','stroke-width':'.7','stroke-dasharray':i?'2 7':'none',transform:`rotate(${i?32:-22} 320 284)`}));
 add(svg,'path',{d:'M300 284h40M320 264v40',stroke:'#56625d','stroke-width':'.6'});
 const paths=add(svg,'g'),clouds=add(svg,'g');
 centers.forEach((c,k)=>{
  const group=add(clouds,'g',{'data-modality':names[k]});
  add(group,'ellipse',{cx:c[0],cy:c[1],rx:65,ry:38,fill:colors[k],'fill-opacity':'.035',stroke:colors[k],'stroke-opacity':'.13',transform:`rotate(${k*22-30} ${c[0]} ${c[1]})`});
  for(let j=0;j<44;j++){const p=point(k,j);add(group,'circle',{cx:p[0],cy:p[1],r:j%7===0?2.4:1.5,fill:colors[k],opacity:.35+(j%5)*.12});}
  add(group,'text',{x:c[0],y:c[1]-55,fill:colors[k],'text-anchor':'middle',class:'plot-label'},names[k]);
  add(group,'text',{x:c[0],y:c[1]+58,fill:colors[k],opacity:'.65','text-anchor':'middle',class:'plot-small'},`P${'₁₂₃₄₅'[k]}`);
 });
 const wb=add(svg,'g'); const halo=add(wb,'circle',{cx:320,cy:280,r:52,fill:`url(#${id}-glow)`});
 const hull=add(wb,'path',{fill:warm,'fill-opacity':'.025',stroke:warm,'stroke-opacity':'.28','stroke-width':1});
 const dots=[];for(let j=0;j<32;j++)dots.push(add(wb,'circle',{r:j%8===0?2.4:1.8,fill:warm,opacity:.45+(j%4)*.15}));
 const ring=add(wb,'circle',{cx:320,cy:280,r:30,fill:'none',stroke:warm,'stroke-opacity':'.32','stroke-dasharray':'2 5'});
 const label=add(wb,'text',{x:320,y:287,'text-anchor':'middle',class:'plot-center',fill:warm},'WB');
 const under=add(wb,'text',{x:320,y:335,'text-anchor':'middle',class:'plot-small',fill:warm},'SHARED ANCHOR Q');
 const trajectories=[];
 centers.forEach((c,k)=>{for(let j=0;j<4;j++){trajectories.push({k,j,path:add(paths,'path',{fill:'none',stroke:colors[k],'stroke-width':'.8',opacity:.17+j*.04,class:'transport-path'})});}});
 return {svg,paths,clouds,wb,halo,hull,dots,ring,label,under,trajectories};
}
function weightedMedian(points,w){let x=points.reduce((s,p,k)=>s+p[0]*w[k],0),y=points.reduce((s,p,k)=>s+p[1]*w[k],0);for(let t=0;t<65;t++){let sx=0,sy=0,d=0;points.forEach((p,k)=>{const a=w[k]/Math.max(.03,Math.hypot(x-p[0],y-p[1]));sx+=a*p[0];sy+=a*p[1];d+=a;});if(!d)break;const nx=sx/d,ny=sy/d;if(Math.hypot(nx-x,ny-y)<.001){x=nx;y=ny;break;}x=nx;y=ny;}return[x,y];}
function draw(v,w,forced){let center=forced||weightedMedian(centers,w);let pts=[];v.dots.forEach((dot,j)=>{let q;if(forced){const a=j*2.39996,r=34*Math.sqrt((j+.5)/32);q=[center[0]+Math.cos(a)*r,center[1]+Math.sin(a)*r*.65];}else q=weightedMedian(centers.map((_,k)=>point(k,j)),w);pts.push(q);dot.setAttribute('cx',q[0]);dot.setAttribute('cy',q[1]);});const[cx,cy]=center;
 for(const n of[v.halo,v.ring]){n.setAttribute('cx',cx);n.setAttribute('cy',cy);}v.label.setAttribute('x',cx);v.label.setAttribute('y',cy+8);v.under.setAttribute('x',cx);v.under.setAttribute('y',cy+65);
 const vertices=[[-43,-10],[0,-40],[45,5],[22,40],[-30,31]].map(p=>[cx+p[0],cy+p[1]]);v.hull.setAttribute('d','M'+vertices.map(p=>p.join(',')).join('L')+'Z');
 v.trajectories.forEach(({path,k,j})=>{let p=point(k,j*9);path.setAttribute('d',`M${p[0]},${p[1]} Q${(p[0]+cx)/2+(j-1.5)*24},${(p[1]+cy)/2-25} ${cx+(j-1.5)*9},${cy+(j-1.5)*5}`);path.setAttribute('opacity',w[k]===0?.025:.13+w[k]*.48);});
 [...v.clouds.children].forEach((n,k)=>n.style.opacity=w[k]===0?.23:1);
 return pts;
}
const weights=[.2,.2,.2,.2,.2];
const hero=setup(document.getElementById('hero-plot'),'hero');draw(hero,weights,[320,282]);
const comparison=setup(document.getElementById('compare-plot'));comparison.svg.setAttribute('viewBox','0 40 640 470');draw(comparison,weights,centers[0]);comparison.label.textContent='T';comparison.under.textContent='TEXT ANCHOR';
let compareFrame;
function transition(v,to,from,label,under){cancelAnimationFrame(compareFrame);if(matchMedia('(prefers-reduced-motion: reduce)').matches){draw(v,weights,to);v.label.textContent=label;v.under.textContent=under;return;}const start=performance.now();function tick(now){const t=Math.min(1,(now-start)/650),e=1-Math.pow(1-t,3);draw(v,weights,[from[0]+(to[0]-from[0])*e,from[1]+(to[1]-from[1])*e]);v.label.textContent=label;v.under.textContent=under;if(t<1)compareFrame=requestAnimationFrame(tick);}requestAnimationFrame(tick);}
let compareMode='anchor';document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>{const wb=button.dataset.mode==='wb';if(button.dataset.mode===compareMode)return;document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));document.getElementById('compare-title').textContent=wb?'The center is learned.':'The center is chosen.';document.getElementById('compare-description').textContent=wb?'The WB is jointly optimized across modality distributions. It becomes the shared geometric reference for volumetric alignment.':'A modality-specific anchor defines the reference. Other modalities align toward that designated modality.';document.getElementById('compare-label').textContent=wb?'JOINTLY OPTIMIZED WB ANCHOR':'MODALITY-SPECIFIC ANCHOR';transition(comparison,wb?[320,282]:centers[0],wb?centers[0]:[320,282],wb?'WB':'T',wb?'SHARED ANCHOR Q':'TEXT ANCHOR');compareMode=button.dataset.mode;}));
const method=setup(document.getElementById('method-plot'));method.svg.setAttribute('viewBox','0 20 640 510');draw(method,weights,[320,280]);
const explorer=setup(document.getElementById('explorer-plot'));let raw=[20,20,20,20,20];const control=document.getElementById('weights');names.forEach((name,k)=>{const div=document.createElement('div');div.className='weight-control';div.style.setProperty('--mod',colors[k]);div.innerHTML=`<label for="weight-${k}"><i aria-hidden="true"></i>${name.charAt(0)+name.slice(1).toLowerCase()} <span>λ<sub>${name.toLowerCase()}</sub></span><output id="weight-value-${k}" for="weight-${k}">0.20</output></label><input id="weight-${k}" aria-label="${name.toLowerCase()} modality contribution" type="range" min="0" max="100" step="1" value="20">`;control.appendChild(div);div.querySelector('input').addEventListener('input',e=>{raw[k]=Number(e.target.value);if(raw.every(v=>v===0)){raw[k]=1;e.target.value='1';}update();});});
function update(){const sum=raw.reduce((a,b)=>a+b,0),w=raw.map(v=>v/sum),pts=draw(explorer,w);let energy=0;pts.forEach((q,j)=>centers.forEach((_,k)=>{const p=point(k,j);energy+=w[k]*Math.hypot(p[0]-q[0],p[1]-q[1])/pts.length/200;}));const entropy=-w.reduce((s,v)=>s+(v?v*Math.log(v):0),0)/Math.log(5);w.forEach((v,k)=>{document.getElementById('weight-value-'+k).textContent=v.toFixed(2);document.getElementById('weight-'+k).setAttribute('aria-valuetext',`${Math.round(v*100)} percent normalized weight`);});document.getElementById('energy').textContent=energy.toFixed(3);document.getElementById('balance').textContent=Math.round(entropy*100)+'%';document.getElementById('active').textContent=w.filter(v=>v>0).length+' / 5';}
document.getElementById('reset').addEventListener('click',()=>{raw=[20,20,20,20,20];raw.forEach((v,k)=>document.getElementById('weight-'+k).value=v);update();});update();
const motion=document.getElementById('motion-toggle');let paused=matchMedia('(prefers-reduced-motion: reduce)').matches;function setMotion(){document.body.classList.toggle('motion-paused',paused);motion.setAttribute('aria-pressed',String(paused));motion.textContent=paused?'Resume motion':'Pause motion';}motion.addEventListener('click',()=>{paused=!paused;setMotion();});setMotion();
window.BaryGraphics={el,add,colors,warm};
})();
