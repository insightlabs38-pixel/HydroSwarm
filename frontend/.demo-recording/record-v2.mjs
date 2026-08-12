/* Local-only v2 editorial-pacing recorder. No request routing/interception. */
import { chromium } from 'playwright';
import { mkdir, rename, stat, writeFile, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..', '..');
const output = path.join(root, 'demo-recordings', 'v2');
const raw = path.join(output, 'raw');
const qc = path.join(output, 'qc');
const manifest = path.join(output, 'manifest');
const ffmpeg = '/root/.cache/ms-playwright/ffmpeg-1011/ffmpeg-linux';
const clips = process.env.CLIPS ? new Set(process.env.CLIPS.split(',').filter(Boolean)) : null;
const captures = [];
for (const dir of [raw, qc, manifest]) await mkdir(dir, { recursive: true });

function exec(args, label) {
  const result = spawnSync(ffmpeg, ['-hide_banner', '-y', ...args], { encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`${label}: ${result.stderr.slice(-1200)}`);
  return `${result.stdout}\n${result.stderr}`;
}
function videoInfo(file) {
  const result = spawnSync(ffmpeg, ['-hide_banner', '-i', file], { encoding: 'utf8' });
  const text = `${result.stdout}\n${result.stderr}`;
  const duration = text.match(/Duration: (\d\d):(\d\d):(\d\d\.\d+)/);
  const video = text.match(/(\d{3,5})x(\d{3,5}).*?(\d+(?:\.\d+)?) fps/);
  if (!duration || !video) throw new Error(`unreadable video ${file}`);
  return { seconds: Number(duration[1])*3600 + Number(duration[2])*60 + Number(duration[3]), width:Number(video[1]), height:Number(video[2]), fps:Number(video[3]) };
}
function sha(file) {
  const result=spawnSync('sha256sum',[file],{encoding:'utf8'}); if(result.status!==0) throw new Error(result.stderr); return result.stdout.split(/\s+/)[0];
}
async function installCursor(page) {
  await page.addStyleTag({ content:`#hs-demo-cursor{position:fixed;left:-100px;top:-100px;width:26px;height:26px;border-radius:50%;pointer-events:none;z-index:2147483647;opacity:0;transform:translate(-50%,-50%);background:rgba(238,244,247,.72);border:1px solid rgba(21,31,37,.78);box-shadow:0 2px 8px rgba(0,0,0,.46),inset 0 0 0 1px rgba(255,255,255,.32);transition:opacity 160ms ease,transform 100ms ease}#hs-demo-cursor.click{transform:translate(-50%,-50%) scale(.85)}#hs-demo-cursor:after{content:'';position:absolute;inset:-5px;border:1px solid rgba(230,240,243,.55);border-radius:50%;opacity:0}#hs-demo-cursor.click:after{animation:ring 190ms ease-out}@keyframes ring{from{transform:scale(.72);opacity:.68}to{transform:scale(1.38);opacity:0}}` });
  await page.evaluate(() => { const el=document.createElement('div'); el.id='hs-demo-cursor'; el.dataset.x='-100'; el.dataset.y='-100'; document.body.append(el); });
}
async function hold(page, ms) { await page.waitForTimeout(ms); }
async function move(page, locator) {
  const box=await locator.boundingBox(); if(!box) throw new Error('cursor target has no box'); const x=box.x+box.width/2, y=box.y+box.height/2;
  await page.evaluate(async ({x,y}) => { const el=document.querySelector('#hs-demo-cursor'); const sx=Number(el?.dataset.x||-100),sy=Number(el?.dataset.y||-100); if(!el)return; el.style.opacity='1'; const distance=Math.hypot(x-sx,y-sy),duration=Math.min(650,Math.max(320,300+distance*.13)),start=performance.now(); await new Promise(done=>{const frame=now=>{const p=Math.min(1,(now-start)/duration),e=p<.5?4*p*p*p:1-Math.pow(-2*p+2,3)/2;el.style.left=`${sx+(x-sx)*e}px`;el.style.top=`${sy+(y-sy)*e}px`;if(p<1)requestAnimationFrame(frame);else done()};requestAnimationFrame(frame)});el.dataset.x=String(x);el.dataset.y=String(y); },{x,y});
}
async function hide(page) { await page.evaluate(()=>{const el=document.querySelector('#hs-demo-cursor');if(el)el.style.opacity='0'}); }
async function click(page, locator) { await move(page,locator); await hold(page,750); await page.evaluate(()=>document.querySelector('#hs-demo-cursor')?.classList.add('click')); await locator.click(); await hold(page,140); await page.evaluate(()=>document.querySelector('#hs-demo-cursor')?.classList.remove('click')); }
async function visible(locator, name) { await locator.waitFor({state:'visible',timeout:45000}); if(!await locator.isVisible()) throw new Error(`${name} not visible`); }
async function noOverflow(page) { const d=await page.evaluate(()=>({w:document.documentElement.clientWidth,s:document.documentElement.scrollWidth})); if(d.s>d.w)throw new Error(`horizontal overflow ${d.s}>${d.w}`); }
async function reference(page) { await page.goto('/'); const start=page.getByRole('button',{name:'Run Reference Incident'}); await visible(start,'gateway'); await start.click(); await visible(page.getByText('REFERENCE INCIDENT · VERIFIED REPLAY'),'reference label'); await visible(page.locator('.map-canvas[role="img"]'),'map'); await page.getByRole('button',{name:'Pause',exact:true}).click(); }
async function milestone(page,n) { for(let i=0;i<n;i+=1){const next=page.getByRole('button',{name:'Next',exact:true});if(await next.isEnabled())await next.click();else await page.locator('.mode-banner-controls button').filter({hasText:/^Replay /}).click();await hold(page,180);} }
async function rail(page, label, heading) { const button=page.getByRole('button',{name:new RegExp(`^${label}:`)}); await button.click(); await visible(page.getByRole('heading',{name:heading}),heading); await hold(page,700); return button; }
async function capture(name, metadata, choreography) {
  if(clips && !clips.has(name)) return;
  const target=path.join(raw,`${name}.webm`); if(existsSync(target)) throw new Error(`${target} already exists; refusing to replace capture`);
  const context=await browser.newContext({baseURL:'http://127.0.0.1:8765',viewport:{width:1920,height:1080},deviceScaleFactor:1,recordVideo:{dir:raw,size:{width:1920,height:1080}}}); const page=await context.newPage(); await installCursor(page); let usableAt=Date.now(); const proof=[];
  const mark=()=>{usableAt=Date.now()}; const proofShot=async label=>{const file=path.join(qc,`${name}-${label}.png`);await page.screenshot({path:file});proof.push(label)};
  try { await choreography(page,mark,proofShot); await noOverflow(page); await hold(page,1100); } catch(error) { await context.close(); throw error; }
  const video=page.video(); await context.close(); if(!video)throw new Error(`${name}: no video`); await rename(await video.path(),target); const detail=videoInfo(target); if(detail.width!==1920||detail.height!==1080||detail.fps!==25)throw new Error(`${name}: ${detail.width}x${detail.height}@${detail.fps}`); if((await stat(target)).size<10000)throw new Error(`${name}: empty`);
  // Decode a terminal frame and the QC frames; this bundled ffmpeg does not
  // provide the usual null/rawvideo muxers, but it does fully decode VP8 into
  // PNG for a representative end-of-file playback check.
  exec(['-ss',Math.max(.01,detail.seconds-.04).toFixed(3),'-i',target,'-frames:v','1',path.join(qc,`${name}-decode-terminal.png`)],`${name} terminal decode`);
  const startSeconds=(usableAt - captureStarted.get(name))/1000;
  const frameTimes=[Math.min(Math.max(.25,startSeconds+.25),detail.seconds-.25),Math.max(startSeconds+.25,(startSeconds+detail.seconds)/2),Math.max(startSeconds+.25,detail.seconds-1)];
  for(const [label,time] of [['first',frameTimes[0]],['mid',frameTimes[1]],['final',frameTimes[2]]]) exec(['-ss',Number(time).toFixed(3),'-i',target,'-frames:v','1',path.join(qc,`${name}-${label}.png`)],`${name} ${label} qc`);
  captures.push({...metadata,name,file:path.relative(root,target),duration:detail.seconds.toFixed(2),usableIn:startSeconds.toFixed(2),usableDuration:Math.max(0,detail.seconds-startSeconds).toFixed(2),resolution:`${detail.width}x${detail.height}`,fps:detail.fps,sha256:sha(target),proof}); console.log(`${name}: ${detail.seconds.toFixed(2)}s usable ${Math.max(0,detail.seconds-startSeconds).toFixed(2)}s`);
}
const captureStarted=new Map();
const originalCapture=capture;
// Record the raw wall-clock start before each named choreography without touching product state.
function paced(name, metadata, choreography) { captureStarted.set(name,Date.now()); return originalCapture(name,metadata,choreography); }

const browser=await chromium.launch({headless:true});
try {
 await paced('01_reference_uncertainty_v2',{mode:'REFERENCE',start:'1 initial_uncertainty',end:'2 evidence_insufficient',purpose:'Long source uncertainty and planning-withheld proof'},async(page,mark)=>{await reference(page);await milestone(page,1);const source=await rail(page,'Source','Ranked source candidates');mark();await hold(page,4000);await move(page,source);await hold(page,3000);const next=page.getByRole('button',{name:'Next',exact:true});await click(page,next);await visible(page.getByText(/Evidence insufficient to plan/i),'planning withheld');await hide(page);await hold(page,13000);});
 await paced('02_reference_sampling_to_posterior_v2',{mode:'REFERENCE',start:'3 sample_recommended',end:'5 posterior_contracted',purpose:'Causal sampling to posterior-contraction proof'},async(page,mark,proof)=>{await reference(page);await milestone(page,3);const sampling=await rail(page,'Sampling','Evidence status');const replay=page.getByRole('button',{name:'Replay sample collection'});await visible(replay,'sample replay');if(await page.getByRole('button',{name:'Next',exact:true}).isEnabled())throw new Error('sample pause bypass enabled');mark();await hold(page,5000);await move(page,sampling);await hold(page,1000);await click(page,replay);await visible(page.getByText(/Sample arrives/i),'sample arrival');await hide(page);await hold(page,4000);await click(page,page.getByRole('button',{name:'Next',exact:true}));await visible(page.getByText('Posterior contracts'),'posterior');await rail(page,'Source','Ranked source candidates');await hide(page);await hold(page,10500);await proof('posterior-proof');});
 await paced('03_reference_unsafe_plan_rejected_v2',{mode:'REFERENCE',start:'6 plans_generated',end:'7 unsafe_plan_rejected',purpose:'Long deterministic unsafe-plan rejection proof'},async(page,mark,proof)=>{await reference(page);await milestone(page,6);await rail(page,'Response','Verified plan comparison');const unsafe=page.locator('.table-plan-button').first();await visible(unsafe,'unsafe plan');mark();await hold(page,5000);await move(page,unsafe);await hold(page,2000);await click(page,unsafe);await hold(page,3500);await click(page,page.getByRole('button',{name:'Next',exact:true}));await visible(page.getByText('REJECTED').first(),'rejected');await visible(page.getByText('PRESSURE_BELOW_MINIMUM').first(),'rejection reason');await hide(page);await hold(page,15000);await proof('rejection-proof');});
 await paced('04_reference_verified_alternative_v2',{mode:'REFERENCE',start:'8 safe_plan_verified',end:'8 safe_plan_verified',purpose:'Verified J4 alternative and consequence proof'},async(page,mark)=>{await reference(page);await milestone(page,8);await rail(page,'Response','Verified plan comparison');const safe=page.locator('.table-plan-button').nth(1);await visible(safe,'verified alternative');mark();await hold(page,3000);await click(page,safe);await visible(page.getByText('VERIFIED').first(),'verified verdict');await move(page,page.getByRole('button',{name:'Next',exact:true}));await hide(page);await hold(page,11500);});
 await paced('05_reference_human_approval_v2',{mode:'REFERENCE',start:'9 human_approval_boundary',end:'10 completed',purpose:'Long human approval boundary and completed-state proof'},async(page,mark,proof)=>{await reference(page);await milestone(page,9);await rail(page,'Approval','Operator approval');const gate=page.getByRole('region',{name:'Decision gate'}),replay=page.getByRole('button',{name:'Replay operator approval'});await visible(gate,'decision gate');await visible(replay,'operator approval replay');await visible(gate.getByText('Infrastructure actuation'),'actuation context');if(await page.getByRole('button',{name:'Next',exact:true}).isEnabled())throw new Error('approval pause bypass enabled');mark();await hold(page,7500);await move(page,gate);await hold(page,3000);await proof('approval-boundary-proof');await click(page,replay);await visible(page.getByText(/Response approved and simulated/i),'completed');await hide(page);await hold(page,5000);});
 await paced('reference_master_take_v2',{mode:'REFERENCE',start:'Gateway',end:'10 completed',purpose:'Long-form presentation-paced authored replay backup'},async(page,mark)=>{await page.goto('/');const starter=page.getByRole('button',{name:'Run Reference Incident'});await visible(starter,'gateway');mark();await hold(page,4000);await click(page,starter);await visible(page.getByText('REFERENCE INCIDENT · VERIFIED REPLAY'),'reference');await page.getByRole('button',{name:'Pause',exact:true}).click();await hold(page,4000);let next=page.getByRole('button',{name:'Next',exact:true});await click(page,next);await rail(page,'Source','Ranked source candidates');await hold(page,5000);await click(page,next);await hold(page,6000);await click(page,next);await rail(page,'Sampling','Evidence status');await hold(page,5000);await click(page,page.getByRole('button',{name:'Replay sample collection'}));await hold(page,4000);await click(page,next);await visible(page.getByText('Posterior contracts'),'posterior');await hold(page,5000);await click(page,next);await rail(page,'Response','Verified plan comparison');await hold(page,5000);const unsafe=page.locator('.table-plan-button').first();await click(page,unsafe);await hold(page,3500);await click(page,next);await visible(page.getByText('REJECTED').first(),'rejected');await hide(page);await hold(page,9500);await click(page,next);const safe=page.locator('.table-plan-button').nth(1);await click(page,safe);await hide(page);await hold(page,6000);await click(page,next);await rail(page,'Approval','Operator approval');await hold(page,9000);await click(page,page.getByRole('button',{name:'Replay operator approval'}));await visible(page.getByText(/Response approved and simulated/i),'completed');await hide(page);await hold(page,7000);});
} finally { await browser.close(); }
const dataFile=path.join(manifest,'recording-data.json');
const prior=existsSync(dataFile)?JSON.parse(await readFile(dataFile,'utf8')):[];
const combined=[...prior.filter(item=>!captures.some(capture=>capture.name===item.name)),...captures];
await writeFile(dataFile,JSON.stringify(combined,null,2));
