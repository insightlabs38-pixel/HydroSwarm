/*
 * Local-only HydroSwarm footage harness.  It intentionally contains no API
 * routing/interception: both REFERENCE and LIVE use the running production
 * runtime at 127.0.0.1:8765.  The REFERENCE journey uses its authored UI
 * controls (including the two non-bypassable replay actions).
 */
import { chromium } from 'playwright';
import { mkdir, rename, copyFile, stat, writeFile, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..', '..');
const output = path.join(root, 'demo-recordings');
const rawDir = path.join(output, 'raw');
const takeDir = path.join(rawDir, 'native-takes');
const mp4Dir = path.join(output, 'mp4');
const qcDir = path.join(output, 'qc');
const manifestDir = path.join(output, 'manifest');
const ffmpeg = '/root/.cache/ms-playwright/ffmpeg-1011/ffmpeg-linux';
const baseURL = 'http://127.0.0.1:8765';
const records = [];
const selected = process.env.CLIPS ? new Set(process.env.CLIPS.split(',').filter(Boolean)) : null;

for (const dir of [rawDir, takeDir, mp4Dir, qcDir, manifestDir]) await mkdir(dir, { recursive: true });
if (!existsSync(ffmpeg)) throw new Error(`Playwright bundled ffmpeg unavailable: ${ffmpeg}`);

function runFfmpeg(args, label) {
  const result = spawnSync(ffmpeg, ['-hide_banner', '-y', ...args], { encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`${label} failed: ${result.stderr.slice(-1800)}`);
  return result.stderr;
}

function info(video) {
  const result = spawnSync(ffmpeg, ['-hide_banner', '-i', video], { encoding: 'utf8' });
  const text = `${result.stdout}\n${result.stderr}`;
  const duration = text.match(/Duration: (\d\d):(\d\d):(\d\d\.\d+)/);
  const dimensions = text.match(/(\d{3,5})x(\d{3,5})/);
  if (!duration || !dimensions) throw new Error(`Unable to inspect ${video}: ${text.slice(-1200)}`);
  const seconds = Number(duration[1]) * 3600 + Number(duration[2]) * 60 + Number(duration[3]);
  return { seconds, width: Number(dimensions[1]), height: Number(dimensions[2]) };
}

async function sha256(file) {
  const result = spawnSync('sha256sum', [file], { encoding: 'utf8' });
  if (result.status !== 0) throw new Error(result.stderr);
  return result.stdout.split(/\s+/)[0];
}

async function installCursor(page) {
  await page.addStyleTag({ content: `
    #hs-demo-cursor { position: fixed; left: 0; top: 0; width: 26px; height: 26px;
      border-radius: 50%; pointer-events:none; z-index:2147483647; opacity:0;
      transform:translate(-50%,-50%); background:rgba(238,244,247,.72);
      border:1px solid rgba(21,31,37,.78); box-shadow:0 2px 8px rgba(0,0,0,.46), inset 0 0 0 1px rgba(255,255,255,.32);
      transition:opacity 160ms ease, transform 100ms ease; }
    #hs-demo-cursor.hs-click { transform:translate(-50%,-50%) scale(.85); }
    #hs-demo-cursor::after { content:''; position:absolute; inset:-5px; border:1px solid rgba(230,240,243,.55);
      border-radius:50%; opacity:0; }
    #hs-demo-cursor.hs-click::after { animation:hs-cursor-ring 190ms ease-out; }
    @keyframes hs-cursor-ring { from { transform:scale(.72); opacity:.68; } to { transform:scale(1.38); opacity:0; } }
  ` });
  await page.evaluate(() => {
    const el = document.createElement('div'); el.id = 'hs-demo-cursor';
    el.dataset.x = '-100'; el.dataset.y = '-100'; document.body.append(el);
  });
}

async function cursor(page, locator, visible = true) {
  const box = await locator.boundingBox();
  if (!box) throw new Error('Cursor target has no box');
  const x = box.x + box.width / 2, y = box.y + box.height / 2;
  await page.evaluate(async ({ x, y, visible }) => {
    const el = document.querySelector('#hs-demo-cursor'); if (!el) return;
    const fromX = Number(el.dataset.x || -100), fromY = Number(el.dataset.y || -100);
    el.style.opacity = visible ? '1' : '0';
    const distance = Math.hypot(x - fromX, y - fromY); const duration = Math.min(650, Math.max(320, 300 + distance * .13));
    const started = performance.now();
    await new Promise((done) => { const step = (now) => { const p=Math.min(1,(now-started)/duration); const e=p<.5?4*p*p*p:1-Math.pow(-2*p+2,3)/2;
      el.style.left=`${fromX+(x-fromX)*e}px`; el.style.top=`${fromY+(y-fromY)*e}px`; if(p<1) requestAnimationFrame(step); else done(); }; requestAnimationFrame(step); });
    el.dataset.x=String(x); el.dataset.y=String(y);
  }, { x, y, visible });
}
async function hideCursor(page) { await page.evaluate(() => { const el=document.querySelector('#hs-demo-cursor'); if(el) el.style.opacity='0'; }); }
async function hold(page, ms) { await page.waitForTimeout(ms); }
async function clickWithCursor(page, locator) {
  await cursor(page, locator); await hold(page, 700);
  await page.evaluate(() => document.querySelector('#hs-demo-cursor')?.classList.add('hs-click'));
  await locator.click(); await hold(page, 140);
  await page.evaluate(() => document.querySelector('#hs-demo-cursor')?.classList.remove('hs-click'));
}
async function expectVisible(locator, label) { await locator.waitFor({ state: 'visible', timeout: 45000 }); if (!(await locator.isVisible())) throw new Error(`${label} not visible`); }
async function noOverflow(page) {
  const dimensions = await page.evaluate(() => ({ client:document.documentElement.clientWidth, scroll:document.documentElement.scrollWidth }));
  if (dimensions.scroll > dimensions.client) throw new Error(`Horizontal overflow: ${dimensions.scroll} > ${dimensions.client}`);
}
async function startReference(page, cursorClick = false) {
  await page.goto('/');
  const reference = page.getByRole('button', { name: 'Run Reference Incident' });
  await expectVisible(reference, 'reference gateway');
  if (cursorClick) await clickWithCursor(page, reference); else await reference.click();
  await expectVisible(page.getByText('REFERENCE INCIDENT · VERIFIED REPLAY'), 'reference provenance');
  await expectVisible(page.locator('.map-canvas[role="img"]'), 'reference map');
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
}
async function atMilestone(page, milestone) {
  for (let index=0; index<milestone; index += 1) {
    const next=page.getByRole('button', {name:'Next', exact:true});
    if (await next.isEnabled()) await next.click();
    else await page.locator('.mode-banner-controls button').filter({hasText:/^Replay /}).click();
    await hold(page, 170);
  }
  await expectVisible(page.locator('.mode-banner-milestone'), `milestone ${milestone}`);
}
async function workspace(page, name, heading) {
  const rail=page.getByRole('button', {name:new RegExp(`^${name}:`)});
  await rail.click(); await expectVisible(page.getByRole('heading', {name:heading}), heading); await hold(page, 700);
}

async function record(name, meta, action) {
  if (selected && !selected.has(name)) return;
  if (existsSync(path.join(rawDir, `${name}.webm`))) {
    console.log(`${name}: existing capture retained`);
    return;
  }
  const started=Date.now();
  const context=await browser.newContext({ baseURL, viewport:{width:1920,height:1080}, deviceScaleFactor:1, recordVideo:{dir:takeDir,size:{width:1920,height:1080}} });
  const page=await context.newPage(); await installCursor(page);
  let usableAt=Date.now();
  try { await action(page, () => { usableAt=Date.now(); return usableAt; }); await noOverflow(page); await hold(page, 1100); }
  catch (error) { await context.close(); throw error; }
  const video=page.video(); await context.close();
  if (!video) throw new Error(`${name}: no video`);
  const native=await video.path();
  const nativeTarget=path.join(takeDir, `${name}.webm`); await rename(native, nativeTarget);
  const trimStart=Math.max(0, (usableAt-started)/1000);
  const raw=path.join(rawDir, `${name}.webm`);
  // Preserve Playwright's native VP8 WebM exactly. The bundled ffmpeg can
  // decode frames for QC but lacks H.264 and re-encodes VP8 too slowly for
  // practical capture. `trimStart` remains in the manifest as the clean
  // editor in-point; native takes are intentionally retained unchanged.
  await copyFile(nativeTarget, raw);
  const details=info(raw);
  if (details.width !== 1920 || details.height !== 1080) throw new Error(`${name}: wrong resolution ${details.width}x${details.height}`);
  if ((await stat(raw)).size < 10_000) throw new Error(`${name}: empty clip`);
  const usableStart=Math.min(Math.max(0.25, trimStart + 0.25), Math.max(0.25, details.seconds - 0.5));
  const frameTimes=[usableStart, Math.max(usableStart, (usableStart + details.seconds) / 2), Math.max(usableStart, details.seconds-1)];
  for (const [label, seconds] of [['first',frameTimes[0]],['mid',frameTimes[1]],['final',frameTimes[2]]])
    runFfmpeg(['-ss',Number(seconds).toFixed(3),'-i',raw,'-frames:v','1',path.join(qcDir,`${name}-${label}.png`)], `${name} ${label} QC`);
  records.push({...meta,name,raw:path.relative(output,raw),native:path.relative(output,nativeTarget),mp4:null,duration:details.seconds.toFixed(2),usableDuration:Math.max(0,details.seconds-trimStart).toFixed(2),sha256:await sha256(raw),trimStart:trimStart.toFixed(2)});
  console.log(`${name}: ${details.seconds.toFixed(2)}s 1920x1080`);
}

const browser=await chromium.launch({headless:true});
try {
  await record('00_gateway_to_reference', {mode:'REFERENCE',start:'Gateway',end:'0 alert',actions:'Run Reference Incident',cursor:'yes',section:'Solution introduction'}, async (page, mark) => {
    await page.goto('/'); const button=page.getByRole('button',{name:'Run Reference Incident'}); await expectVisible(button,'gateway'); mark(); await hold(page,4000); await clickWithCursor(page,button); await expectVisible(page.getByText('REFERENCE INCIDENT · VERIFIED REPLAY'),'reference'); await expectVisible(page.locator('.map-canvas[role="img"]'),'map'); await hideCursor(page); await hold(page,4000); return Date.now();
  });
  await record('01_reference_uncertainty', {mode:'REFERENCE',start:'1 initial_uncertainty',end:'2 evidence_insufficient',actions:'Source workspace; Next',cursor:'yes',section:'Uncertainty and planning withheld'}, async (page, mark) => {
    await startReference(page); await atMilestone(page,1); await workspace(page,'Source','Ranked source candidates'); mark(); await hold(page,5000); const next=page.getByRole('button',{name:'Next',exact:true}); await clickWithCursor(page,next); await expectVisible(page.getByText(/Evidence insufficient to plan/i),'evidence gate'); await hideCursor(page); await hold(page,4200); return Date.now();
  });
  await record('02_reference_sampling_to_posterior', {mode:'REFERENCE',start:'3 sample_recommended',end:'5 posterior_contracted',actions:'Replay sample collection; Next',cursor:'yes',section:'Adaptive sampling to posterior contraction'}, async (page, mark) => {
    await startReference(page); await atMilestone(page,3); await workspace(page,'Sampling','Evidence status'); const replay=page.getByRole('button',{name:'Replay sample collection'}); await expectVisible(replay,'sample replay'); if(await page.getByRole('button',{name:'Next',exact:true}).isEnabled()) throw new Error('sample pause Next unexpectedly enabled'); mark(); await hold(page,4000); await clickWithCursor(page,replay); await expectVisible(page.getByText(/Sample arrives/i),'sample arrives'); await hold(page,2200); await clickWithCursor(page,page.getByRole('button',{name:'Next',exact:true})); await expectVisible(page.getByText('Posterior contracts'),'posterior'); await hideCursor(page); await hold(page,5000); return Date.now();
  });
  await record('03_reference_unsafe_plan_rejected', {mode:'REFERENCE',start:'6 plans_generated',end:'7 unsafe_plan_rejected',actions:'Select unsafe candidate; Next',cursor:'yes',section:'Unsafe response rejection'}, async (page, mark) => {
    await startReference(page); await atMilestone(page,6); await workspace(page,'Response','Verified plan comparison'); const unsafe=page.locator('.table-plan-button').first(); await expectVisible(unsafe,'unsafe plan'); mark(); await hold(page,2200); await clickWithCursor(page,unsafe); await hold(page,1000); await clickWithCursor(page,page.getByRole('button',{name:'Next',exact:true})); await expectVisible(page.getByText('REJECTED').first(),'rejected'); await expectVisible(page.getByText('PRESSURE_BELOW_MINIMUM').first(),'actual rejection reason'); await hideCursor(page); await hold(page,3300); return Date.now();
  });
  await record('04_reference_verified_alternative', {mode:'REFERENCE',start:'8 safe_plan_verified',end:'8 safe_plan_verified',actions:'Select verified alternative',cursor:'yes',section:'Verified alternative and consequences'}, async (page, mark) => {
    await startReference(page); await atMilestone(page,8); await workspace(page,'Response','Verified plan comparison'); const safe=page.locator('.table-plan-button').nth(1); await expectVisible(safe,'safe plan'); mark(); await hold(page,1500); await clickWithCursor(page,safe); await expectVisible(page.getByText('VERIFIED').first(),'verified'); await hideCursor(page); await hold(page,4200); return Date.now();
  });
  await record('05_reference_human_approval', {mode:'REFERENCE',start:'9 human_approval_boundary',end:'10 completed',actions:'Replay operator approval',cursor:'yes',section:'Human approval boundary'}, async (page, mark) => {
    await startReference(page); await atMilestone(page,9); await workspace(page,'Approval','Operator approval'); const replay=page.getByRole('button',{name:'Replay operator approval'}); await expectVisible(page.getByRole('region',{name:'Decision gate'}),'decision gate'); await expectVisible(replay,'approval replay'); await expectVisible(page.getByRole('region',{name:'Decision gate'}).getByText('Infrastructure actuation'),'actuation context'); if(await page.getByRole('button',{name:'Next',exact:true}).isEnabled()) throw new Error('approval pause Next unexpectedly enabled'); mark(); await hold(page,3000); await clickWithCursor(page,replay); await expectVisible(page.getByText(/Response approved and simulated/i),'completed state'); await hideCursor(page); await hold(page,3200); return Date.now();
  });
  await record('06_live_computation_proof', {mode:'LIVE',start:'Gateway',end:'awaiting_approval or completed',actions:'Run Live Example; collect real sample',cursor:'yes',section:'Live production computation proof'}, async (page, mark) => {
    await page.goto('/'); const live=page.getByRole('button',{name:'Run Live Example'}); await expectVisible(live,'live gateway'); mark(); await hold(page,1500); await clickWithCursor(page,live); await expectVisible(page.getByText('LIVE COMPUTATION · REFERENCE INPUTS'),'live provenance'); await expectVisible(page.getByRole('heading',{name:/Importing|Creating|Running|Real sampling/}),'live stage'); await expectVisible(page.getByRole('button',{name:'Collect reference sample'}),'real sample pause'); await hold(page,5000); await clickWithCursor(page,page.getByRole('button',{name:'Collect reference sample'})); await expectVisible(page.getByRole('heading',{name:/Running real exact WNTR\/EPANET verification|Verified plan ready for approval|Recording approval/}),'real verification/approval'); await hideCursor(page); await hold(page,12000); return Date.now();
  });
  await record('07_completed_incident', {mode:'REFERENCE',start:'10 completed',end:'10 completed',actions:'Static completed incident',cursor:'no',section:'Closing b-roll'}, async (page, mark) => {
    await startReference(page); await atMilestone(page,10); await expectVisible(page.getByText(/Response approved and simulated/i),'completed'); mark(); await hideCursor(page); await hold(page,9000); return Date.now();
  });
  await record('08_model_authority_broll', {mode:'REFERENCE',start:'8 safe_plan_verified',end:'8 safe_plan_verified',actions:'Open Model & Authority',cursor:'yes',section:'Optional model authority b-roll'}, async (page, mark) => {
    await startReference(page); await atMilestone(page,8); const authority=page.getByRole('button',{name:/^Model & Authority:/}); await authority.click(); await expectVisible(page.getByText(/Authority ladder/i),'authority ladder'); mark(); await cursor(page,authority); await hold(page,1200); await hideCursor(page); await hold(page,7500); return Date.now();
  });
  await record('reference_master_take', {mode:'REFERENCE',start:'Gateway',end:'10 completed',actions:'Full authored replay with both replay-only actions',cursor:'yes',section:'Master transition backup'}, async (page, mark) => {
    await page.goto('/'); const ref=page.getByRole('button',{name:'Run Reference Incident'}); await expectVisible(ref,'gateway'); mark(); await hold(page,2000); await clickWithCursor(page,ref); await expectVisible(page.getByText('REFERENCE INCIDENT · VERIFIED REPLAY'),'reference'); await page.getByRole('button',{name:'Pause',exact:true}).click(); await hold(page,1600);
    for(let index=0;index<10;index++) { const next=page.getByRole('button',{name:'Next',exact:true}); if(await next.isEnabled()) { await clickWithCursor(page,next); } else { const action=page.locator('.mode-banner-controls button').filter({hasText:/^Replay /}); await clickWithCursor(page,action); } await hold(page,1500); }
    await expectVisible(page.getByText(/Response approved and simulated/i),'master completed'); await hideCursor(page); await hold(page,2600); return Date.now();
  });
} finally { await browser.close(); }

const recordingData = path.join(manifestDir,'recording-data.json');
let previous = [];
if (existsSync(recordingData)) previous = JSON.parse(await readFile(recordingData, 'utf8'));
const combined = [...previous.filter((item) => !records.some((record) => record.name === item.name)), ...records];
await writeFile(recordingData, JSON.stringify(combined,null,2));
