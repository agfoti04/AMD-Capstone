// Exercise the shipped browser logic with deterministic WebSocket/DOM doubles.
const assert=require('node:assert/strict'), fs=require('node:fs'), path=require('node:path'), vm=require('node:vm');
function element(key){return {dataset:{key},events:{},value:'',textContent:'',className:'',style:{},classList:{toggle(){}},addEventListener(k,fn){this.events[k]=fn;},setPointerCapture(){}};}
const buttons=['w','a','s','d'].map(element), elements={};
for(const id of ['camera','speed','turn','speedValue','turnValue','status','stop','detections','driveMode','fps','inference','latency','age','simSpeed'])elements['#'+id]=element();
elements['#speed'].value='0.30';elements['#turn'].value='0.80';
let draws=0, time=10000, ack=true;
elements['#camera'].getContext=()=>({drawImage(){draws++;}});
const events={}, intervals=new Map(), timers=[], sockets=[];
class Socket{
 static CONNECTING=0;static OPEN=1;static CLOSING=2;static CLOSED=3;
 constructor(url){this.url=url;this.readyState=0;this.bufferedAmount=0;this.sent=[];sockets.push(this);}
 open(){this.readyState=1;this.onopen?.();}
 send(data){this.sent.push(data);if(ack&&this.url.endsWith('/control'))this.onmessage?.({data:JSON.stringify({seq:JSON.parse(data).seq,ok:true})});}
 close(){this.readyState=3;this.onclose?.();}
}
const document={hidden:false,querySelector:s=>elements[s],querySelectorAll:()=>buttons,addEventListener(k,fn){events[k]=fn;}};
const context=vm.createContext({document,location:{protocol:'http:',host:'localhost:8771'},performance:{now:()=>time},WebSocket:Socket,
 window:{addEventListener(k,fn){events[k]=fn;}},setInterval:(fn,ms)=>intervals.set(ms,fn),setTimeout:(fn,ms)=>timers.push(fn),
 createImageBitmap:async()=>({width:960,height:540,close(){}})});
const html=fs.readFileSync(path.join(__dirname,'../rover_perception/live_dashboard.html'),'utf8');
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
const control=sockets.find(s=>s.url.endsWith('/control')), view=sockets.find(s=>s.url.endsWith('/view'));
const key=(name,type='keydown',repeat=false,editable=false)=>events[type]({key:name,repeat,target:{matches:()=>editable},preventDefault(){}});
const latest=()=>{const c=JSON.parse(control.sent.at(-1));return {linear:c.linear,angular:c.angular};};
const snapshot={type:'frame',controls_enabled:true,ready:true,error:null,camera_timeout_seconds:3,frame_age_seconds:0.01,inference_ms:20,sim_speed:1,detections:[]};
async function frame(s={}){await view.onmessage({data:JSON.stringify({...snapshot,...s})});time+=10;await view.onmessage({data:{}});}
(async()=>{
 control.open();view.open();
 key('w');assert.deepEqual(latest(),{linear:0,angular:0}); // wait for an actually displayed frame
 await frame();assert.equal(draws,1);assert.equal(view.sent.length,2); // demand only after draw
 key('w');assert.deepEqual(latest(),{linear:0.3,angular:0});
 time+=50;intervals.get(50)();assert.deepEqual(latest(),{linear:0.3,angular:0});
 elements['#speed'].value='0.60';elements['#speed'].events.input();assert.deepEqual(latest(),{linear:0.6,angular:0});
 assert.equal(elements['#speedValue'].textContent,'0.60 m/s');
 key('d');assert.deepEqual(latest(),{linear:0.6,angular:-0.8});
 elements['#turn'].value='1.50';elements['#turn'].events.input();assert.deepEqual(latest(),{linear:0.6,angular:-1.5});
 key('w','keyup');assert.deepEqual(latest(),{linear:0,angular:-1.5});
 key(' ');assert.deepEqual(latest(),{linear:0,angular:0});
 key('ArrowUp','keydown',false,true);assert.deepEqual(latest(),{linear:0,angular:0}); // editing a slider does not drive
 key('ArrowUp');assert.deepEqual(latest(),{linear:0.6,angular:0});
 time+=1400;intervals.get(50)();assert.deepEqual(latest(),{linear:0.6,angular:0}); // brief frame gap
 events.blur();assert.deepEqual(latest(),{linear:0,angular:0});
 key('w','keydown',true);assert.equal(vm.runInContext('held.size',context),0); // no auto-resume
 await frame();
 buttons[0].events.pointerdown({preventDefault(){},pointerId:1});assert.deepEqual(latest(),{linear:0.6,angular:0});
 buttons[0].events.pointercancel();assert.deepEqual(latest(),{linear:0,angular:0});
 key('w');time+=3100;intervals.get(50)();assert.deepEqual(latest(),{linear:0,angular:0});
 await frame();key('w','keydown',true);assert.equal(vm.runInContext('held.size',context),0);
 key('w');ack=false;key('w','keyup');assert.deepEqual(latest(),{linear:0,angular:0}); // release never waits for an ACK
 ack=true;pendingReset();
 key('w');document.hidden=true;events.visibilitychange();assert.deepEqual(latest(),{linear:0,angular:0});assert.equal(control.readyState,3);
 document.hidden=false;events.visibilitychange();
 const resumed=sockets.filter(s=>s.url.endsWith('/control')).at(-1);resumed.open();assert.equal(vm.runInContext('held.size',context),0);
 ack=false;intervals.get(50)();time+=501;intervals.get(50)();assert.equal(resumed.readyState,3); // expired acknowledgements disconnect
 console.log('PASS: held WASD/arrows, live speed sliders, release before ACK, Space/blur/hidden stops, pointer cancellation, stale views, reconnect and bounded frame delivery');
})().catch(e=>{console.error(e);process.exitCode=1;});
function pendingReset(){vm.runInContext('pending.clear()',context);}
