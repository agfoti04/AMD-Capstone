// Exercise the shipped browser script with a deterministic DOM/HTTP harness.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function element(key) {
  return {dataset: {key}, events: {}, className: '', textContent: '',
    classList: {toggle() {}}, addEventListener(name, fn) {this.events[name] = fn;},
    setPointerCapture() {}};
}
const buttons = ['w', 'a', 's', 'd'].map(element);
const status = element(), stop = element(), detections = element();
const events = {}, intervals = new Map(), requests = [];
let camera = {ready: true, error: null, loop_fps: 8, inference_ms: 20,
              frame_age_seconds: 0.1, detections: []};
let commandFailed = false;
let delayCommand = false, finishCommand;
const document = {hidden: false, events: {},
  querySelector: selector => ({'#status': status, '#stop': stop, '#detections': detections})[selector],
  querySelectorAll: () => buttons,
  addEventListener(name, fn) {this.events[name] = fn;}};
const context = vm.createContext({document,
  window: {addEventListener: (name, fn) => {events[name] = fn;}},
  setInterval: (fn, delay) => {intervals.set(delay, fn);},
  AbortSignal: {timeout: () => undefined},
  fetch: async (url, options) => {
    if (url === '/status') return {ok: true, json: async () => camera};
    requests.push(JSON.parse(options.body));
    if(delayCommand){delayCommand=false;await new Promise(resolve=>{finishCommand=resolve;});}
    return {ok: !commandFailed};
  }});
const html = fs.readFileSync(path.join(__dirname, '../rover_perception/live_dashboard.html'), 'utf8');
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
const tick = () => new Promise(resolve => setImmediate(resolve));
const key = (name, type='keydown', repeat=false) => events[type]({key: name, repeat, preventDefault() {}});
const latest = () => requests.at(-1);

(async () => {
  await tick();
  key('w'); await tick();
  assert.deepEqual(latest(), {linear: 0.3, angular: 0});
  intervals.get(100)(); await tick();
  assert.deepEqual(latest(), {linear: 0.3, angular: 0}); // heartbeat while held
  key('d'); await tick();
  assert.deepEqual(latest(), {linear: 0.3, angular: -0.8});
  key('w', 'keyup'); await tick();
  assert.deepEqual(latest(), {linear: 0, angular: -0.8});
  key(' '); await tick();
  assert.deepEqual(latest(), {linear: 0, angular: 0});
  key('ArrowUp'); await tick();
  events.blur(); await tick();
  assert.deepEqual(latest(), {linear: 0, angular: 0});
  key('w', 'keydown', true); await tick();
  assert.equal(vm.runInContext('held.size', context), 0); // repeats cannot restart after blur
  const forward = buttons[0];
  forward.events.pointerdown({preventDefault() {}, pointerId: 1}); await tick();
  assert.deepEqual(latest(), {linear: 0.3, angular: 0});
  forward.events.pointercancel(); await tick();
  assert.deepEqual(latest(), {linear: 0, angular: 0});
  key('w'); await tick();
  document.hidden = true;
  document.events.visibilitychange(); await tick();
  assert.deepEqual(latest(), {linear: 0, angular: 0});
  document.hidden = false;
  key('w'); await tick();
  camera = {...camera, ready: false, error: 'Camera unavailable'};
  await intervals.get(500)(); await tick();
  assert.deepEqual(latest(), {linear: 0, angular: 0});
  assert.equal(vm.runInContext('held.size', context), 0);
  camera = {...camera, ready: true, error: null};
  await intervals.get(500)(); await tick();
  key('w', 'keydown', true); await tick();
  assert.equal(vm.runInContext('held.size', context), 0);
  delayCommand = true;
  key('w'); await tick();
  const pendingCount = requests.length;
  key('w', 'keyup'); await tick();
  assert.equal(requests.length, pendingCount); // no overlapping move/stop requests
  finishCommand(); await tick();
  assert.deepEqual(latest(), {linear: 0, angular: 0});
  commandFailed = true;
  key('w'); await tick();
  assert.equal(vm.runInContext('held.size', context), 0);
  assert.match(status.textContent, /disconnected/);
  events.pagehide(); await tick();
  assert.deepEqual(latest(), {linear: 0, angular: 0});
  console.log('PASS: keyboard, arrows, heartbeat, release, Stop/Space, pointer cancellation, blur, hidden page, stale camera, reconnect and command failure');
})().catch(error => {console.error(error); process.exitCode = 1;});
