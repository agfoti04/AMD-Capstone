// Exercise the shipped browser script with a deterministic DOM/HTTP harness.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function element(key) {
  return {dataset: {key}, events: {}, className: '', textContent: '',
    style: {}, classList: {toggle() {}}, addEventListener(name, fn) {this.events[name] = fn;},
    setPointerCapture() {}};
}
const buttons = ['w', 'a', 's', 'd'].map(element);
const status = element(), stop = element(), detections = element();
const driveMode = element(), browserControls = element();
const events = {}, intervals = new Map(), requests = [];
let camera = {ready: true, controls_enabled: true, control_mode: 'browser', error: null, loop_fps: 8, inference_ms: 20,
              frame_age_seconds: 0.1, camera_timeout_seconds: 3,
              camera_delayed: false, detections: []};
let commandFailed = false;
let delayCommand = false, finishCommand;
const document = {hidden: false, events: {},
  querySelector: selector => ({'#status': status, '#stop': stop, '#detections': detections,
    '#driveMode': driveMode, '#browserControls': browserControls})[selector],
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
  camera = {...camera, frame_age_seconds: 1.41, camera_delayed: true};
  await intervals.get(500)(); await tick();
  assert.equal(vm.runInContext('held.size', context), 1);
  assert.match(status.textContent, /Camera delayed/);
  intervals.get(100)(); await tick();
  assert.deepEqual(latest(), {linear: 0.3, angular: 0}); // brief camera gap preserves held input
  camera = {...camera, frame_age_seconds: 0.1, camera_delayed: false};
  camera = {...camera, ready: false, frame_age_seconds: 3.2};
  await intervals.get(500)(); await tick();
  assert.equal(vm.runInContext('held.size', context), 0);
  assert.deepEqual(latest(), {linear: 0, angular: 0});
  assert.match(status.textContent, /Driving is paused/);
  camera = {...camera, ready: true, frame_age_seconds: 0.1};
  await intervals.get(500)(); await tick();
  key('w', 'keydown', true); await tick();
  assert.equal(vm.runInContext('held.size', context), 0);
  key('w'); await tick();
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
  commandFailed = false;
  camera = {...camera, controls_enabled: false, control_mode: 'gazebo'};
  await intervals.get(500)(); await tick();
  assert.equal(browserControls.style.display, 'none');
  assert.match(driveMode.textContent, /Gazebo Teleop keyboard/);
  const viewerCount = requests.length;
  key('w'); key(' '); events.blur(); events.pagehide();
  forward.events.pointerdown({preventDefault() {}, pointerId: 2});
  forward.events.pointercancel(); stop.events.click();
  document.events.visibilitychange(); intervals.get(100)(); await tick();
  assert.equal(requests.length, viewerCount); // viewer NEVER sends idle/release/close stops
  camera = {...camera, ready: false, error: 'Camera stopped updating.'};
  await intervals.get(500)(); await tick();
  assert.equal(requests.length, viewerCount); // inference/camera stalls cannot stop native driving
  assert.equal(vm.runInContext('held.size', context), 0);
  console.log('PASS: browser keyboard/pointer controls, heartbeats, camera grace/stop, reconnect, serialized requests and Gazebo viewer sends no commands');
})().catch(error => {console.error(error); process.exitCode = 1;});
