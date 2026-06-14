import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

document.querySelector('#app').innerHTML = `
  <div id="canvas-wrap"></div>

  <div class="hud closed" id="hud">
    <button id="panelToggle">▶</button>

    <div class="hud-body">
      <h1>Smart Parking Live 3D</h1>
      <p>Live 3D parking simulation with entry/exit routes, barriers, smart red/green slot sensors, and 360 camera.</p>

      <div class="stats">
        <div class="stat"><span>Occupied</span><strong id="occupied">0</strong></div>
        <div class="stat"><span>Occupancy</span><strong id="occupancyRate">0%</strong></div>
        <div class="stat"><span>Arrived</span><strong id="arrived">0</strong></div>
        <div class="stat"><span>Parked</span><strong id="parked">0</strong></div>
        <div class="stat"><span>Exited</span><strong id="exited">0</strong></div>
        <div class="stat"><span>Rejected</span><strong id="rejected">0</strong></div>
        <div class="stat"><span>Avg Wait</span><strong id="avgWait">0s</strong></div>
        <div class="stat"><span>Demand</span><strong id="demand">Low</strong></div>
      </div>

      <div class="controls">
        <button id="pauseBtn">Pause</button>
        <button id="addCarBtn">Add Car</button>
        <button id="resetBtn">Reset</button>
      </div>
    </div>
  </div>

  <div class="help">
    Drag: rotate 360<br>
    Scroll: zoom in/out<br>
    Right drag: pan
  </div>
`;

const ROWS = 10;
const COLS = 30;
const HALF_COLS = COLS / 2;
const TOTAL_SLOTS = ROWS * COLS;

const SLOT_WIDTH = 2.05;
const SLOT_LENGTH = 3.25;
const SLOT_GAP = 0.26;
const ROW_GAP = 1.28;
const CENTER_AISLE = 8.4;

const CAR_Y = 0.43;

const ENTRY_GATE_X = -11.5;
const EXIT_GATE_X = 11.5;
const ENTRY_LANE_X = -2.25;
const EXIT_LANE_X = 2.25;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdbeafe);

const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.1, 1200);
camera.position.set(0, 76, 66);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById('canvas-wrap').appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 2);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.enableZoom = true;
controls.enablePan = true;
controls.minDistance = 20;
controls.maxDistance = 115;
controls.update();

const hemi = new THREE.HemisphereLight(0xffffff, 0x94a3b8, 1.85);
scene.add(hemi);

const sun = new THREE.DirectionalLight(0xffffff, 2.25);
sun.position.set(24, 42, 28);
sun.castShadow = true;
sun.shadow.mapSize.width = 2048;
sun.shadow.mapSize.height = 2048;
scene.add(sun);

const blockWidth = HALF_COLS * (SLOT_WIDTH + SLOT_GAP) - SLOT_GAP;
const lotWidth = blockWidth * 2 + CENTER_AISLE + 12;
const lotDepth = ROWS * (SLOT_LENGTH + ROW_GAP) - ROW_GAP;
const minZ = -lotDepth / 2;
const maxZ = lotDepth / 2;
const frontZ = maxZ + 10;
const backZ = minZ - 7;

const slots = [];
const activeMovers = [];

let paused = false;
let simTime = 0;
let lastSpawn = 0;
let lastDeparture = 0;
let totalArrived = 0;
let totalParked = 0;
let totalExited = 0;
let totalRejected = 0;
let waitTimes = [];
let currentDemand = 0.45;
let demandLabelStable = 'Medium';
let nextDemandCheckAt = 0;
let carCounter = 0;

const clock = new THREE.Clock();

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function randomChoice(list) {
  return list[Math.floor(Math.random() * list.length)];
}

function mat(color, roughness = 0.7, metalness = 0.04) {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness });
}

function addBox(name, sx, sy, sz, x, y, z, color, cast = false, receive = true) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat(color));
  mesh.name = name;
  mesh.position.set(x, y, z);
  mesh.castShadow = cast;
  mesh.receiveShadow = receive;
  scene.add(mesh);
  return mesh;
}

function addCylinder(name, radius, height, x, y, z, color) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, height, 24), mat(color));
  mesh.name = name;
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene.add(mesh);
  return mesh;
}

function makeSign(text, background, x, z) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 220;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = background;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 14;
  ctx.strokeRect(14, 14, canvas.width - 28, canvas.height - 28);
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 86px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture }));
  sprite.position.set(x, 4.2, z);
  sprite.scale.set(7.3, 3.05, 1);
  scene.add(sprite);

  addCylinder(`${text}-pole-left`, 0.08, 3.5, x - 2.15, 1.7, z, 0x334155);
  addCylinder(`${text}-pole-right`, 0.08, 3.5, x + 2.15, 1.7, z, 0x334155);
}

function addFloorArrow(label, x, z, rotation, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, 256, 256);
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.92;
  ctx.beginPath();
  ctx.moveTo(128, 18);
  ctx.lineTo(218, 110);
  ctx.lineTo(168, 110);
  ctx.lineTo(168, 230);
  ctx.lineTo(88, 230);
  ctx.lineTo(88, 110);
  ctx.lineTo(38, 110);
  ctx.closePath();
  ctx.fill();

  ctx.globalAlpha = 1;
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 42px Arial';
  ctx.textAlign = 'center';
  ctx.fillText(label, 128, 156);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(3.0, 3.0),
    new THREE.MeshBasicMaterial({ map: texture, transparent: true, side: THREE.DoubleSide, depthWrite: false })
  );

  mesh.position.set(x, 0.13, z);
  mesh.rotation.x = -Math.PI / 2;
  mesh.rotation.z = rotation;
  scene.add(mesh);
}

function addBarrierGate(name, x, z, color) {
  addCylinder(`${name}-post-1`, 0.13, 1.65, x - 2.2, 0.82, z, 0x334155);
  addCylinder(`${name}-post-2`, 0.13, 1.65, x + 2.2, 0.82, z, 0x334155);

  addBox(`${name}-barrier-arm`, 4.8, 0.16, 0.16, x, 1.42, z, color, true, true);

  for (let i = -2; i <= 2; i++) {
    addBox(`${name}-stripe-${i}`, 0.18, 0.19, 0.18, x + i * 0.85, 1.43, z, 0xffffff, true, true);
  }
}

function addSafetyBlocks() {
  const yellow = 0xfacc15;
  const dark = 0x111827;

  for (let x = -lotWidth / 2 + 5; x <= lotWidth / 2 - 5; x += 7) {
    addBox('front-barrier-yellow', 2.2, 0.35, 0.35, x, 0.22, frontZ + 5.6, yellow, true, true);
    addBox('front-barrier-black', 0.55, 0.38, 0.38, x + 0.3, 0.24, frontZ + 5.6, dark, true, true);
  }

  for (let z = backZ; z <= frontZ; z += 7) {
    addBox('left-side-barrier', 0.35, 0.42, 2.2, -lotWidth / 2 + 1.4, 0.25, z, yellow, true, true);
    addBox('right-side-barrier', 0.35, 0.42, 2.2, lotWidth / 2 - 1.4, 0.25, z, yellow, true, true);
  }
}

function buildBasement() {
  addBox('basement-floor', lotWidth, 0.18, frontZ - backZ + 9, 0, -0.1, (frontZ + backZ) / 2, 0xf8fafc);

  addBox('main-entry-road', 3.2, 0.05, frontZ - backZ + 4, ENTRY_LANE_X, 0.04, (frontZ + backZ) / 2, 0xcbd5e1);
  addBox('main-exit-road', 3.2, 0.05, frontZ - backZ + 4, EXIT_LANE_X, 0.04, (frontZ + backZ) / 2, 0xcbd5e1);
  addBox('front-entry-road', 12.5, 0.05, 4.4, ENTRY_GATE_X + 3.2, 0.05, frontZ + 2.5, 0xb7c3d0);
  addBox('front-exit-road', 12.5, 0.05, 4.4, EXIT_GATE_X - 3.2, 0.05, frontZ + 2.5, 0xb7c3d0);

  addBox('back-wall', lotWidth, 2.0, 0.35, 0, 1, backZ - 2, 0xe2e8f0);
  addBox('left-wall', 0.35, 2.0, frontZ - backZ + 8, -lotWidth / 2, 1, (frontZ + backZ) / 2, 0xe2e8f0);
  addBox('right-wall', 0.35, 2.0, frontZ - backZ + 8, lotWidth / 2, 1, (frontZ + backZ) / 2, 0xe2e8f0);

  for (let z = minZ - 2; z <= maxZ + 2; z += 9.2) {
    addCylinder('pillar-entry-side', 0.36, 3.1, -CENTER_AISLE / 2 - 0.85, 1.55, z, 0x94a3b8);
    addCylinder('pillar-exit-side', 0.36, 3.1, CENTER_AISLE / 2 + 0.85, 1.55, z, 0x94a3b8);
  }

  makeSign('ENTRY', '#16a34a', ENTRY_GATE_X, frontZ + 5.2);
  makeSign('EXIT', '#dc2626', EXIT_GATE_X, frontZ + 5.2);

  addBarrierGate('entry-gate', ENTRY_GATE_X, frontZ + 1.2, 0x16a34a);
  addBarrierGate('exit-gate', EXIT_GATE_X, frontZ + 1.2, 0xdc2626);

  for (let z = frontZ - 1; z >= minZ - 1; z -= 7.4) {
    addFloorArrow('IN', ENTRY_LANE_X, z, Math.PI, '#16a34a');
  }

  for (let z = minZ - 1; z <= frontZ - 1; z += 7.4) {
    addFloorArrow('OUT', EXIT_LANE_X, z, 0, '#dc2626');
  }

  addSafetyBlocks();
}

function getSlotX(col) {
  const sideIndex = col < HALF_COLS ? col : col - HALF_COLS;
  const sign = col < HALF_COLS ? -1 : 1;
  const localX = (sideIndex - (HALF_COLS - 1) / 2) * (SLOT_WIDTH + SLOT_GAP);
  const blockCenter = sign * (CENTER_AISLE / 2 + blockWidth / 2);
  return blockCenter + localX;
}

function getSlotZ(row) {
  return (row - (ROWS - 1) / 2) * (SLOT_LENGTH + ROW_GAP);
}

function buildSlots() {
  const slotGeometry = new THREE.BoxGeometry(SLOT_WIDTH, 0.05, SLOT_LENGTH);

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      const x = getSlotX(col);
      const z = getSlotZ(row);

      const slotBase = new THREE.Mesh(slotGeometry, mat(0xf1f5f9, 0.8, 0.02));
      slotBase.position.set(x, 0.06, z);
      slotBase.receiveShadow = true;
      scene.add(slotBase);

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(slotGeometry),
        new THREE.LineBasicMaterial({ color: 0x64748b })
      );
      edges.position.copy(slotBase.position);
      scene.add(edges);

      addBox('wheel-stop', 1.25, 0.16, 0.16, x, 0.18, z - SLOT_LENGTH * 0.38, 0x94a3b8, true, true);

      const sensor = new THREE.Mesh(
        new THREE.SphereGeometry(0.16, 18, 18),
        new THREE.MeshStandardMaterial({
          color: 0x22c55e,
          emissive: 0x22c55e,
          emissiveIntensity: 1.3
        })
      );
      sensor.position.set(x + SLOT_WIDTH * 0.32, 0.34, z - SLOT_LENGTH * 0.35);
      scene.add(sensor);

      slots.push({
        row,
        col,
        position: new THREE.Vector3(x, CAR_Y, z),
        occupied: false,
        reserved: false,
        car: null,
        sensor
      });
    }
  }
}

function setSensor(slot, occupied) {
  const color = occupied ? 0xef4444 : 0x22c55e;
  slot.sensor.material.color.setHex(color);
  slot.sensor.material.emissive.setHex(color);
}

function createCar() {
  carCounter += 1;
  const car = new THREE.Group();
  car.name = `car-${carCounter}`;

  const colors = [0x2563eb, 0xf97316, 0x14b8a6, 0xa855f7, 0xef4444, 0x0f172a, 0xeab308];
  const color = randomChoice(colors);

  const body = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.42, 2.15), mat(color, 0.45, 0.15));
  body.position.y = 0.3;
  body.castShadow = true;
  body.receiveShadow = true;
  car.add(body);

  const cabin = new THREE.Mesh(new THREE.BoxGeometry(0.86, 0.36, 0.88), mat(0xdbeafe, 0.25, 0.05));
  cabin.position.y = 0.68;
  cabin.position.z = -0.08;
  cabin.castShadow = true;
  car.add(cabin);

  const wheelGeometry = new THREE.CylinderGeometry(0.21, 0.21, 0.24, 18);
  const wheelMaterial = mat(0x111827, 0.8, 0.05);

  [[-0.68, 0.18, -0.68], [0.68, 0.18, -0.68], [-0.68, 0.18, 0.68], [0.68, 0.18, 0.68]].forEach((pos) => {
    const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
    wheel.rotation.z = Math.PI / 2;
    wheel.position.set(pos[0], pos[1], pos[2]);
    wheel.castShadow = true;
    car.add(wheel);
  });

  return car;
}

function parkInitialCars() {
  const startingCars = Math.floor(TOTAL_SLOTS * 0.28);

  for (let i = 0; i < startingCars; i++) {
    const freeSlots = slots.filter((slot) => !slot.occupied && !slot.reserved);
    const slot = randomChoice(freeSlots);
    const car = createCar();

    car.position.copy(slot.position);
    car.rotation.y = 0;
    scene.add(car);

    slot.occupied = true;
    slot.car = car;
    setSensor(slot, true);
  }
}

function routeBusy() {
  return activeMovers.length > 0;
}

function spawnCar() {
  if (routeBusy()) return;

  totalArrived += 1;

  const freeSlots = slots.filter((slot) => !slot.occupied && !slot.reserved);

  if (freeSlots.length === 0) {
    totalRejected += 1;
    updateStats();
    return;
  }

  const slot = randomChoice(freeSlots);
  slot.reserved = true;

  const car = createCar();
  car.position.set(ENTRY_GATE_X, CAR_Y, frontZ + 7.5);
  scene.add(car);

  const path = [
    car.position.clone(),
    new THREE.Vector3(ENTRY_GATE_X, CAR_Y, frontZ + 2.4),
    new THREE.Vector3(ENTRY_LANE_X, CAR_Y, frontZ),
    new THREE.Vector3(ENTRY_LANE_X, CAR_Y, slot.position.z),
    slot.position.clone()
  ];

  activeMovers.push({
    type: 'enter',
    mesh: car,
    slot,
    path,
    index: 1,
    speed: 10.2,
    startTime: simTime
  });
}

function departCar() {
  if (routeBusy()) return;

  const occupiedSlots = slots.filter((slot) => slot.occupied && slot.car && !slot.reserved);

  if (occupiedSlots.length === 0) return;

  const slot = randomChoice(occupiedSlots);
  const car = slot.car;

  slot.occupied = false;
  slot.car = null;
  setSensor(slot, false);

  const path = [
    car.position.clone(),
    new THREE.Vector3(EXIT_LANE_X, CAR_Y, slot.position.z),
    new THREE.Vector3(EXIT_LANE_X, CAR_Y, frontZ),
    new THREE.Vector3(EXIT_GATE_X, CAR_Y, frontZ + 2.4),
    new THREE.Vector3(EXIT_GATE_X, CAR_Y, frontZ + 7.5)
  ];

  activeMovers.push({
    type: 'exit',
    mesh: car,
    slot,
    path,
    index: 1,
    speed: 10.8
  });
}

function updateMover(mover, dt) {
  const target = mover.path[mover.index];
  const direction = target.clone().sub(mover.mesh.position);
  const distance = direction.length();

  if (distance < 0.08) {
    mover.index += 1;

    if (mover.index >= mover.path.length) {
      finishMover(mover);
      return true;
    }

    return false;
  }

  const step = Math.min(distance, mover.speed * dt);
  direction.normalize();
  mover.mesh.position.add(direction.multiplyScalar(step));

  const next = target.clone().sub(mover.mesh.position);
  if (next.length() > 0.001) {
    mover.mesh.rotation.y = Math.atan2(next.x, next.z);
  }

  return false;
}

function finishMover(mover) {
  if (mover.type === 'enter') {
    mover.slot.occupied = true;
    mover.slot.reserved = false;
    mover.slot.car = mover.mesh;
    mover.mesh.position.copy(mover.slot.position);
    mover.mesh.rotation.y = 0;
    setSensor(mover.slot, true);

    totalParked += 1;
    waitTimes.push(simTime - mover.startTime);
  }

  if (mover.type === 'exit') {
    scene.remove(mover.mesh);
    totalExited += 1;
  }
}

function calculateDemand() {
  // Demand updates only every 6 seconds, not every frame.
  // It changes based on occupancy rate, so it stays stable.
  if (simTime < nextDemandCheckAt) return;

  nextDemandCheckAt = simTime + 6;

  const occupied = slots.filter((slot) => slot.occupied).length;
  const occupancyRate = occupied / TOTAL_SLOTS;

  // Stable demand levels based on occupancy rate:
  // Low: below 35%
  // Medium: 35% to 70%
  // High: above 70%
  if (occupancyRate >= 0.70) {
    demandLabelStable = 'High';
    currentDemand = 0.82;
  } else if (occupancyRate >= 0.35) {
    demandLabelStable = 'Medium';
    currentDemand = 0.55;
  } else {
    demandLabelStable = 'Low';
    currentDemand = 0.28;
  }
}

function updateSimulation(dt) {
  simTime += dt;
  calculateDemand();

  const spawnInterval = 1.2 + (1 - currentDemand) * 1.8;
  const departureInterval = 1.8 + currentDemand * 2.0;

  if (simTime - lastSpawn > spawnInterval) {
    spawnCar();
    lastSpawn = simTime;
  }

  if (simTime - lastDeparture > departureInterval) {
    departCar();
    lastDeparture = simTime;
  }

  for (let i = activeMovers.length - 1; i >= 0; i--) {
    const finished = updateMover(activeMovers[i], dt);
    if (finished) {
      activeMovers.splice(i, 1);
    }
  }

  updateStats();
}

function updateStats() {
  const occupied = slots.filter((slot) => slot.occupied).length;
  const occupancyRate = Math.round((occupied / TOTAL_SLOTS) * 100);
  const avgWait = waitTimes.length
    ? (waitTimes.reduce((sum, value) => sum + value, 0) / waitTimes.length).toFixed(1)
    : '0';

  const demandLabel = demandLabelStable;

  document.getElementById('occupied').textContent = `${occupied}/${TOTAL_SLOTS}`;
  document.getElementById('occupancyRate').textContent = `${occupancyRate}%`;
  document.getElementById('arrived').textContent = totalArrived;
  document.getElementById('parked').textContent = totalParked;
  document.getElementById('exited').textContent = totalExited;
  document.getElementById('rejected').textContent = totalRejected;
  document.getElementById('avgWait').textContent = `${avgWait}s`;
  document.getElementById('demand').textContent = demandLabel;
}

function resetSimulation() {
  activeMovers.forEach((mover) => scene.remove(mover.mesh));
  activeMovers.length = 0;

  slots.forEach((slot) => {
    if (slot.car) scene.remove(slot.car);

    slot.occupied = false;
    slot.reserved = false;
    slot.car = null;
    setSensor(slot, false);
  });

  simTime = 0;
  lastSpawn = 0;
  lastDeparture = 0;
  totalArrived = 0;
  totalParked = 0;
  totalExited = 0;
  totalRejected = 0;
  waitTimes = [];
  currentDemand = 0.45;
  demandLabelStable = 'Medium';
  nextDemandCheckAt = 0;

  parkInitialCars();
  updateStats();
}

document.getElementById('panelToggle').addEventListener('click', () => {
  const hud = document.getElementById('hud');
  const closed = hud.classList.toggle('closed');
  document.getElementById('panelToggle').textContent = closed ? '▶' : '◀';
});

document.getElementById('pauseBtn').addEventListener('click', () => {
  paused = !paused;
  document.getElementById('pauseBtn').textContent = paused ? 'Resume' : 'Pause';
});

document.getElementById('addCarBtn').addEventListener('click', () => {
  spawnCar();
  updateStats();
});

document.getElementById('resetBtn').addEventListener('click', () => {
  resetSimulation();
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

buildBasement();
buildSlots();
parkInitialCars();
updateStats();

function animate() {
  requestAnimationFrame(animate);

  const dt = Math.min(clock.getDelta(), 0.05);

  if (!paused) updateSimulation(dt);

  controls.update();
  renderer.render(scene, camera);
}

animate();
