/**
 * app.js - Reusable 3D Simulation Viewer & Interactive Diagnostic Cockpit.
 * 
 * Synchronized with PyBullet CorridorSimEnv via REST API.
 * Uses Three.js for interactive 3D rendering, trajectory trail, raycasts,
 * and live contract-compliant diagnostic telemetry inspection.
 */

// Global State
let scene, camera, renderer, controls;
let droneMesh, corridorGroup, obstaclesGroup, trajectoryLine, raysGroup, goalMesh;
let sceneSpec = null;
let activeControllerId = "ppo_fine_tuned";
let isReplayMode = false;
let replaySnapshots = [];
let pollInterval = null;
let currentCameraMode = "orbit";
let hasPausedForObs3 = false;
let lastEpisodeStep = 0;
let activeEpisodeSeed = 50001;
let currentResetSeq = 0;
let isResetting = false;
let lastEpisodeStatus = "running";

const container = document.getElementById('viewport-container');

// Raycast beam labels matching sim/corridor_env.py
const RAY_NAMES = [
    "Fwd (0°)", "FL15 (+15°)", "FR15 (-15°)", "FL30 (+30°)", "FR30 (-30°)",
    "Left (+90°)", "Right (-90°)", "Back (180°)", "BL (+135°)", "BR (-135°)",
    "Up (+Z)", "Down (-Z)"
];

// Initialize 3D Viewport
function initThree() {
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0e14);
    scene.fog = new THREE.FogExp2(0x0c0e14, 0.035);

    camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.set(-2.5, -4.5, 3.5);
    camera.up.set(0, 0, 1); // Z is Up

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(5.0, 0.0, 1.0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight.position.set(5, -5, 10);
    scene.add(dirLight);

    // Groups
    corridorGroup = new THREE.Group();
    obstaclesGroup = new THREE.Group();
    raysGroup = new THREE.Group();
    scene.add(corridorGroup);
    scene.add(obstaclesGroup);
    scene.add(raysGroup);

    // Drone Mesh
    droneMesh = createDroneMesh();
    scene.add(droneMesh);

    // Trajectory Line
    const maxTrailPoints = 1200;
    const trailPositions = new Float32Array(maxTrailPoints * 3);
    const trailGeometry = new THREE.BufferGeometry();
    trailGeometry.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
    const trailMaterial = new THREE.LineBasicMaterial({
        color: 0x38bdf8,
        linewidth: 2,
        transparent: true,
        opacity: 0.85
    });
    trajectoryLine = new THREE.Line(trailGeometry, trailMaterial);
    scene.add(trajectoryLine);

    initRangefinderGrid();

    window.addEventListener('resize', onWindowResize);
    animate();
}

function createDroneMesh() {
    const group = new THREE.Group();

    // 1. Center Fuselage & Avionics Plate (Raspberry Pi 5 + RadioLink CrossFlight)
    const bodyGeo = new THREE.CylinderGeometry(0.10, 0.12, 0.07, 16);
    bodyGeo.rotateX(Math.PI / 2);
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.3, metalness: 0.8 });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    group.add(body);

    // 2. Sensor Turret / LiDAR Dome
    const turretGeo = new THREE.CylinderGeometry(0.04, 0.045, 0.03, 16);
    turretGeo.rotateX(Math.PI / 2);
    const turretMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.2, metalness: 0.9 });
    const turret = new THREE.Mesh(turretGeo, turretMat);
    turret.position.set(0.0, 0.0, 0.045);
    group.add(turret);

    // 3. Forward Heading Indicator Arrow
    const arrowGeo = new THREE.ConeGeometry(0.06, 0.18, 8);
    arrowGeo.rotateZ(-Math.PI / 2);
    arrowGeo.translate(0.18, 0, 0);
    const arrowMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const arrow = new THREE.Mesh(arrowGeo, arrowMat);
    group.add(arrow);

    // 4. Quadcopter Arms (Diagonal = 0.46m motor-to-motor, F450 frame)
    const armGeo = new THREE.BoxGeometry(0.44, 0.024, 0.02);
    const armMat = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.5 });
    const arm1 = new THREE.Mesh(armGeo, armMat);
    arm1.rotation.z = Math.PI / 4;
    const arm2 = new THREE.Mesh(armGeo, armMat);
    arm2.rotation.z = -Math.PI / 4;
    group.add(arm1);
    group.add(arm2);

    // 5. 4 Rotor Motors and Propeller Discs (Tip-to-tip diagonal = 0.65m)
    // Motor center offset = [+/-0.150, +/-0.150] -> distance = 0.212m
    // Propeller radius = 0.113m -> Tip radius = 0.212 + 0.113 = 0.325m (0.65m diagonal)
    const motorGeo = new THREE.CylinderGeometry(0.02, 0.02, 0.025, 12);
    motorGeo.rotateX(Math.PI / 2);
    const motorMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.9 });

    const rotorGeo = new THREE.CylinderGeometry(0.113, 0.113, 0.004, 16);
    rotorGeo.rotateX(Math.PI / 2);
    const rotorMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.35 });

    const offsets = [
        [0.150, 0.150, 0.025],
        [-0.150, 0.150, 0.025],
        [0.150, -0.150, 0.025],
        [-0.150, -0.150, 0.025]
    ];
    offsets.forEach(off => {
        const motor = new THREE.Mesh(motorGeo, motorMat);
        motor.position.set(off[0], off[1], off[2] - 0.012);
        group.add(motor);

        const rotor = new THREE.Mesh(rotorGeo, rotorMat);
        rotor.position.set(off[0], off[1], off[2]);
        group.add(rotor);
    });

    // 6. Landing Skids (Total height = 0.22m)
    const skidGeo = new THREE.CylinderGeometry(0.006, 0.006, 0.24, 8);
    const skidMat = new THREE.MeshStandardMaterial({ color: 0x334155 });
    const skid1 = new THREE.Mesh(skidGeo, skidMat);
    skid1.position.set(0.0, 0.10, -0.06);
    skid1.rotation.y = Math.PI / 2;
    const skid2 = new THREE.Mesh(skidGeo, skidMat);
    skid2.position.set(0.0, -0.10, -0.06);
    skid2.rotation.y = Math.PI / 2;
    group.add(skid1);
    group.add(skid2);

    // 7. Physical Collision Envelope (Conservative R = 0.325m bounding sphere)
    const sphereGeo = new THREE.SphereGeometry(0.325, 16, 16);
    const sphereMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        wireframe: true,
        transparent: true,
        opacity: 0.12
    });
    const collisionSphere = new THREE.Mesh(sphereGeo, sphereMat);
    group.add(collisionSphere);

    return group;
}

function onWindowResize() {
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

function animate() {
    requestAnimationFrame(animate);

    // If chase cam is active, update camera position relative to drone
    if (currentCameraMode === "chase" && droneMesh) {
        const dPos = droneMesh.position;
        camera.position.set(dPos.x - 2.0, dPos.y, dPos.z + 0.8);
        controls.target.set(dPos.x + 2.0, dPos.y, dPos.z);
    }

    controls.update();
    renderer.render(scene, camera);
}

// Build 3D Corridor from Exact Simulator Specs
function buildCorridorFromSpec(spec) {
    sceneSpec = spec;
    while (corridorGroup.children.length > 0) corridorGroup.remove(corridorGroup.children[0]);
    while (obstaclesGroup.children.length > 0) obstaclesGroup.remove(obstaclesGroup.children[0]);

    const c = spec.corridor;
    const len = c.x_end - c.x_start;
    const centerX = (c.x_start + c.x_end) / 2;
    const centerY = c.y_center;
    const centerZ = (c.z_floor + c.z_ceiling) / 2;

    // Floor wireframe / grid
    const floorGeo = new THREE.PlaneGeometry(len, c.width, Math.round(len), Math.round(c.width));
    const floorMat = new THREE.MeshBasicMaterial({
        color: 0x1e293b,
        wireframe: true,
        transparent: true,
        opacity: 0.4
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.position.set(centerX, centerY, c.z_floor);
    corridorGroup.add(floor);

    // Ceiling wireframe
    const ceilGeo = new THREE.PlaneGeometry(len, c.width, Math.round(len / 2), Math.round(c.width / 2));
    const ceilMat = new THREE.MeshBasicMaterial({
        color: 0x1e293b,
        wireframe: true,
        transparent: true,
        opacity: 0.2
    });
    const ceil = new THREE.Mesh(ceilGeo, ceilMat);
    ceil.position.set(centerX, centerY, c.z_ceiling);
    corridorGroup.add(ceil);

    // Left Wall (+Y)
    const wallGeo = new THREE.BoxGeometry(len, 0.04, c.height);
    const wallMat = new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        transparent: true,
        opacity: 0.35,
        roughness: 0.5
    });
    const leftWall = new THREE.Mesh(wallGeo, wallMat);
    leftWall.position.set(centerX, c.y_max, centerZ);
    corridorGroup.add(leftWall);

    // Right Wall (-Y)
    const rightWall = new THREE.Mesh(wallGeo, wallMat);
    rightWall.position.set(centerX, c.y_min, centerZ);
    corridorGroup.add(rightWall);

    // Bounding Box outline
    const boxGeo = new THREE.BoxGeometry(len, c.width, c.height);
    const edgesGeo = new THREE.EdgesGeometry(boxGeo);
    const edgesMat = new THREE.LineBasicMaterial({ color: 0x334155, linewidth: 1 });
    const outline = new THREE.LineSegments(edgesGeo, edgesMat);
    outline.position.set(centerX, centerY, centerZ);
    corridorGroup.add(outline);

    // Goal Destination Pad
    if (goalMesh) scene.remove(goalMesh);
    const goalGeo = new THREE.SphereGeometry(c.goal_radius, 16, 16);
    const goalMat = new THREE.MeshBasicMaterial({
        color: 0x10b981,
        wireframe: true,
        transparent: true,
        opacity: 0.6
    });
    goalMesh = new THREE.Mesh(goalGeo, goalMat);
    goalMesh.position.set(c.goal_position[0], c.goal_position[1], c.goal_position[2]);
    scene.add(goalMesh);

    // Build Obstacles
    const privObsList = document.getElementById('priv-obstacles-list');
    if (privObsList) privObsList.innerHTML = '';

    spec.obstacles.forEach((obs, idx) => {
        const cylGeo = new THREE.CylinderGeometry(obs.radius, obs.radius, obs.height, 24);
        cylGeo.rotateX(Math.PI / 2); // Cylinder along Z
        const cylMat = new THREE.MeshStandardMaterial({
            color: idx === 2 ? 0xf43f5e : 0xe11d48, // Highlight Obstacle 3
            roughness: 0.4,
            transparent: true,
            opacity: 0.85
        });
        const cyl = new THREE.Mesh(cylGeo, cylMat);
        cyl.position.set(obs.center[0], obs.center[1], obs.center[2]);
        obstaclesGroup.add(cyl);

        // Add to Privileged list
        if (privObsList) {
            const item = document.createElement('div');
            item.className = 'priv-obs-item';
            item.innerHTML = `
                <span><strong>Obstacle ${idx + 1}:</strong> Center [${obs.center[0].toFixed(2)}, ${obs.center[1].toFixed(2)}, ${obs.center[2].toFixed(2)}]m</span>
                <span class="mono">R=${obs.radius.toFixed(2)}m, H=${obs.height.toFixed(2)}m</span>
            `;
            privObsList.appendChild(item);
        }
    });
}

// Update Rangefinder Raycasts in 3D
function updateRaycasts(rays) {
    while (raysGroup.children.length > 0) raysGroup.remove(raysGroup.children[0]);
    if (!rays || rays.length === 0) return;

    rays.forEach((r, idx) => {
        const points = [
            new THREE.Vector3(r.start[0], r.start[1], r.start[2]),
            new THREE.Vector3(r.end[0], r.end[1], r.end[2])
        ];
        const rayGeo = new THREE.BufferGeometry().setFromPoints(points);
        let color = 0x38bdf8;
        if (r.distance_m < 0.35) color = 0xf43f5e;
        else if (r.distance_m < 0.80) color = 0xf59e0b;

        const rayMat = new THREE.LineBasicMaterial({
            color: color,
            linewidth: 1,
            transparent: true,
            opacity: 0.65
        });
        const line = new THREE.Line(rayGeo, rayMat);
        raysGroup.add(line);

        // Update rangefinder bar in Tab 2
        updateRangefinderBar(idx, r.distance_m);
    });
}

function initRangefinderGrid() {
    const grid = document.getElementById('rays-grid');
    if (!grid) return;
    grid.innerHTML = '';
    RAY_NAMES.forEach((name, i) => {
        const card = document.createElement('div');
        card.className = 'ray-bar-card';
        card.innerHTML = `
            <div class="ray-bar-header">
                <span class="ray-name">${name}</span>
                <span class="ray-val" id="ray-val-${i}">4.00 m</span>
            </div>
            <div class="ray-track">
                <div class="ray-fill" id="ray-fill-${i}" style="width: 100%;"></div>
            </div>
        `;
        grid.appendChild(card);
    });
}

function updateRangefinderBar(idx, distM) {
    const valSpan = document.getElementById(`ray-val-${idx}`);
    const fillBar = document.getElementById(`ray-fill-${idx}`);
    if (!valSpan || !fillBar) return;

    valSpan.textContent = `${distM.toFixed(2)} m`;
    const pct = Math.min(100, Math.max(0, (distM / 4.0) * 100));
    fillBar.style.width = `${pct}%`;

    fillBar.className = 'ray-fill';
    if (distM < 0.35) {
        fillBar.classList.add('close');
        valSpan.style.color = '#f43f5e';
    } else if (distM < 0.80) {
        fillBar.classList.add('caution');
        valSpan.style.color = '#f59e0b';
    } else {
        valSpan.style.color = '#38bdf8';
    }
}

// Update Trajectory Trail
function updateTrajectory(trail) {
    if (!trail || trail.length === 0) return;
    const positions = trajectoryLine.geometry.attributes.position.array;
    let idx = 0;
    for (let i = 0; i < trail.length && idx < positions.length - 3; i++) {
        positions[idx++] = trail[i][0];
        positions[idx++] = trail[i][1];
        positions[idx++] = trail[i][2];
    }
    trajectoryLine.geometry.setDrawRange(0, trail.length);
    trajectoryLine.geometry.attributes.position.needsUpdate = true;
}

// Update HUD & Diagnostic Telemetry Readouts
function updateHUD(snap) {
    if (snap.seed !== undefined) {
        updateActiveSeedDisplay(snap.seed);
    }
    lastEpisodeStatus = snap.episode_status;

    // Top HUD
    document.getElementById('hud-time').textContent = `${snap.sim_time_s.toFixed(2)} s`;
    document.getElementById('hud-step').textContent = snap.step;
    document.getElementById('hud-obs-clearance').textContent = `${snap.obstacle_clearance.toFixed(2)} m`;
    document.getElementById('hud-wall-clearance').textContent = `${snap.boundary_distance.toFixed(2)} m`;
    document.getElementById('hud-speed').textContent = `${snap.speed_mps.toFixed(2)} m/s`;
    document.getElementById('hud-goal-dist').textContent = `${snap.goal_distance.toFixed(2)} m`;
    document.getElementById('hud-interventions').textContent = snap.safety_interventions_count || 0;

    // Status Pill
    const statusPill = document.getElementById('hud-status');
    statusPill.textContent = snap.episode_status.replace('_', ' ').toUpperCase();
    statusPill.className = `status-pill ${snap.episode_status}`;

    // Safety Supervisor Pill
    const safetyPill = document.getElementById('hud-safety');
    safetyPill.textContent = snap.safety_state.replace('_', ' ').toUpperCase();
    safetyPill.className = `status-pill ${snap.safety_state}`;

    // Active Controller Badge
    const ctrlBadge = document.getElementById('active-ctrl-badge');
    if (ctrlBadge && snap.controller_display_name) {
        ctrlBadge.textContent = snap.controller_display_name;
    }

    // Drone 3D position & rotation
    droneMesh.position.set(snap.position[0], snap.position[1], snap.position[2]);
    droneMesh.rotation.set(snap.orientation_euler[0], snap.orientation_euler[1], snap.orientation_euler[2]);

    // TAB 1: Actions & Kinematics
    if (snap.proposed_action) {
        document.getElementById('act-prop-vx').textContent = `${snap.proposed_action.vx.toFixed(2)} m/s`;
        document.getElementById('act-prop-vy').textContent = `${snap.proposed_action.vy.toFixed(2)} m/s`;
        document.getElementById('act-prop-vz').textContent = `${snap.proposed_action.vz.toFixed(2)} m/s`;
        document.getElementById('act-prop-yaw').textContent = `${snap.proposed_action.yaw_rate.toFixed(2)} rad/s`;
    }
    if (snap.applied_action) {
        document.getElementById('act-appl-vx').textContent = `${snap.applied_action.vx.toFixed(2)} m/s`;
        document.getElementById('act-appl-vy').textContent = `${snap.applied_action.vy.toFixed(2)} m/s`;
        document.getElementById('act-appl-vz').textContent = `${snap.applied_action.vz.toFixed(2)} m/s`;
        document.getElementById('act-appl-yaw').textContent = `${snap.applied_action.yaw_rate.toFixed(2)} rad/s`;
    }
    if (snap.normalized_action) {
        document.getElementById('act-norm-vec').textContent = `[${snap.normalized_action.map(x => x.toFixed(2)).join(', ')}]`;
    }

    document.getElementById('tele-pos').textContent = `[${snap.position.map(x => x.toFixed(3)).join(', ')}] m`;
    if (snap.orientation_euler_deg) {
        document.getElementById('tele-euler').textContent = `[${snap.orientation_euler_deg.map(x => x.toFixed(1) + '°').join(', ')}]`;
    }
    document.getElementById('tele-vel').textContent = `[${snap.velocity.map(x => x.toFixed(3)).join(', ')}] m/s`;
    if (snap.goal_position) {
        document.getElementById('tele-goal').textContent = `[${snap.goal_position.map(x => x.toFixed(2)).join(', ')}] m`;
    }

    document.getElementById('tele-safety-state').textContent = snap.safety_state.toUpperCase();
    document.getElementById('tele-safety-state').className = `tele-val mono ${snap.safety_state === 'nominal' ? 'text-emerald' : 'text-rose'}`;
    document.getElementById('tele-safety-reason').textContent = snap.safety_reason || 'Nominal';
    document.getElementById('tele-safety-count').textContent = snap.safety_interventions_count || 0;
    document.getElementById('tele-flags-outcome').textContent = snap.episode_status.toUpperCase();

    // TAB 3: 23-Dim Contract Groups
    if (snap.observation_groups) {
        const og = snap.observation_groups;
        document.getElementById('obs-grp-rays').textContent = `[${og.distance_rays_norm.map(x => x.toFixed(3)).join(', ')}]`;
        document.getElementById('obs-grp-goal').textContent = `[${og.rel_goal_direction_body.map(x => x.toFixed(3)).join(', ')}]`;
        document.getElementById('obs-grp-vel').textContent = `[${og.body_velocity_mps.map(x => x.toFixed(3)).join(', ')}]`;
        document.getElementById('obs-grp-att').textContent = `[${og.attitude_trig.map(x => x.toFixed(3)).join(', ')}]`;
        document.getElementById('obs-grp-yaw').textContent = `${og.yaw_rate_radps.toFixed(3)} rad/s`;
    }
    if (snap.deployable_observation_23) {
        document.getElementById('obs-raw-full').textContent = `[${snap.deployable_observation_23.map(x => x.toFixed(3)).join(', ')}]`;
    }

    // TAB 4: Simulator Ground Truth (Privileged)
    if (snap.privileged_simulation_ground_truth) {
        const gt = snap.privileged_simulation_ground_truth;
        document.getElementById('priv-min-obs').textContent = `${gt.obstacle_clearance_m.toFixed(3)} m`;
        document.getElementById('priv-min-wall').textContent = `${gt.boundary_distance_m.toFixed(3)} m`;
        document.getElementById('priv-world-pos').textContent = `[${gt.world_position_m.map(x => x.toFixed(3)).join(', ')}] m`;
        document.getElementById('priv-world-vel').textContent = `[${gt.world_velocity_mps.map(x => x.toFixed(3)).join(', ')}] m/s`;
    }
}

// Camera Preset Views
function setCameraView(viewName) {
    currentCameraMode = viewName;
    document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById(`cam-${viewName}`);
    if (btn) btn.classList.add('active');

    if (!sceneSpec) return;
    const c = sceneSpec.corridor;
    const centerX = (c.x_start + c.x_end) / 2;

    if (viewName === 'orbit') {
        camera.position.set(-2.5, -4.5, 3.5);
        controls.target.set(centerX, 0.0, 1.0);
    } else if (viewName === 'entrance') {
        camera.position.set(-1.0, 0.0, 1.0);
        controls.target.set(3.0, 0.0, 1.0);
    } else if (viewName === 'topdown') {
        camera.position.set(centerX, 0.0, 8.5);
        controls.target.set(centerX, 0.0, 1.0);
    } else if (viewName === 'obs3') {
        // Focus on Obstacle 3 (X=7.0m in 10m scenario)
        camera.position.set(5.2, -2.4, 2.2);
        controls.target.set(7.0, -0.3, 1.0);
    }
}

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
    if (activeBtn) activeBtn.classList.add('active');

    const pane = document.getElementById(tabId);
    if (pane) pane.classList.add('active');
}

function updateActiveSeedDisplay(seed) {
    activeEpisodeSeed = seed;
    const hudSeed = document.getElementById('hud-active-seed');
    if (hudSeed) hudSeed.textContent = seed;
    const cardSeed = document.getElementById('active-seed-display');
    if (cardSeed) cardSeed.textContent = seed;
}

// Unified Simulation Reset Function
async function executeReset(requestedSeed = null) {
    let seedVal = requestedSeed;
    if (seedVal === null || seedVal === undefined) {
        const rawInput = document.getElementById('seed-input').value.trim();
        if (!rawInput || isNaN(rawInput)) {
            showBanner('⚠️ Error: Random Seed must be a valid positive integer.');
            return false;
        }
        seedVal = parseInt(rawInput, 10);
    }

    if (isNaN(seedVal) || seedVal < 1 || seedVal > 999999999) {
        showBanner(`⚠️ Error: Invalid seed ${seedVal}. Must be a positive integer between 1 and 999999999.`);
        return false;
    }

    const thisSeq = ++currentResetSeq;
    isResetting = true;

    try {
        await fetch('/api/pause', { method: 'POST' });
    } catch (e) {}

    isReplayMode = false;
    hasPausedForObs3 = false;
    hideBanner();
    document.getElementById('replay-status').textContent = 'Live Sim';
    document.getElementById('seed-input').value = seedVal;

    try {
        const res = await fetch('/api/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ seed: seedVal })
        });

        if (thisSeq !== currentResetSeq) {
            return false;
        }

        const data = await res.json();
        if (!res.ok) {
            showBanner(`⚠️ Reset Error: ${data.error || 'Server rejected reset'}`);
            return false;
        }

        updateActiveSeedDisplay(data.seed);
        await fetchScene();
        await pollState();
        return true;
    } catch (err) {
        showBanner(`⚠️ Reset failed: ${err.message}`);
        return false;
    } finally {
        if (thisSeq === currentResetSeq) {
            isResetting = false;
        }
    }
}

// Quick Seed Selection using unified executeReset
async function setSeed(seedVal) {
    await executeReset(seedVal);
}

function showBanner(text) {
    const banner = document.getElementById('sim-banner');
    const bannerText = document.getElementById('sim-banner-text');
    if (banner && bannerText) {
        bannerText.textContent = text;
        banner.classList.remove('hidden');
    }
}

function hideBanner() {
    const banner = document.getElementById('sim-banner');
    if (banner) banner.classList.add('hidden');
}

// API Communication
async function fetchScene() {
    try {
        const res = await fetch('/api/scene');
        const data = await res.json();
        buildCorridorFromSpec(data);
    } catch (err) {
        console.error('Failed to fetch scene:', err);
    }
}

async function fetchControllers() {
    try {
        const res = await fetch('/api/controllers');
        const data = await res.json();
        const sel = document.getElementById('controller-select');
        sel.innerHTML = '';

        data.controllers.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c.controller_id;
            opt.textContent = `${c.display_name} [${c.milestone_tag}]`;
            if (!c.is_implemented) opt.textContent += ' (Planned)';
            if (c.controller_id === data.active_controller_id) opt.selected = true;
            sel.appendChild(opt);
        });

        updateControllerMetadata(data.controllers, data.active_controller_id);
    } catch (err) {
        console.error('Failed to fetch controllers:', err);
    }
}

function updateControllerMetadata(controllers, activeId) {
    const c = controllers.find(item => item.controller_id === activeId);
    if (!c) return;
    document.getElementById('meta-checkpoint').textContent = c.checkpoint_id;
    const statusTag = document.getElementById('meta-status');
    statusTag.textContent = c.is_implemented ? 'Implemented' : 'Planned';
    statusTag.style.color = c.is_implemented ? '#10b981' : '#f59e0b';
    document.getElementById('meta-description').textContent = c.description;
}

async function pollState() {
    if (isReplayMode) return;
    try {
        const res = await fetch('/api/state');
        const snap = await res.json();
        updateHUD(snap);
        updateTrajectory(snap.trajectory_trail);
        updateRaycasts(snap.rays);

        // Update Replay Slider max
        const slider = document.getElementById('replay-slider');
        slider.max = snap.step;
        slider.value = snap.step;
        slider.disabled = snap.step === 0;
        document.getElementById('scrub-step-label').textContent = `Step ${snap.step}`;

        // Auto-pause before Obstacle 3 (X >= 5.95m)
        const chkObs3 = document.getElementById('chk-pause-obs3');
        if (chkObs3 && chkObs3.checked && !hasPausedForObs3 && snap.is_playing) {
            if (snap.position[0] >= 5.95 && snap.position[0] <= 6.5) {
                hasPausedForObs3 = true;
                await fetch('/api/pause', { method: 'POST' });
                showBanner(`⏸ Auto-Paused before Obstacle 3 (X=${snap.position[0].toFixed(2)}m) — Use 'Single Step' to inspect failure`);
            }
        }

        // Show banner on terminal state
        if (snap.episode_status !== 'running' && lastEpisodeStep !== snap.step) {
            lastEpisodeStep = snap.step;
            if (snap.episode_status === 'crash_obstacle') {
                showBanner(`💥 Collision with Obstacle at Step ${snap.step} (X=${snap.position[0].toFixed(2)}m)`);
            } else if (snap.episode_status === 'crash_boundary') {
                showBanner(`⚠️ Boundary Wall Collision at Step ${snap.step} (X=${snap.position[0].toFixed(2)}m)`);
            } else if (snap.episode_status === 'success') {
                showBanner(`🎯 Goal Reached Successfully in ${snap.sim_time_s.toFixed(2)}s!`);
            }
        }
    } catch (err) {
        // Server might be paused or resetting
    }
}

async function fetchEvaluationSummary() {
    try {
        const res = await fetch('/api/eval_summary');
        const data = await res.json();
        const badge = document.getElementById('eval-count-badge');
        badge.textContent = `${data.total_episodes || 0} Episodes`;

        if (!data.has_data) {
            document.getElementById('stat-success').textContent = '-';
            document.getElementById('stat-collision').textContent = '-';
            document.getElementById('stat-boundary').textContent = '-';
            document.getElementById('stat-time').textContent = '-';
            return;
        }

        document.getElementById('stat-success').textContent = `${data.success_rate_pct.toFixed(1)}%`;
        document.getElementById('stat-collision').textContent = `${data.collision_rate_pct.toFixed(1)}%`;
        document.getElementById('stat-boundary').textContent = `${data.boundary_violation_pct.toFixed(1)}%`;
        document.getElementById('stat-time').textContent = `${data.mean_flight_time_s.toFixed(2)}s`;

        // Populate Table
        const tbody = document.getElementById('eval-tbody');
        tbody.innerHTML = '';
        data.records.reverse().forEach(r => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="mono">#${r.episode_id}</td>
                <td class="mono">${r.seed}</td>
                <td><strong class="mono">${r.controller_id}</strong> <br><small class="mono text-muted">${r.checkpoint_id}</small></td>
                <td><span class="status-pill ${r.status}">${r.status.toUpperCase()}</span></td>
                <td class="mono">${r.steps}</td>
                <td class="mono">${r.flight_time_s.toFixed(2)}s</td>
                <td class="mono">${r.min_boundary_distance_m.toFixed(2)}m</td>
                <td class="mono">${(r.min_obstacle_clearance_m || 0).toFixed(2)}m</td>
                <td class="mono">${r.final_distance_to_goal_m.toFixed(2)}m</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error('Failed to fetch eval summary:', err);
    }
}

// Event Listeners
function setupEvents() {
    document.getElementById('btn-play').addEventListener('click', async () => {
        const rawVal = document.getElementById('seed-input').value.trim();
        const enteredSeed = parseInt(rawVal, 10);
        const needsReset = isNaN(enteredSeed) || (enteredSeed !== activeEpisodeSeed) || (lastEpisodeStatus !== 'running');
        if (needsReset) {
            const ok = await executeReset(isNaN(enteredSeed) ? null : enteredSeed);
            if (!ok) return;
        }
        isReplayMode = false;
        document.getElementById('replay-status').textContent = 'Live Sim';
        hideBanner();
        await fetch('/api/play', { method: 'POST' });
    });

    document.getElementById('btn-pause').addEventListener('click', async () => {
        await fetch('/api/pause', { method: 'POST' });
    });

    document.getElementById('btn-step').addEventListener('click', async () => {
        const rawVal = document.getElementById('seed-input').value.trim();
        const enteredSeed = parseInt(rawVal, 10);
        const needsReset = isNaN(enteredSeed) || (enteredSeed !== activeEpisodeSeed) || (lastEpisodeStatus !== 'running');
        if (needsReset) {
            const ok = await executeReset(isNaN(enteredSeed) ? null : enteredSeed);
            if (!ok) return;
        }
        isReplayMode = false;
        document.getElementById('replay-status').textContent = 'Live Sim';
        await fetch('/api/step', { method: 'POST' });
        pollState();
    });

    document.getElementById('btn-reset').addEventListener('click', async () => {
        await executeReset();
    });

    document.getElementById('speed-select').addEventListener('change', async (e) => {
        await fetch('/api/speed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ speed: parseFloat(e.target.value) })
        });
    });

    document.getElementById('scenario-select').addEventListener('change', async (e) => {
        const scenario = e.target.value;
        const res = await fetch('/api/select_scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario_id: scenario })
        });
        const data = await res.json();
        if (data.success) {
            hasPausedForObs3 = false;
            hideBanner();
            await fetchScene();
            pollState();
        } else {
            alert(data.message);
        }
    });

    document.getElementById('controller-select').addEventListener('change', async (e) => {
        const ctrlId = e.target.value;
        const res = await fetch('/api/select_controller', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ controller_id: ctrlId })
        });
        const data = await res.json();
        if (!data.success) {
            alert(data.message);
            fetchControllers(); // Revert
        } else {
            hasPausedForObs3 = false;
            fetchControllers();
            pollState();
        }
    });

    document.getElementById('btn-eval-batch').addEventListener('click', async () => {
        const btn = document.getElementById('btn-eval-batch');
        btn.disabled = true;
        btn.textContent = 'Running Batch (5 Seeds)...';
        try {
            await fetch('/api/eval_batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ seeds: [50000, 50001, 50002, 50003, 50004] })
            });
            await fetchEvaluationSummary();
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<span class="btn-icon">⚡</span> Run Held-Out Batch (5 Seeds)';
        }
    });

    // Replay Slider Scrubbing
    const slider = document.getElementById('replay-slider');
    slider.addEventListener('input', async (e) => {
        const step = parseInt(e.target.value, 10);
        document.getElementById('scrub-step-label').textContent = `Step ${step}`;
        if (!isReplayMode) {
            isReplayMode = true;
            document.getElementById('replay-status').textContent = 'Scrubbing Episode';
            await fetch('/api/pause', { method: 'POST' });
            const res = await fetch('/api/replay');
            const data = await res.json();
            replaySnapshots = data.snapshots || [];
        }

        if (replaySnapshots.length > step) {
            const snap = replaySnapshots[step];
            updateHUD(snap);
            updateRaycasts(snap.rays);
        }
    });
}

// Main Boot
window.addEventListener('DOMContentLoaded', async () => {
    initThree();
    setupEvents();
    await fetchScene();
    await fetchControllers();
    await fetchEvaluationSummary();

    // Start 20 Hz polling loop for synchronized real-time display
    pollInterval = setInterval(pollState, 50);
});
