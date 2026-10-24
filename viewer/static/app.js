/**
 * app.js - Reusable 3D Simulation Viewer & Telemetry HUD.
 * 
 * Synchronized with PyBullet CorridorSimEnv via REST API.
 * Uses Three.js for interactive 3D rendering, trajectory trail, and raycast visualization.
 */

// Global State
let scene, camera, renderer, controls;
let droneMesh, corridorGroup, obstaclesGroup, trajectoryLine, raysGroup, goalMesh;
let sceneSpec = null;
let activeControllerId = "nominal_baseline";
let isReplayMode = false;
let replaySnapshots = [];
let pollInterval = null;

const container = document.getElementById('viewport-container');

// Initialize 3D Viewport
function initThree() {
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0e14);
    scene.fog = new THREE.FogExp2(0x0c0e14, 0.04);

    camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    // Position camera slightly behind and above entrance, looking along +X axis
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
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
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
    const maxTrailPoints = 1000;
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

    window.addEventListener('resize', onWindowResize);
    animate();
}

function createDroneMesh() {
    const group = new THREE.Group();

    // Center body
    const bodyGeo = new THREE.CylinderGeometry(0.12, 0.14, 0.06, 16);
    bodyGeo.rotateX(Math.PI / 2);
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.3, metalness: 0.8 });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    group.add(body);

    // Forward arrow indicator
    const arrowGeo = new THREE.ConeGeometry(0.06, 0.16, 8);
    arrowGeo.rotateZ(-Math.PI / 2);
    arrowGeo.translate(0.16, 0, 0);
    const arrowMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const arrow = new THREE.Mesh(arrowGeo, arrowMat);
    group.add(arrow);

    // 4 Rotors
    const armGeo = new THREE.BoxGeometry(0.36, 0.02, 0.02);
    const armMat = new THREE.MeshBasicMaterial({ color: 0x64748b });
    const arm1 = new THREE.Mesh(armGeo, armMat);
    arm1.rotation.z = Math.PI / 4;
    const arm2 = new THREE.Mesh(armGeo, armMat);
    arm2.rotation.z = -Math.PI / 4;
    group.add(arm1);
    group.add(arm2);

    // Rotor discs
    const rotorGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.005, 12);
    rotorGeo.rotateX(Math.PI / 2);
    const rotorMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.4 });
    const offsets = [
        [0.13, 0.13, 0.02],
        [-0.13, 0.13, 0.02],
        [0.13, -0.13, 0.02],
        [-0.13, -0.13, 0.02]
    ];
    offsets.forEach(off => {
        const rotor = new THREE.Mesh(rotorGeo, rotorMat);
        rotor.position.set(off[0], off[1], off[2]);
        group.add(rotor);
    });

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
    spec.obstacles.forEach(obs => {
        const cylGeo = new THREE.CylinderGeometry(obs.radius, obs.radius, obs.height, 24);
        cylGeo.rotateX(Math.PI / 2); // Cylinder along Z
        const cylMat = new THREE.MeshStandardMaterial({
            color: 0xf43f5e,
            roughness: 0.4,
            transparent: true,
            opacity: 0.8
        });
        const cyl = new THREE.Mesh(cylGeo, cylMat);
        cyl.position.set(obs.center[0], obs.center[1], obs.center[2]);
        obstaclesGroup.add(cyl);
    });
}

// Update Rangefinder Raycasts in 3D
function updateRaycasts(rays) {
    while (raysGroup.children.length > 0) raysGroup.remove(raysGroup.children[0]);
    if (!rays || rays.length === 0) return;

    rays.forEach(r => {
        const points = [
            new THREE.Vector3(r.start[0], r.start[1], r.start[2]),
            new THREE.Vector3(r.end[0], r.end[1], r.end[2])
        ];
        const rayGeo = new THREE.BufferGeometry().setFromPoints(points);
        // Color based on clearance: Red if close (<0.4m), Green if clear (>1.5m)
        let color = 0x38bdf8;
        if (r.distance_m < 0.35) color = 0xf43f5e;
        else if (r.distance_m < 0.80) color = 0xf59e0b;

        const rayMat = new THREE.LineBasicMaterial({
            color: color,
            linewidth: 1,
            transparent: true,
            opacity: 0.6
        });
        const line = new THREE.Line(rayGeo, rayMat);
        raysGroup.add(line);
    });
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

// Update HUD & Telemetry Readouts
function updateHUD(snap) {
    document.getElementById('hud-time').textContent = `${snap.sim_time_s.toFixed(2)} s`;
    document.getElementById('hud-step').textContent = snap.step;
    document.getElementById('hud-clearance').textContent = `${snap.boundary_distance.toFixed(2)} m`;
    document.getElementById('hud-speed').textContent = `${snap.speed_mps.toFixed(2)} m/s`;
    document.getElementById('hud-goal-dist').textContent = `${snap.goal_distance.toFixed(2)} m`;

    // Status Pill
    const statusPill = document.getElementById('hud-status');
    statusPill.textContent = snap.episode_status.replace('_', ' ').toUpperCase();
    statusPill.className = `status-pill ${snap.episode_status}`;

    // Safety Supervisor Pill
    const safetyPill = document.getElementById('hud-safety');
    safetyPill.textContent = snap.safety_state.replace('_', ' ').toUpperCase();
    safetyPill.className = `status-pill ${snap.safety_state}`;

    // Update drone 3D position
    droneMesh.position.set(snap.position[0], snap.position[1], snap.position[2]);
    droneMesh.rotation.set(snap.orientation_euler[0], snap.orientation_euler[1], snap.orientation_euler[2]);
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
        document.getElementById('scrub-step-label').textContent = `Step ${snap.step}`;
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
        isReplayMode = false;
        document.getElementById('replay-status').textContent = 'Live Sim';
        await fetch('/api/play', { method: 'POST' });
    });

    document.getElementById('btn-pause').addEventListener('click', async () => {
        await fetch('/api/pause', { method: 'POST' });
    });

    document.getElementById('btn-step').addEventListener('click', async () => {
        isReplayMode = false;
        document.getElementById('replay-status').textContent = 'Live Sim';
        await fetch('/api/step', { method: 'POST' });
        pollState();
    });

    document.getElementById('btn-reset').addEventListener('click', async () => {
        isReplayMode = false;
        document.getElementById('replay-status').textContent = 'Live Sim';
        const seed = parseInt(document.getElementById('seed-input').value, 10);
        await fetch('/api/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ seed: seed })
        });
        await fetchScene();
        pollState();
    });

    document.getElementById('speed-select').addEventListener('change', async (e) => {
        await fetch('/api/speed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ speed: parseFloat(e.target.value) })
        });
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
            fetchControllers();
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
                body: JSON.stringify({ seeds: [1000, 1001, 1002, 1003, 1004] })
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
