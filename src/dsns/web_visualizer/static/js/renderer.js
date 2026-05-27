import * as THREE from 'three';

const PLANET_ALIGN_QUAT = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(1, 0, 0),
    -Math.PI / 2
);

export default class WebRenderer {
    constructor(canvas, initialState, spaceScale) {
        this.canvas = canvas;
        this.spaceScale = spaceScale;
        this.width = canvas.clientWidth;
        this.height = canvas.clientHeight;
        this.satelliteSize = 0.15;
        this.near = 0.01;
        this.far = 10000;
        this.bgColor = 0x050510;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.bgColor);
        this.scene.fog = new THREE.FogExp2(this.bgColor, 0.002);

        this.rootGroup = new THREE.Group();
        this.rootGroup.rotation.x = Math.PI / 2;
        this.scene.add(this.rootGroup);

        this.camera = new THREE.PerspectiveCamera(
            45, this.width / this.height, this.near, this.far
        );
        this.camera.position.set(20, 20, 20);
        this.camera.lookAt(0, 0, 0);

        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        this.controls = null;

        this.stars = this._buildStars(2000);
        this.scene.add(this.stars);

        // Soft multi-source lighting (no shadow terminator)
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.4);
        hemiLight.position.set(0, 20, 0);
        this.scene.add(hemiLight);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.15);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
        dirLight.position.set(5, 10, 7);
        this.scene.add(dirLight);

        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        dirLight2.position.set(-5, 5, -7);
        this.scene.add(dirLight2);

        const dirLight3 = new THREE.DirectionalLight(0xffffff, 0.2);
        dirLight3.position.set(0, -5, 5);
        this.scene.add(dirLight3);

        this._initFromState(initialState);
        this._resizeHandler = () => this._onResize();
        window.addEventListener("resize", this._resizeHandler);
    }

    async loadControls() {
        const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    _initFromState(state) {
        this._buildSatellites(state);
        this._buildLinks(state);
        this._buildPlanets(state);
        this._updateFromState(state);
    }

    rebuildFromState(state) {
        // Dispose old satellite instanced mesh
        if (this.satInstancedMesh) {
            this.satInstancedMesh.dispose();
            this.rootGroup.remove(this.satInstancedMesh);
        }
        // Dispose old link geometries
        if (this.islLineSegments) {
            this.islLineSegments.geometry.dispose();
            this.rootGroup.remove(this.islLineSegments);
        }
        if (this.illLineSegments) {
            this.illLineSegments.geometry.dispose();
            this.rootGroup.remove(this.illLineSegments);
        }
        // Dispose old planet meshes
        for (const mesh of this.planetMeshes) {
            mesh.geometry.dispose();
            if (mesh.material.map) mesh.material.map.dispose();
            mesh.material.dispose();
            this.rootGroup.remove(mesh);
        }
        this.planets = [];
        this.planetMeshes = [];

        this._initFromState(state);
    }

    _buildSatellites(state) {
        const numSats = state.num_sats;
        this.satGeometry = new THREE.SphereGeometry(1, 12, 8);
        const satMaterial = new THREE.MeshStandardMaterial({
            color: new THREE.Color(...state.sat_color),
            roughness: 0.5,
            metalness: 0.5,
        });
        this.satInstancedMesh = new THREE.InstancedMesh(
            this.satGeometry, satMaterial, numSats
        );
        this.satInstancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        this.rootGroup.add(this.satInstancedMesh);
    }

    _buildLinks(state) {
        this.islLineMaterial = new THREE.LineBasicMaterial({
            color: new THREE.Color(...state.isl_color),
            linewidth: 1,
        });
        this.illLineMaterial = new THREE.LineBasicMaterial({
            color: new THREE.Color(...state.ill_color),
            linewidth: 1,
        });

        // ISL lines
        const maxIsls = Math.max(64, state.num_isls);
        const islGeometry = new THREE.BufferGeometry();
        const islPositions = new Float32Array(maxIsls * 6);
        islGeometry.setAttribute("position", new THREE.BufferAttribute(islPositions, 3));
        this.islLineSegments = new THREE.LineSegments(islGeometry, this.islLineMaterial);
        this.islLineSegments.frustumCulled = false;
        this.rootGroup.add(this.islLineSegments);

        // ILL lines
        const maxIlls = Math.max(64, state.num_ills);
        const illGeometry = new THREE.BufferGeometry();
        const illPositions = new Float32Array(maxIlls * 6);
        illGeometry.setAttribute("position", new THREE.BufferAttribute(illPositions, 3));
        this.illLineSegments = new THREE.LineSegments(illGeometry, this.illLineMaterial);
        this.illLineSegments.frustumCulled = false;
        this.rootGroup.add(this.illLineSegments);
    }

    _buildPlanets(state) {
        this.planets = [];
        this.planetMeshes = [];
        const planets = state.planets;
        for (const p of planets) {
            const geometry = new THREE.SphereGeometry(p.radius, 32, 16);
            let material;
            if (p.texture) {
                const textureLoader = new THREE.TextureLoader();
                textureLoader.setCrossOrigin("anonymous");
                const texture = textureLoader.load(p.texture);
                material = new THREE.MeshStandardMaterial({
                    map: texture,
                    roughness: 0.8,
                    metalness: 0.1,
                });
            } else {
                material = new THREE.MeshStandardMaterial({
                    color: new THREE.Color(...p.color),
                    roughness: 0.7,
                    metalness: 0.3,
                });
            }
            const mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            mesh.quaternion.copy(PLANET_ALIGN_QUAT);
            this.rootGroup.add(mesh);
            this.planets.push(p);
            this.planetMeshes.push(mesh);
        }
    }

    _buildStars(count) {
        const positions = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = 500 + Math.random() * 500;
            positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            positions[i * 3 + 2] = r * Math.cos(phi);
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        const material = new THREE.PointsMaterial({
            color: 0xffffff, size: 1.5, sizeAttenuation: false, fog: false,
        });
        return new THREE.Points(geometry, material);
    }

    /**
     * Update satellite positions from a flat Float32Array.
     * Used when we have a single state (no interpolation).
     */
    updateSatellites(positions) {
        const dummy = new THREE.Object3D();
        const numSats = this.satInstancedMesh.count;
        for (let i = 0; i < numSats; i++) {
            dummy.position.set(
                positions[i * 3],
                positions[i * 3 + 1],
                positions[i * 3 + 2]
            );
            dummy.scale.set(this.satelliteSize, this.satelliteSize, this.satelliteSize);
            dummy.updateMatrix();
            this.satInstancedMesh.setMatrixAt(i, dummy.matrix);
        }
        this.satInstancedMesh.instanceMatrix.needsUpdate = true;
    }

    /**
     * Interpolate satellite positions between two flat Float32Arrays.
     * Alpha in [0, 1] controls the blend factor.
     */
    lerpSatellites(prevPositions, nextPositions, alpha) {
        const dummy = new THREE.Object3D();
        const numSats = this.satInstancedMesh.count;
        const oneMinusAlpha = 1.0 - alpha;
        for (let i = 0; i < numSats; i++) {
            const i3 = i * 3;
            dummy.position.set(
                prevPositions[i3] * oneMinusAlpha + nextPositions[i3] * alpha,
                prevPositions[i3 + 1] * oneMinusAlpha + nextPositions[i3 + 1] * alpha,
                prevPositions[i3 + 2] * oneMinusAlpha + nextPositions[i3 + 2] * alpha
            );
            dummy.scale.set(this.satelliteSize, this.satelliteSize, this.satelliteSize);
            dummy.updateMatrix();
            this.satInstancedMesh.setMatrixAt(i, dummy.matrix);
        }
        this.satInstancedMesh.instanceMatrix.needsUpdate = true;
    }

    updateLinks(state, prevPositions = null, nextPositions = null, alpha = 1.0) {
        const satPos = state.satellite_positions;
        const numIsls = state.num_isls;
        const numIlls = state.num_ills;
        const doLerp = prevPositions && nextPositions;
        const oneMinusAlpha = 1.0 - alpha;

        // Helper to write an interpolated endpoint into a flat position array
        const writeEndpoint = (outArr, outIdx, id) => {
            const i3 = id * 3;
            if (doLerp) {
                outArr[outIdx]     = prevPositions[i3]     * oneMinusAlpha + nextPositions[i3]     * alpha;
                outArr[outIdx + 1] = prevPositions[i3 + 1] * oneMinusAlpha + nextPositions[i3 + 1] * alpha;
                outArr[outIdx + 2] = prevPositions[i3 + 2] * oneMinusAlpha + nextPositions[i3 + 2] * alpha;
            } else {
                outArr[outIdx]     = satPos[i3];
                outArr[outIdx + 1] = satPos[i3 + 1];
                outArr[outIdx + 2] = satPos[i3 + 2];
            }
        };

        // Update ISL lines
        const islPositions = this.islLineSegments.geometry.attributes.position.array;
        let posIdx = 0;
        for (let i = 0; i < numIsls; i++) {
            const id1 = state.isls[i * 2];
            const id2 = state.isls[i * 2 + 1];
            if (id1 >= 0 && id2 >= 0) {
                writeEndpoint(islPositions, posIdx, id1);
                posIdx += 3;
                writeEndpoint(islPositions, posIdx, id2);
                posIdx += 3;
            }
        }
        while (posIdx < islPositions.length) {
            islPositions[posIdx++] = 0;
        }
        this.islLineSegments.geometry.attributes.position.needsUpdate = true;
        this.islLineSegments.visible = numIsls > 0;

        // Update ILL lines
        const illPositions = this.illLineSegments.geometry.attributes.position.array;
        posIdx = 0;
        for (let i = 0; i < numIlls; i++) {
            const id1 = state.ills[i * 2];
            const id2 = state.ills[i * 2 + 1];
            if (id1 >= 0 && id2 >= 0) {
                writeEndpoint(illPositions, posIdx, id1);
                posIdx += 3;
                writeEndpoint(illPositions, posIdx, id2);
                posIdx += 3;
            }
        }
        while (posIdx < illPositions.length) {
            illPositions[posIdx++] = 0;
        }
        this.illLineSegments.geometry.attributes.position.needsUpdate = true;
        this.illLineSegments.visible = numIlls > 0;
    }

    updatePlanets(state) {
        const planets = state.planets;
        for (let i = 0; i < planets.length; i++) {
            const p = planets[i];
            const mesh = this.planetMeshes[i];
            mesh.position.set(...p.position);
            const quat = new THREE.Quaternion(...p.rotation);
            mesh.quaternion.copy(PLANET_ALIGN_QUAT).premultiply(quat);
        }
    }

    /**
     * Interpolate planet positions and rotations between two states.
     * Alpha in [0, 1] controls the blend factor.
     */
    lerpPlanets(prevState, nextState, alpha) {
        const prevPlanets = prevState.planets;
        const nextPlanets = nextState.planets;
        const oneMinusAlpha = 1.0 - alpha;
        for (let i = 0; i < nextPlanets.length; i++) {
            const prevP = prevPlanets[i];
            const nextP = nextPlanets[i];
            const mesh = this.planetMeshes[i];
            mesh.position.set(
                prevP.position[0] * oneMinusAlpha + nextP.position[0] * alpha,
                prevP.position[1] * oneMinusAlpha + nextP.position[1] * alpha,
                prevP.position[2] * oneMinusAlpha + nextP.position[2] * alpha
            );
            const prevQuat = new THREE.Quaternion(...prevP.rotation);
            const nextQuat = new THREE.Quaternion(...nextP.rotation);
            const slerped = new THREE.Quaternion().copy(prevQuat).slerp(nextQuat, alpha);
            mesh.quaternion.copy(PLANET_ALIGN_QUAT).premultiply(slerped);
        }
    }

    updateUI(state, dt, frameCount) {
        const simTimeLabel = document.getElementById("sim-time");
        if (simTimeLabel) {
            simTimeLabel.textContent = `Time: ${state.sim_time.toFixed(1)}s`;
        }
        const satCountLabel = document.getElementById("sat-count");
        if (satCountLabel) {
            satCountLabel.textContent = `Sats: ${state.num_sats}`;
        }
        const fpsLabel = document.getElementById("fps");
        if (fpsLabel && dt > 0) {
            const fps = Math.round(1.0 / dt);
            fpsLabel.textContent = `FPS: ${fps}`;
        }
    }

    _updateFromState(state) {
        this.updateSatellites(state.satellite_positions);
        this.updateLinks(state);
        this.updatePlanets(state);
    }

    render(dt) {
        if (this.controls) this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    _onResize() {
        this.width = this.canvas.clientWidth;
        this.height = this.canvas.clientHeight;
        this.camera.aspect = this.width / this.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.width, this.height);
    }

    dispose() {
        window.removeEventListener("resize", this._resizeHandler);
        this.renderer.dispose();
        this.scene.clear();
    }
}
