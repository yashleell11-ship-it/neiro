/**
 * A real avatar: three + @pixiv/three-vrm, loaded only when there is a
 * VRM to load.
 *
 * Everything is imported dynamically, and the page never imports this
 * module's dependencies itself, for two reasons. The drawn face must
 * keep working with no vendor/ directory at all — it is what proves the
 * blending and the visemes before an avatar exists. And the libraries
 * come from web/vendor/ through the import map in index.html, never a
 * CDN: the demo has to run with Wi-Fi off, and the network this project
 * is built on blocks GitHub release assets (CLAUDE.md rule 3), so they
 * are copied out of node_modules by `bun add` — see web/README.md.
 *
 * The object returned has the same shape as FallbackFace — `available`,
 * `setExpression()`, `setMouth()`, `draw(t)` — so index.html does not
 * care which one it is holding.
 */

import { EXPRESSIONS, VISEMES } from './face.js';

// The camera frames the head: this far in front of it, at this field of
// view, a face fills the stage without the shoulders being cut.
const CAMERA_DISTANCE_M = 0.75;
const CAMERA_FOV_DEG = 28;
// Auto-blink, same rhythm as the drawn face.
const BLINK_PERIOD_S = 4.2;
const BLINK_PHASE = 0.97;

/**
 * Load `url` and mount the renderer in `container`. Rejects if the
 * libraries or the file are missing; the caller keeps the drawn face.
 */
export async function loadVrm(container, url) {
  const THREE = await import('three');
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  const { VRMLoaderPlugin, VRMUtils } = await import('@pixiv/three-vrm');

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(CAMERA_FOV_DEG, 1, 0.05, 20);
  const key = new THREE.DirectionalLight(0xffffff, Math.PI);
  key.position.set(1, 1.5, 1.5);
  scene.add(key, new THREE.AmbientLight(0xffffff, 0.8));

  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error(`${url} is not a VRM`);
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  (VRMUtils.combineSkeletons ?? VRMUtils.removeUnnecessaryJoints)?.(gltf.scene);
  // VRM 0.x models face the other way; this makes both face the camera.
  VRMUtils.rotateVRM0(vrm);
  scene.add(vrm.scene);
  if (vrm.lookAt) vrm.lookAt.target = camera;

  // Aim at the head, not the model origin (its feet).
  const head = vrm.humanoid?.getNormalizedBoneNode('head');
  const focus = new THREE.Vector3(0, 1.4, 0);
  if (head) head.getWorldPosition(focus);
  camera.position.set(focus.x, focus.y, focus.z + CAMERA_DISTANCE_M);
  camera.lookAt(focus);

  const manager = vrm.expressionManager;
  const has = (name) => Boolean(manager?.getExpression?.(name) ?? manager?.expressionMap?.[name]);
  const available = new Set([...EXPRESSIONS, 'neutral'].filter(has));
  const visemes = VISEMES.filter(has);
  const blinks = has('blink');

  function resize() {
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener('resize', resize);
  container.append(renderer.domElement);

  let last = null;
  return {
    kind: 'vrm',
    available,
    element: renderer.domElement,
    setExpression(weights) {
      if (!manager) return;
      for (const k of EXPRESSIONS) if (available.has(k)) manager.setValue(k, weights[k] ?? 0);
    },
    setMouth(shape, amount) {
      if (!manager) return;
      for (const v of visemes) manager.setValue(v, v === shape ? amount : 0);
    },
    draw(t) {
      const dt = last === null ? 0 : t - last;
      last = t;
      if (manager && blinks) {
        const phase = (t % BLINK_PERIOD_S) / BLINK_PERIOD_S;
        manager.setValue('blink', phase > BLINK_PHASE ? 1 : 0);
      }
      vrm.update(dt);
      renderer.render(scene, camera);
    },
    dispose() {
      window.removeEventListener('resize', resize);
      renderer.domElement.remove();
      renderer.dispose();
    },
  };
}
