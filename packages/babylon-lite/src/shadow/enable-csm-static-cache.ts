import type { EngineContext } from "../engine/engine.js";
import type { ShadowGenerator } from "./shadow-generator.js";
import type { Mesh } from "../mesh/mesh.js";

/** Options for {@link enableCsmStaticCache}. */
export interface CsmStaticCacheOptions {
    /** Accumulated sun-rotation angle (radians) since the last cascade refit that forces the next one. */
    refitAngle: number;
    /** Force a refit after this much wall time, including while the light is paused. Default 0 (no interval floor). */
    refitMaxIntervalMs?: number;
    /** Spread a drift-only refit's static re-render over frames: at most this many cascades are
     *  re-rendered per frame, round-robin, so the periodic refresh costs a slice of every frame instead
     *  of one long frame. Refits caused by the camera, the content or the caster membership still
     *  re-render every cascade in their own frame. 0 or omitted keeps the single-frame re-render. */
    staticCascadesPerFrame?: number;
    /** Which casters stand still, if the application knows: those it says yes to are drawn into the
     *  cache from their first refit and stay there, re-rendered where they now are when they change; the
     *  rest are drawn over it every frame, however long they stand. Omitted, the cache learns it: every
     *  caster starts over the cache and is moved into it once it has stood still a while. */
    still?: (mesh: Mesh) => boolean;
}

let enabling: WeakMap<ShadowGenerator, Promise<void>> | null = null;

/**
 * Enables the opt-in static/dynamic caster cache for a CSM generator.
 *
 * Call and await this before registering the scene or requesting the generator's
 * receiver texture. The cache implementation is dynamically imported, so CSM scenes
 * that do not call this function fetch none of its code.
 */
export function enableCsmStaticCache(engine: EngineContext, sg: ShadowGenerator, options: CsmStaticCacheOptions): Promise<void> {
    if (sg._shadowType !== "csm") {
        return Promise.reject(new Error("enableCsmStaticCache requires a CSM shadow generator"));
    }
    if (!Number.isFinite(options.refitAngle) || options.refitAngle <= 0) {
        return Promise.reject(new RangeError("enableCsmStaticCache requires a positive finite refitAngle"));
    }
    const staticCascadesPerFrame = options.staticCascadesPerFrame ?? 0;
    if (!Number.isInteger(staticCascadesPerFrame) || staticCascadesPerFrame < 0) {
        return Promise.reject(new RangeError("enableCsmStaticCache requires a non-negative integer staticCascadesPerFrame"));
    }
    if (sg._shadowTaskState || sg._csmReceiverTexture || engine.surfaces.some((surface) => surface._renderingContexts.length > 0)) {
        return Promise.reject(new Error("enableCsmStaticCache must run before scene registration and receiver texture access"));
    }
    if (sg._csmCache?._loaded) {
        return Promise.resolve();
    }
    const active = enabling?.get(sg);
    if (active) {
        return active;
    }
    const oldTexture = sg._depthTexture;
    const mapSize = sg._config._mapSize;
    sg._depthTexture = engine._device.createTexture({
        size: { width: mapSize, height: mapSize, depthOrArrayLayers: sg._csmCascadeCount! },
        format: "depth32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    sg._csmCache = { _refitAngle: options.refitAngle, _refitMaxIntervalMs: options.refitMaxIntervalMs ?? 0, _staticCascadesPerFrame: staticCascadesPerFrame, _still: options.still };
    oldTexture.destroy();
    const promise = import("./csm-shadow-cache.js").then((module) => {
        sg._csmCache!._loaded = true;
        const config = sg._config as import("./csm-shadow-task-hooks.js").CsmConfig;
        const ensure: NonNullable<ShadowGenerator["_ensureShadowTaskState"]> = (eng, scene, casterMeshes) => {
            const state = module.ensureCsmShadowCacheState(eng, scene, sg, config, casterMeshes, sg._shadowTaskState ?? null);
            sg._shadowTaskState = state;
            return state;
        };
        const render: NonNullable<ShadowGenerator["_renderShadowMap"]> = (eng, state) =>
            module.renderCsmShadowMapCached(eng, sg, state as import("./csm-shadow-task-hooks.js").CsmTaskState, config);
        if (sg._replaceShadowTaskHooks) {
            sg._replaceShadowTaskHooks(ensure, render);
        } else {
            sg._ensureShadowTaskState = ensure;
            sg._renderShadowMap = render;
        }
    });
    (enabling ??= new WeakMap()).set(sg, promise);
    void promise.then(
        () => enabling?.delete(sg),
        () => enabling?.delete(sg)
    );
    return promise;
}

/**
 * Hold a static-cache CSM map to its static casters alone, or let it have its dynamic ones again.
 *
 * While on, each frame copies the static cache into the map and draws no dynamic caster over it, so a
 * pass rendered then is shadowed by what stands still and by nothing that moves; the first frame after
 * draws the dynamic casters over the cache again, whether or not they moved. A generator without the
 * static cache ignores it.
 *
 * Turning it off says whether a frame drew the map so while it was on: none did while the generator
 * was off or its caster set was still loading, and what was shaded then was shaded by whatever the map
 * held before.
 */
export function setShadowStaticOnly(sg: ShadowGenerator, on: boolean): boolean {
    const drawn = sg._staticOnly === "drawn";
    sg._staticOnly = on ? "asked" : undefined;
    return drawn;
}
