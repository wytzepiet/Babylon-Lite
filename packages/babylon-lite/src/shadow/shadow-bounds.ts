import type { ShadowGenerator } from "./shadow-generator.js";

/**
 * Set the world-space box a directional PCF shadow map covers, in place of the box round all its
 * casters (Babylon.js `autoUpdateExtends = false` with ortho extents). Use it when the casters
 * spread far wider than what is seen, as a world loaded in chunks round the camera does. The box
 * also sets the map's depth range, so the light's position no longer matters and a bias means the
 * same whatever the box. The map
 * is redrawn only when the box changes. `null` returns to fitting the casters.
 */
export function setShadowGeneratorBounds(sg: ShadowGenerator, min: readonly [number, number, number] | null, max?: readonly [number, number, number]): void {
    const config = sg._config;
    const bounds = config._bounds;
    if (!min || !max) {
        if (bounds) {
            config._bounds = null;
            config._boundsVersion = (config._boundsVersion ?? 0) + 1;
        }
        return;
    }
    const next = new Float32Array([min[0], min[1], min[2], max[0], max[1], max[2]]);
    // Compared as stored: a box given in doubles that a float cannot hold would never match itself.
    if (bounds && next.every((v, i) => v === bounds[i])) {
        return;
    }
    config._bounds = next;
    config._boundsVersion = (config._boundsVersion ?? 0) + 1;
}
