/**
 * Mesh data for the vertex and per-instance attributes that material plugins declare
 * (`MaterialPlugin.getAttributes`). Opt-in: only scenes that import it pay for it.
 */

import { BU } from "../engine/gpu-flags.js";
import type { EngineContext } from "../engine/engine.js";
import { bumpVisibilityEpoch } from "../engine/engine.js";
import type { Mesh } from "./mesh.js";

/**
 * Set the data of a plugin attribute on `mesh`: one value per vertex, or per thin instance for an
 * attribute declared `perInstance`, packed as float32 with the attribute's own width.
 *
 * Writing data that fits the attribute's current buffer keeps that buffer, so recorded draws stay
 * valid; larger data replaces it and re-records the draws that bind it.
 */
export function setMeshAttribute(engine: EngineContext, mesh: Mesh, name: string, data: Float32Array): void {
    const attributes = (mesh._attributes ??= {});
    let buffer = attributes[name];
    if (!buffer || buffer.size < data.byteLength) {
        buffer?.destroy();
        buffer = attributes[name] = engine._device.createBuffer({
            size: Math.max(16, (data.byteLength + 3) & ~3),
            usage: BU.VERTEX | BU.COPY_DST,
        });
        bumpVisibilityEpoch();
    }
    engine._device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
}
