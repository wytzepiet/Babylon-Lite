/**
 * Create a 2D texture from raw pixel bytes (CPU-generated data).
 *
 * This is the generic analog to Babylon.js `RawTexture`: it uploads an
 * application-provided byte buffer into a GPU texture rather than decoding an
 * image from a URL. Use it for procedurally generated images, decoded asset
 * formats, or palette / lookup tables.
 *
 * The default sampler is nearest-neighbor with clamp-to-edge addressing and no
 * mipmaps — the common case for pixel-art / data textures. Override via options.
 */

import { TU } from "../engine/gpu-flags.js";
import type { Texture2D } from "./texture-2d.js";
import type { EngineContext } from "../engine/engine.js";
import { acquireTexture } from "../resource/texture-acquire.js";
import { getOrCreateSampler, type TextureSamplerDescriptor } from "../resource/texture-sampler-pool.js";

/** Sampler and format overrides for `createTexture2DFromPixels()`. */
export interface PixelsTexture2DOptions {
    /** Address mode U. Default 'clamp-to-edge'. */
    addressModeU?: GPUAddressMode;
    /** Address mode V. Default 'clamp-to-edge'. */
    addressModeV?: GPUAddressMode;
    /** Min filter. Default 'nearest'. */
    minFilter?: GPUFilterMode;
    /** Mag filter. Default 'nearest'. */
    magFilter?: GPUFilterMode;
    /** Use sRGB format (rgba8unorm-srgb) so the hardware converts to linear on
     *  sample. Use for color data; leave false for lookup tables. Default false. */
    srgb?: boolean;
    /** Texel format of `data`. Default 'rgba8unorm' (or its sRGB form). Half-float formats take
     *  their texels as the half floats' bits in a `Uint16Array`; 32-bit float formats as a
     *  `Float32Array`. Filtering a 32-bit float format needs the device's `float32-filterable`. */
    format?: PixelsTextureFormat;
    /** Build a full mip chain, box-filtered, and rebuild it on every update. Default false. */
    mipmaps?: boolean;
    /** Mip filter, when `mipmaps` is set. Default 'linear'. */
    mipmapFilter?: GPUMipmapFilterMode;
}

/** Formats {@link createTexture2DFromPixels} takes. */
export type PixelsTextureFormat = "r8unorm" | "rg8unorm" | "rgba8unorm" | "r16float" | "rg16float" | "rgba16float" | "r32float" | "rg32float" | "rgba32float";

const TEXEL_BYTES: Record<string, number> = {
    r8unorm: 1,
    rg8unorm: 2,
    rgba8unorm: 4,
    "rgba8unorm-srgb": 4,
    r16float: 2,
    rg16float: 4,
    rgba16float: 8,
    r32float: 4,
    rg32float: 8,
    rgba32float: 16,
};

function texelBytes(format: GPUTextureFormat): number {
    const bytes = TEXEL_BYTES[format];
    if (!bytes) {
        throw new Error(`Pixel textures do not take format ${format}.`);
    }
    return bytes;
}

let _mipmaps: typeof import("./mipmap-preparation.js") | undefined;

/** Rebuild a pixel texture's mips from its level 0 (no-op without mips). */
function writeMipmaps(engine: EngineContext, texture: GPUTexture): void {
    if (texture.mipLevelCount <= 1) {
        return;
    }
    const encoder = engine._device.createCommandEncoder();
    _mipmaps!.recordPreparedMipmaps(encoder, _mipmaps!.prepareMipmaps(engine, texture));
    engine._device.queue.submit([encoder.finish()]);
}

/** Load the mip builder, once, before creating pixel textures with `mipmaps: true`. */
export async function enablePixelTextureMipmaps(): Promise<void> {
    _mipmaps ??= await import("./mipmap-preparation.js");
}

/**
 * Create a `Texture2D` from a tightly-packed RGBA8 byte buffer.
 *
 * @param engine - Engine context.
 * @param data - `width * height * 4` bytes, row-major, top-to-bottom, straight alpha.
 * @param width - Texture width in pixels (\>= 1).
 * @param height - Texture height in pixels (\>= 1).
 * @param options - Sampler / format overrides.
 */
export function createTexture2DFromPixels(
    engine: EngineContext,
    data: Uint8Array | Uint16Array | Float32Array,
    width: number,
    height: number,
    options: PixelsTexture2DOptions = {}
): Texture2D {
    if (width < 1 || height < 1) {
        throw new Error(`createTexture2DFromPixels: width/height must be >= 1 (got ${width}x${height})`);
    }
    const device = engine._device;
    const format: GPUTextureFormat = options.format ?? (options.srgb ? "rgba8unorm-srgb" : "rgba8unorm");
    const bytesPerTexel = texelBytes(format);
    const expected = width * height * bytesPerTexel;
    if (data.byteLength < expected) {
        throw new Error(`createTexture2DFromPixels: data too short — need ${expected} bytes for ${width}x${height} ${format}, got ${data.byteLength}`);
    }
    if (options.mipmaps && !_mipmaps) {
        throw new Error("createTexture2DFromPixels: call enablePixelTextureMipmaps() before asking for mipmaps.");
    }

    const texture = device.createTexture({
        size: { width, height },
        format,
        mipLevelCount: options.mipmaps ? Math.floor(Math.log2(Math.max(width, height))) + 1 : 1,
        usage: TU.TEXTURE_BINDING | TU.COPY_DST | (options.mipmaps ? TU.RENDER_ATTACHMENT : 0),
    });

    device.queue.writeTexture({ texture }, data as Uint8Array<ArrayBuffer>, { bytesPerRow: width * bytesPerTexel, rowsPerImage: height }, { width, height });
    writeMipmaps(engine, texture);

    const samplerDesc: TextureSamplerDescriptor = {
        addressModeU: options.addressModeU ?? "clamp-to-edge",
        addressModeV: options.addressModeV ?? "clamp-to-edge",
        minFilter: options.minFilter ?? "nearest",
        magFilter: options.magFilter ?? "nearest",
        ...(options.mipmaps ? { mipmapFilter: options.mipmapFilter ?? "linear" } : {}),
    };
    const sampler = getOrCreateSampler(engine, samplerDesc);

    const tex: Texture2D = { texture, view: texture.createView(), sampler, width, height };
    engine._dlr?.p(tex, data, options, bytesPerTexel);
    acquireTexture(tex);
    return tex;
}

/** Sampler / format overrides for `createRenderTexture2D()`. */
export interface RenderTexture2DOptions {
    /** Address mode U. Default 'clamp-to-edge'. */
    addressModeU?: GPUAddressMode;
    /** Address mode V. Default 'clamp-to-edge'. */
    addressModeV?: GPUAddressMode;
    /** Min filter. Default 'linear'. */
    minFilter?: GPUFilterMode;
    /** Mag filter. Default 'linear'. */
    magFilter?: GPUFilterMode;
    /**
     * Color format. Default `engine.format` so it can be sampled and presented.
     *
     * ⚠️ Only the default `engine.format` is compatible with a `SpriteRenderer`
     * target (`setSpriteRendererTarget`): sprite pipelines are created with
     * `engine.format`, and a render pass whose color attachment format differs from
     * the bound pipeline fails WebGPU validation at pass begin. Override this **only**
     * for offscreen targets you render into by some OTHER means (a custom pass /
     * `EffectRenderer`), never as a sprite-render target.
     */
    format?: GPUTextureFormat;
}

/**
 * Create an empty `Texture2D` usable as **both a render target and a sampled texture**
 * (`RENDER_ATTACHMENT | TEXTURE_BINDING`). This is the building block for offscreen
 * render-to-texture: render a pass into `tex.view`, then sample `tex` in a later pass
 * (e.g. a fullscreen post-process). Defaults to the engine's swapchain format + a
 * linear sampler so the result can be presented directly.
 *
 * To use the result as a `SpriteRenderer` target (via `setSpriteRendererTarget`), leave
 * `format` at its default `engine.format` — sprite pipelines bake in that format, so a
 * differently-formatted target trips WebGPU validation at render-pass begin. A custom
 * `format` is for offscreen targets driven by some other pass, not the sprite renderer.
 */
export function createRenderTexture2D(engine: EngineContext, width: number, height: number, options: RenderTexture2DOptions = {}): Texture2D {
    if (width < 1 || height < 1) {
        throw new Error(`createRenderTexture2D: width/height must be >= 1 (got ${width}x${height})`);
    }
    const device = engine._device;
    const format = options.format ?? engine.format;
    const texture = device.createTexture({
        size: { width, height },
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    });
    const samplerDesc: TextureSamplerDescriptor = {
        addressModeU: options.addressModeU ?? "clamp-to-edge",
        addressModeV: options.addressModeV ?? "clamp-to-edge",
        minFilter: options.minFilter ?? "linear",
        magFilter: options.magFilter ?? "linear",
    };
    const sampler = getOrCreateSampler(engine, samplerDesc);
    const tex: Texture2D = { texture, view: texture.createView(), sampler, width, height, _uvTransformDisabled: true };
    engine._dlr?.r(tex, width, height, format, samplerDesc);
    acquireTexture(tex);
    return tex;
}

/**
 * Update a rectangular region of an existing `Texture2D` from a tightly-packed RGBA8 byte buffer.
 *
 * The texture must have been created with `COPY_DST` usage (as `createTexture2DFromPixels` does).
 * This is the runtime counterpart to `createTexture2DFromPixels` — for data textures the app mutates
 * each frame / on demand (e.g. a terrain carve heightmap stamped by a dig tool).
 *
 * @param engine - Engine context.
 * @param tex - Target texture (from `createTexture2DFromPixels`).
 * @param data - `width * height * 4` bytes for the sub-region, row-major, straight alpha.
 * @param x - Destination origin X in texels (default 0).
 * @param y - Destination origin Y in texels (default 0).
 * @param width - Region width in texels (default `tex.width`).
 * @param height - Region height in texels (default `tex.height`).
 */
export function updateTexture2DFromPixels(
    engine: EngineContext,
    tex: Texture2D,
    data: Uint8Array | Uint16Array | Float32Array,
    x = 0,
    y = 0,
    width = tex.width,
    height = tex.height
): void {
    if (width < 1 || height < 1) {
        throw new Error(`updateTexture2DFromPixels: width/height must be >= 1 (got ${width}x${height})`);
    }
    const bytesPerTexel = texelBytes(tex.texture.format);
    const expected = width * height * bytesPerTexel;
    if (data.byteLength < expected) {
        throw new Error(`updateTexture2DFromPixels: data too short — need ${expected} bytes for ${width}x${height} ${tex.texture.format}, got ${data.byteLength}`);
    }
    engine._device.queue.writeTexture(
        { texture: tex.texture, origin: { x, y } },
        data as Uint8Array<ArrayBuffer>,
        { bytesPerRow: width * bytesPerTexel, rowsPerImage: height },
        { width, height }
    );
    writeMipmaps(engine, tex.texture);
    engine._dlr?.w(tex, data, x, y, width, height, 0, width * bytesPerTexel);
}

/** Sampler / format overrides for `createTexture3DFromPixels()`. */
export interface PixelsTexture3DOptions {
    /** Address mode U/V/W. Default 'clamp-to-edge' (the right choice for a colour LUT — the cube edges
     *  must not wrap). */
    addressMode?: GPUAddressMode;
    /** Min/mag filter. Default 'linear' so a colour-grading LUT interpolates trilinearly in one sample. */
    filter?: GPUFilterMode;
    /** Use sRGB format (rgba8unorm-srgb). Leave false for a linear/display LUT. Default false. */
    srgb?: boolean;
}

/** A `Texture2D` handle whose underlying GPU texture is `dimension:"3d"`, plus its depth. Bind it to a
 *  fullscreen effect with `viewDimension:"3d"` and sample it in WGSL as `texture_3d<f32>`. */
export type Texture3D = Texture2D & { depth: number };

/**
 * Create a **3D** texture from a tightly-packed RGBA8 byte buffer — the volumetric analog of
 * `createTexture2DFromPixels`. The primary use is a colour-grading LUT (a `.cube`/HALD colour cube):
 * upload the N×N×N RGBA volume once and sample it trilinearly with a single `textureSample`.
 *
 * The default sampler is linear with clamp-to-edge on all three axes — exactly what a LUT wants (linear
 * = trilinear interpolation between grid points; clamp = no wrap at the cube faces). The returned handle
 * is a `Texture2D` (so it drops straight into `setEffectTexture`) with an extra `depth` field; its `view`
 * is created with `dimension:"3d"`.
 *
 * @param engine - Engine context.
 * @param data - `width * height * depth * 4` bytes, RGBA8, ordered x fastest, then y, then z (slice-major).
 * @param width - Cube size along R (\>= 1).
 * @param height - Cube size along G (\>= 1).
 * @param depth - Cube size along B (\>= 1).
 * @param options - Sampler / format overrides.
 */
export function createTexture3DFromPixels(engine: EngineContext, data: Uint8Array, width: number, height: number, depth: number, options: PixelsTexture3DOptions = {}): Texture3D {
    if (width < 1 || height < 1 || depth < 1) {
        throw new Error(`createTexture3DFromPixels: width/height/depth must be >= 1 (got ${width}x${height}x${depth})`);
    }
    const expected = width * height * depth * 4;
    if (data.length < expected) {
        throw new Error(`createTexture3DFromPixels: data too short — need ${expected} bytes for ${width}x${height}x${depth} RGBA, got ${data.length}`);
    }

    const device = engine._device;
    const format: GPUTextureFormat = options.srgb ? "rgba8unorm-srgb" : "rgba8unorm";

    const texture = device.createTexture({
        size: { width, height, depthOrArrayLayers: depth },
        dimension: "3d",
        format,
        usage: TU.TEXTURE_BINDING | TU.COPY_DST,
    });

    device.queue.writeTexture({ texture }, data as Uint8Array<ArrayBuffer>, { bytesPerRow: width * 4, rowsPerImage: height }, { width, height, depthOrArrayLayers: depth });

    const address = options.addressMode ?? "clamp-to-edge";
    const filter = options.filter ?? "linear";
    const sampler = getOrCreateSampler(engine, {
        addressModeU: address,
        addressModeV: address,
        addressModeW: address,
        minFilter: filter,
        magFilter: filter,
    });

    const tex: Texture3D = { texture, view: texture.createView({ dimension: "3d" }), sampler, width, height, depth };
    acquireTexture(tex);
    return tex;
}
