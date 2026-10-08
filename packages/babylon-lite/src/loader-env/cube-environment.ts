/**
 * A PBR environment drawn by the application: six float faces it writes itself (a sky computed from
 * a time of day, a palette, a probe read back) and may redraw whenever it likes. The faces become the
 * specular cube (box-filtered mips) and their spherical harmonics the diffuse irradiance, as Babylon.js
 * does for a `RawCubeTexture` set as `scene.environmentTexture`.
 */

import { F64, U16 } from "../engine/typed-arrays.js";
import { TU } from "../engine/gpu-flags.js";
import type { SceneContext } from "../scene/scene.js";
import type { EnvironmentTextures } from "./load-env.js";
import { polynomialToPreScaledHarmonics } from "./load-env.js";
import { assembleEnvironmentTextures } from "./env-helpers.js";
import { generateBrdfLut } from "../loader-hdr/hdr-ibl-pipeline.js";
import { shToPolynomial } from "../math/spherical-harmonics.js";
import { mipLevelCount } from "../texture/mip-count.js";
import { prepareMipmaps, recordPreparedMipmaps, type PreparedMipmapLevel } from "../texture/mipmap-preparation.js";
import { _invalidateSceneUboCaches, registerEnvSceneUniforms } from "../scene/scene-ubo-extras.js";

/** Opaque handle for an environment created by {@link createCubeEnvironment}. */
export interface CubeEnvironment {
    /** @internal */
    readonly _scene: SceneContext;
    /** @internal */
    readonly _size: number;
    /** @internal */
    readonly _texture: GPUTexture;
    /** @internal */
    readonly _textures: EnvironmentTextures;
    /** @internal */
    readonly _mipmaps: readonly (readonly PreparedMipmapLevel[])[];
    /** @internal Half-float staging for one face. */
    readonly _half: Uint16Array;
}

/** Options for {@link createCubeEnvironment}. */
export interface CubeEnvironmentOptions {
    /** Texels along a face's side. */
    readonly size: number;
    /** How fast reflections blur with roughness (Babylon.js `lodGenerationScale`). Default 0.8. */
    readonly lodGenerationScale?: number;
}

/**
 * Create the environment of an unregistered scene from six faces of linear RGBA float texels,
 * `size × size` each, in Babylon.js cube order and orientation (+X, −X, +Y, −Y, +Z, −Z).
 */
export function createCubeEnvironment(scene: SceneContext, faces: readonly Float32Array[], options: CubeEnvironmentOptions): CubeEnvironment {
    if (scene._built) {
        throw new Error("createCubeEnvironment must run before the scene is registered.");
    }
    if (scene._envTextures) {
        throw new Error("createCubeEnvironment requires a scene without an existing environment.");
    }
    const engine = scene.surface.engine;
    const size = options.size;
    const texture = engine._device.createTexture({
        size: [size, size, 6],
        mipLevelCount: mipLevelCount(size, size),
        format: "rgba16float",
        usage: TU.TEXTURE_BINDING | TU.COPY_DST | TU.RENDER_ATTACHMENT,
    });
    const brdfLut = generateBrdfLut(engine);
    const mipmaps: PreparedMipmapLevel[][] = [];
    for (let face = 0; face < 6; face++) {
        mipmaps.push(prepareMipmaps(engine, texture, face));
    }
    const polynomial = cubePolynomial(faces, size);
    const environment: CubeEnvironment = {
        _scene: scene,
        _size: size,
        _texture: texture,
        _textures: assembleEnvironmentTextures(texture, brdfLut, polynomial, options.lodGenerationScale ?? 0.8, engine),
        _mipmaps: mipmaps,
        _half: new U16(size * size * 4),
    };
    upload(environment, faces);
    scene._envTextures = environment._textures;
    registerEnvSceneUniforms(scene);
    scene._disposables.push(() => {
        if (scene._envTextures === environment._textures) {
            scene._envTextures = undefined;
        }
        texture.destroy();
        brdfLut.destroy();
    });
    return environment;
}

/** Redraw the environment from new faces, laid out as for {@link createCubeEnvironment}. */
export function updateCubeEnvironment(environment: CubeEnvironment, faces: readonly Float32Array[]): void {
    upload(environment, faces);
    const textures = environment._textures;
    const polynomial = cubePolynomial(faces, environment._size);
    textures.irradianceSH.set(polynomial);
    textures.sphericalHarmonics.set(polynomialToPreScaledHarmonics(polynomial));
    _invalidateSceneUboCaches(environment._scene);
}

function upload(environment: CubeEnvironment, faces: readonly Float32Array[]): void {
    const device = environment._scene.surface.engine._device;
    const size = environment._size;
    const half = environment._half;
    for (let face = 0; face < 6; face++) {
        const texels = faces[face]!;
        for (let i = 0; i < half.length; i++) {
            half[i] = toHalf(texels[i]!);
        }
        device.queue.writeTexture({ texture: environment._texture, origin: [0, 0, face] }, half, { bytesPerRow: size * 8 }, [size, size, 1]);
    }
    const encoder = device.createCommandEncoder();
    for (const faceMipmaps of environment._mipmaps) {
        recordPreparedMipmaps(encoder, faceMipmaps);
    }
    device.queue.submit([encoder.finish()]);
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** Round a float to the nearest half float's bits (finite, non-negative light; no NaN or infinity). */
function toHalf(value: number): number {
    f32[0] = value;
    const bits = u32[0]!;
    const sign = (bits >>> 16) & 0x8000;
    const exponent = ((bits >>> 23) & 0xff) - 112;
    if (exponent <= 0) {
        return sign;
    }
    if (exponent >= 31) {
        return sign | 0x7bff;
    }
    return sign | (exponent << 10) | (((bits & 0x7fffff) + 0x1000) >>> 13);
}

// Face orientations of Babylon.js `_FileFaces`: the outward normal, then the axes a face's texel rows and columns run along.
const FACES = [
    [1, 0, 0, 0, 0, -1, 0, -1, 0],
    [-1, 0, 0, 0, 0, 1, 0, -1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 1],
    [0, -1, 0, 1, 0, 0, 0, 0, -1],
    [0, 0, 1, 1, 0, 0, 0, -1, 0],
    [0, 0, -1, -1, 0, 0, 0, -1, 0],
] as const;

const PI = Math.PI;
const SH_BASIS = [
    Math.sqrt(1 / (4 * PI)),
    Math.sqrt(3 / (4 * PI)),
    Math.sqrt(3 / (4 * PI)),
    Math.sqrt(3 / (4 * PI)),
    Math.sqrt(15 / (4 * PI)),
    Math.sqrt(15 / (4 * PI)),
    Math.sqrt(5 / (16 * PI)),
    Math.sqrt(15 / (4 * PI)),
    Math.sqrt(15 / (16 * PI)),
];
const SH_COS_KERNEL = [PI, (2 * PI) / 3, (2 * PI) / 3, (2 * PI) / 3, PI / 4, PI / 4, PI / 4, PI / 4, PI / 4];

function areaElement(x: number, y: number): number {
    return Math.atan2(x * y, Math.sqrt(x * x + y * y + 1));
}

/** The faces' irradiance as Babylon.js's spherical polynomial (`CubeMapToSphericalPolynomialTools`). */
function cubePolynomial(faces: readonly Float32Array[], size: number): Float32Array {
    const du = 2 / size;
    const halfTexel = du / 2;
    const sh = new F64(27);
    let total = 0;
    for (let face = 0; face < 6; face++) {
        const texels = faces[face]!;
        const [nx, ny, nz, xx, xy, xz, yx, yy, yz] = FACES[face]!;
        for (let row = 0; row < size; row++) {
            const v = halfTexel - 1 + row * du;
            for (let col = 0; col < size; col++) {
                const u = halfTexel - 1 + col * du;
                const dx = xx * u + yx * v + nx;
                const dy = xy * u + yy * v + ny;
                const dz = xz * u + yz * v + nz;
                const inv = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz);
                const x = dx * inv;
                const y = dy * inv;
                const z = dz * inv;
                const solidAngle =
                    areaElement(u - halfTexel, v - halfTexel) -
                    areaElement(u - halfTexel, v + halfTexel) -
                    areaElement(u + halfTexel, v - halfTexel) +
                    areaElement(u + halfTexel, v + halfTexel);
                const terms = [1, y, z, x, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y];
                const i = (row * size + col) * 4;
                const r = texels[i]!;
                const g = texels[i + 1]!;
                const b = texels[i + 2]!;
                for (let k = 0; k < 9; k++) {
                    const w = solidAngle * SH_BASIS[k]! * terms[k]!;
                    sh[k] = sh[k]! + r * w;
                    sh[9 + k] = sh[9 + k]! + g * w;
                    sh[18 + k] = sh[18 + k]! + b * w;
                }
                total += solidAngle;
            }
        }
    }
    // Normalise to the sphere, convolve with the cosine lobe, and take Lambertian radiance.
    for (let k = 0; k < 27; k++) {
        sh[k] = (sh[k]! * ((4 * PI) / total) * SH_COS_KERNEL[k % 9]!) / PI;
    }
    return shToPolynomial(sh);
}
