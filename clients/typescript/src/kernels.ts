import type { KernelSpec } from "./types.js";

/**
 * Helpers that build wire-format kernel specs (`{type, params}`), mirroring the
 * Python kernel classes. You can also pass a `KernelSpec` literal directly.
 */

/** Exponential decay; `halfLife` in seconds. */
export function exponential(
  halfLife: number,
  opts: { horizonHalfLives?: number } = {},
): KernelSpec {
  const params: { half_life: number; horizon_half_lives?: number } = { half_life: halfLife };
  if (opts.horizonHalfLives !== undefined) params.horizon_half_lives = opts.horizonHalfLives;
  return { type: "exponential", params };
}

/** Flat window between `minLag` and `maxLag` seconds. */
export function window(minLag = 0, maxLag = 3600): KernelSpec {
  return { type: "window", params: { min_lag: minLag, max_lag: maxLag } };
}

/** Log-normal lag; `median` in seconds. */
export function lognormal(
  median: number,
  opts: { sigma?: number; tailSigmas?: number } = {},
): KernelSpec {
  const params: { median: number; sigma?: number; tail_sigmas?: number } = { median };
  if (opts.sigma !== undefined) params.sigma = opts.sigma;
  if (opts.tailSigmas !== undefined) params.tail_sigmas = opts.tailSigmas;
  return { type: "lognormal", params };
}

/** Gamma lag with `shape` and `scale` (seconds). */
export function gamma(shape: number, scale: number, opts: { tailStds?: number } = {}): KernelSpec {
  const params: { shape: number; scale: number; tail_stds?: number } = { shape, scale };
  if (opts.tailStds !== undefined) params.tail_stds = opts.tailStds;
  return { type: "gamma", params };
}
