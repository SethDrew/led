export interface Branch {
  id: string;
  parent: string | null;
  path: [number, number][];
  ledCount: number;
}

export interface Topology {
  name: string;
  numLeds: number;
  branches: Branch[];
}

export interface ComputedTopology {
  positions: [number, number][];   // one per LED
  branchId: string[];              // which branch each LED belongs to
  branchProgress: number[];        // 0-1 progress along its branch
  globalProgress: number[];        // 0-1 strip index: root tips=0, trunk≈0.5, canopy tips=1
  stripId: number[];               // which strip (leaf→trunk chain) this LED belongs to
}

// Compute arc length along a polyline
function polylineLength(path: [number, number][]): number {
  let len = 0;
  for (let i = 1; i < path.length; i++) {
    const dx = path[i][0] - path[i - 1][0];
    const dy = path[i][1] - path[i - 1][1];
    len += Math.sqrt(dx * dx + dy * dy);
  }
  return len;
}

// Interpolate a position at a given distance along a polyline
function interpolateAlongPolyline(
  path: [number, number][],
  distance: number
): [number, number] {
  let remaining = distance;
  for (let i = 1; i < path.length; i++) {
    const dx = path[i][0] - path[i - 1][0];
    const dy = path[i][1] - path[i - 1][1];
    const segLen = Math.sqrt(dx * dx + dy * dy);
    if (remaining <= segLen || i === path.length - 1) {
      const t = segLen > 0 ? remaining / segLen : 0;
      return [
        path[i - 1][0] + dx * t,
        path[i - 1][1] + dy * t,
      ];
    }
    remaining -= segLen;
  }
  return path[path.length - 1];
}

function dist2d(a: [number, number], b: [number, number]): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.sqrt(dx * dx + dy * dy);
}

export function computeLedPositions(topology: Topology): ComputedTopology {
  const positions: [number, number][] = [];
  const branchId: string[] = [];
  const branchProgress: number[] = [];

  // Build hierarchy
  const branchById = new Map<string, Branch>();
  const childrenOf = new Map<string, string[]>();
  for (const b of topology.branches) {
    branchById.set(b.id, b);
    if (b.parent) {
      const c = childrenOf.get(b.parent) || [];
      c.push(b.id);
      childrenOf.set(b.parent, c);
    }
  }

  // Place LEDs on branches, track indices per branch
  const branchLedIdx = new Map<string, number[]>();

  for (const branch of topology.branches) {
    const totalLen = polylineLength(branch.path);
    const indices: number[] = [];

    for (let i = 0; i < branch.ledCount; i++) {
      const t = branch.ledCount > 1 ? i / (branch.ledCount - 1) : 0.5;
      const distance = t * totalLen;
      const pos = interpolateAlongPolyline(branch.path, distance);

      const idx = positions.length;
      positions.push(pos);
      branchId.push(branch.id);
      branchProgress.push(t);
      indices.push(idx);
    }
    branchLedIdx.set(branch.id, indices);
  }

  // --- Merge nearby junction points ---
  // Snap first LED of each child to last LED of its parent
  const MERGE_THRESHOLD = 0.02;
  for (const branch of topology.branches) {
    if (!branch.parent) continue;
    const parentIdx = branchLedIdx.get(branch.parent);
    const myIdx = branchLedIdx.get(branch.id);
    if (!parentIdx?.length || !myIdx?.length) continue;

    const parentLast = parentIdx[parentIdx.length - 1];
    const myFirst = myIdx[0];
    if (dist2d(positions[parentLast], positions[myFirst]) < MERGE_THRESHOLD) {
      positions[myFirst] = [...positions[parentLast]];
    }
  }

  // --- Build strips and compute global progress ---
  // Each leaf traces back to trunk = one strip.
  // Root strips: root_tip → trunk (progress: 0 at root tip, increases toward trunk)
  // Canopy strips: trunk → canopy_tip (progress: continues from trunk to 1 at tip)
  // Combined: root_tip=0, trunk≈middle, canopy_tip=1

  const leaves = topology.branches.filter(
    b => !(childrenOf.get(b.id)?.length)
  );

  // For each leaf, get the chain of branch IDs from trunk to leaf
  function getChain(leafId: string): string[] {
    const chain: string[] = [];
    let cur: string | null = leafId;
    while (cur) {
      chain.unshift(cur);
      cur = branchById.get(cur)?.parent ?? null;
    }
    return chain; // trunk first, leaf last
  }

  // Compute "signed LED distance from trunk" for each LED.
  // Canopy LEDs: positive (trunk→tip direction)
  // Root LEDs: negative (trunk→root_tip direction)
  const signedDist = new Float32Array(positions.length); // default 0
  const stripId = new Int32Array(positions.length).fill(-1);

  for (let s = 0; s < leaves.length; s++) {
    const leaf = leaves[s];
    const isRoot = leaf.id.startsWith('root_');
    const chain = getChain(leaf.id);

    // Collect LED indices along this strip (trunk → leaf order)
    const stripLeds: number[] = [];
    for (const bid of chain) {
      const indices = branchLedIdx.get(bid) || [];
      for (const idx of indices) {
        stripLeds.push(idx);
      }
    }

    // Assign strip ID and signed distance
    // stripLeds[0] = trunk end, stripLeds[last] = leaf tip
    for (let i = 0; i < stripLeds.length; i++) {
      const idx = stripLeds[i];
      const ledIndex = i; // physical LED index from trunk end

      // Only assign if unassigned (trunk LEDs get first strip's assignment)
      if (stripId[idx] === -1) {
        stripId[idx] = s;
        // Root branches: negative distance (tip is far below trunk)
        // Canopy branches: positive distance (tip is far above trunk)
        signedDist[idx] = isRoot ? -((stripLeds.length - 1) - ledIndex) : ledIndex;
      }
    }
  }

  // Normalize signed distances to [0, 1]
  // Most negative (deepest root tip) → 0
  // Most positive (highest canopy tip) → 1
  let minDist = Infinity;
  let maxDist = -Infinity;
  for (let i = 0; i < positions.length; i++) {
    if (signedDist[i] < minDist) minDist = signedDist[i];
    if (signedDist[i] > maxDist) maxDist = signedDist[i];
  }
  const range = maxDist - minDist;

  const globalProgress = new Array<number>(positions.length);
  for (let i = 0; i < positions.length; i++) {
    globalProgress[i] = range > 0 ? (signedDist[i] - minDist) / range : 0;
  }

  return {
    positions,
    branchId,
    branchProgress,
    globalProgress,
    stripId: Array.from(stripId),
  };
}
