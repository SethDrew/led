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
  globalProgress: number[];        // 0-1 physical height (roots=0, canopy tips=1)
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

// Compute LED positions from topology
export function computeLedPositions(topology: Topology): ComputedTopology {
  const positions: [number, number][] = [];
  const branchId: string[] = [];
  const branchProgress: number[] = [];
  // First pass: compute LED positions along branches
  for (const branch of topology.branches) {
    const totalLen = polylineLength(branch.path);

    for (let i = 0; i < branch.ledCount; i++) {
      const t = branch.ledCount > 1 ? i / (branch.ledCount - 1) : 0.5;
      const distance = t * totalLen;
      const pos = interpolateAlongPolyline(branch.path, distance);

      positions.push(pos);
      branchId.push(branch.id);
      branchProgress.push(t);
    }
  }

  // Second pass: compute globalProgress from physical height (Y coordinate)
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [, y] of positions) {
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const yRange = maxY - minY;
  const globalProgress = positions.map(([, y]) =>
    yRange > 0 ? (y - minY) / yRange : 0
  );

  return { positions, branchId, branchProgress, globalProgress };
}
