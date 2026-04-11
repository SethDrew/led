"""
Extract branch topology from Yggdrasil tree mask using skeleton-based analysis.

Approach: skeletonize → build pixel graph → find junctions/endpoints →
trace branches between them → root at trunk → prune noise spurs →
name hierarchically → distribute LEDs.

Usage:
    .venv/bin/python audio-reactive/jam/tools/extract_branch_topology.py [--min-branch-px 40] [--num-leds 3000]
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize, disk, closing
from skimage.measure import approximate_polygon, find_contours
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt, convolve, gaussian_filter1d
from scipy.signal import find_peaks
import networkx as nx


# ============================================================================
# Pixel-level helpers
# ============================================================================

def load_mask(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode == 'RGBA':
        return np.array(img)[:, :, 3] > 128
    return np.array(img.convert('L')) > 128


def pixel_to_topo(row, col, img_h, img_w):
    tx = (col / img_w) * 1.7 - 0.85
    ty = (1 - row / img_h) * 1.5 - 0.55
    return (round(tx, 4), round(ty, 4))


NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]


# ============================================================================
# Core pipeline
# ============================================================================

def extract_skeleton(mask, close_radius=5):
    """Close mask to bridge texture gaps, then skeletonize."""
    closed = closing(mask, disk(close_radius))
    skel = skeletonize(closed)
    dist = distance_transform_edt(mask)
    return skel, dist, closed


def classify_pixels(skel):
    """Find junction (3+ neighbors) and endpoint (1 neighbor) pixels."""
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    n = convolve(skel.astype(int), kernel) * skel
    junctions = set(zip(*np.where(n >= 3)))
    endpoints = set(zip(*np.where(n == 1)))
    return junctions, endpoints


def merge_nearby_nodes(nodes: set, radius=5.0):
    """Cluster nearby junction pixels into single representative nodes."""
    if not nodes:
        return {}, {}
    pts = np.array(list(nodes))
    used = set()
    clusters = {}  # cluster_id → centroid (row, col)
    pixel_to_cluster = {}  # (row, col) → cluster_id

    for i, p in enumerate(pts):
        if i in used:
            continue
        dists = np.linalg.norm(pts - p, axis=1)
        members = np.where(dists <= radius)[0]
        centroid = tuple(pts[members].mean(axis=0))
        cid = len(clusters)
        clusters[cid] = centroid
        for m in members:
            used.add(m)
            pixel_to_cluster[tuple(pts[m])] = cid

    return clusters, pixel_to_cluster


def trace_branches(skel, junctions_merged, endpoints, pixel_to_cluster):
    """
    Walk the skeleton to find branches: paths between special nodes
    (junctions or endpoints) through chains of degree-2 pixels.

    Returns list of (start_node_id, end_node_id, pixel_path).
    Node IDs: 'j{cluster_id}' for junctions, 'e{row}_{col}' for endpoints.
    """
    h, w = skel.shape

    # Map every junction pixel to its merged node ID
    special = {}  # (row, col) → node_id
    for px, cid in pixel_to_cluster.items():
        special[px] = f'j{cid}'
    for ep in endpoints:
        special[ep] = f'e{ep[0]}_{ep[1]}'

    all_special = set(special.keys())

    # For each special pixel, walk outward along skeleton until we hit another special pixel
    visited_branches = set()  # frozenset(start_id, end_id) to avoid duplicates
    branches = []

    for start_px, start_id in special.items():
        r, c = start_px
        # Try each 8-neighbor
        for dr, dc in NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not skel[nr, nc]:
                continue

            next_px = (nr, nc)
            if next_px in all_special:
                # Direct junction-to-junction connection
                end_id = special[next_px]
                if start_id == end_id:
                    continue  # Self-loop from merged junction pixels
                branch_key = frozenset([start_id, end_id])
                if branch_key not in visited_branches:
                    visited_branches.add(branch_key)
                    branches.append((start_id, end_id, [start_px, next_px]))
                continue

            # Walk through degree-2 chain
            path = [start_px, next_px]
            visited_walk = {start_px, next_px}

            while True:
                cr, cc = path[-1]
                found_next = False
                for dr2, dc2 in NEIGHBORS_8:
                    nr2, nc2 = cr + dr2, cc + dc2
                    nxt = (nr2, nc2)
                    if nxt in visited_walk:
                        continue
                    if not (0 <= nr2 < h and 0 <= nc2 < w) or not skel[nr2, nc2]:
                        continue

                    if nxt in all_special:
                        # Reached another special pixel
                        path.append(nxt)
                        end_id = special[nxt]
                        if start_id != end_id:
                            branch_key = frozenset([start_id, end_id])
                            if branch_key not in visited_branches:
                                visited_branches.add(branch_key)
                                branches.append((start_id, end_id, path[:]))
                        found_next = False  # Done with this walk
                        break
                    else:
                        path.append(nxt)
                        visited_walk.add(nxt)
                        found_next = True
                        break  # Continue walking from the new pixel

                if not found_next:
                    break  # Dead end or reached a special pixel

    return branches


def build_branch_graph(branches, junctions_merged, endpoints, dist_transform):
    """Build a NetworkX graph from traced branches."""
    G = nx.Graph()

    # Add junction nodes
    for cid, centroid in junctions_merged.items():
        nid = f'j{cid}'
        G.add_node(nid, pos=np.array(centroid), is_junction=True)

    # Add endpoint nodes
    for ep in endpoints:
        nid = f'e{ep[0]}_{ep[1]}'
        G.add_node(nid, pos=np.array(ep, dtype=float), is_junction=False)

    # Add edges
    for start_id, end_id, pixel_path in branches:
        length = len(pixel_path)
        # Mean radius (branch thickness) from distance transform
        radii = [dist_transform[int(p[0]), int(p[1])] for p in pixel_path]
        mean_radius = np.mean(radii) if radii else 0

        # If duplicate edge exists, keep the longer one
        if G.has_edge(start_id, end_id):
            if G[start_id][end_id]['length'] >= length:
                continue

        G.add_edge(start_id, end_id,
                   pixels=pixel_path, length=length, mean_radius=mean_radius)

    # Remove isolated nodes (no edges)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    return G


def connect_components(G, max_gap=20.0, root_max_gap=None, trunk_row=None):
    """Connect nearby disconnected components by adding bridge edges.

    Uses a larger gap threshold for root-region components (where at least one
    node is below trunk_row) to bridge the wider gaps in woodcut texture.
    """
    from scipy.spatial import KDTree

    if root_max_gap is None:
        root_max_gap = max_gap

    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return G

    # Build a KDTree per component
    comp_data = []
    for comp in components:
        nodes = list(comp)
        positions = np.array([G.nodes[n]['pos'] for n in nodes])
        comp_data.append((nodes, positions))

    bridges_added = 0
    # For each pair of components, find closest nodes
    # More efficient: build one KDTree with all nodes, tag by component
    all_nodes = []
    all_positions = []
    comp_labels = []
    for ci, (nodes, positions) in enumerate(comp_data):
        all_nodes.extend(nodes)
        all_positions.extend(positions)
        comp_labels.extend([ci] * len(nodes))

    all_positions = np.array(all_positions)
    comp_labels = np.array(comp_labels)

    # Determine which components are in root region
    comp_is_root = {}
    if trunk_row is not None:
        for ci, (nodes, positions) in enumerate(comp_data):
            comp_is_root[ci] = any(pos[0] > trunk_row for pos in positions)
    else:
        comp_is_root = {ci: False for ci in range(len(comp_data))}

    tree = KDTree(all_positions)
    effective_max_gap = max(max_gap, root_max_gap)

    # For each node, find nearest node in a different component
    connected_pairs = set()
    for i, pos in enumerate(all_positions):
        # Query nearest neighbors
        dists, indices = tree.query(pos, k=10)
        for d, j in zip(dists, indices):
            if d > effective_max_gap:
                break
            if comp_labels[i] != comp_labels[j]:
                ci, cj = comp_labels[i], comp_labels[j]
                # Use larger gap if either component is in root region
                gap_threshold = root_max_gap if (comp_is_root.get(ci) or comp_is_root.get(cj)) else max_gap
                if d > gap_threshold:
                    continue
                pair = frozenset([ci, cj])
                if pair not in connected_pairs:
                    connected_pairs.add(pair)
                    n1, n2 = all_nodes[i], all_nodes[j]
                    G.add_edge(n1, n2, pixels=[(all_positions[i][0], all_positions[i][1]),
                                                (all_positions[j][0], all_positions[j][1])],
                               length=d, mean_radius=1.0)
                    bridges_added += 1

    print(f'   Connected {bridges_added} component pairs (canopy gap={max_gap}px, root gap={root_max_gap}px)')
    return G


def prepare_tree(G, mask_shape):
    """Connect nearby components, then extract largest."""
    trunk_row = int(mask_shape[0] * 0.55)

    # First, connect nearby disconnected components
    # Use larger gap for root region where woodcut texture creates wider gaps
    G = connect_components(G, max_gap=50.0, root_max_gap=80.0, trunk_row=trunk_row)

    # Now extract largest component
    components = list(nx.connected_components(G))
    if not components:
        raise ValueError("No connected components in skeleton graph")
    largest = max(components, key=len)
    G = G.subgraph(largest).copy()
    print(f'   Largest component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
    print(f'   ({len(components)} total components, dropped {len(components)-1} small ones)')
    return G


def break_cycles(G):
    """Break cycles by removing thinnest edges."""
    cycles_broken = 0
    while True:
        try:
            cycle = nx.find_cycle(G)
            # Remove edge with smallest mean_radius in this cycle
            edge_to_remove = min(cycle,
                                 key=lambda e: G[e[0]][e[1]].get('mean_radius', 999))
            G.remove_edge(edge_to_remove[0], edge_to_remove[1])
            cycles_broken += 1
        except nx.NetworkXNoCycle:
            break
    print(f'   Broke {cycles_broken} cycles')
    return G


def root_and_direct(G, mask_shape):
    """Root graph at trunk junction, return directed tree."""
    h, w = mask_shape
    trunk_target = np.array([h * 0.55, w * 0.5])

    # Find closest junction to trunk area
    best_node = None
    best_dist = np.inf
    for node in G.nodes():
        if G.nodes[node].get('is_junction', False):
            pos = G.nodes[node]['pos']
            d = np.linalg.norm(pos - trunk_target)
            if d < best_dist:
                best_dist = d
                best_node = node

    if best_node is None:
        best_node = list(G.nodes())[0]

    print(f'   Root: {best_node} (dist={best_dist:.0f}px from trunk center)')

    # BFS to create directed tree
    T = nx.DiGraph()
    visited = {best_node}
    queue = deque([best_node])

    while queue:
        node = queue.popleft()
        T.add_node(node, **G.nodes[node])
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                T.add_edge(node, neighbor, **G[node][neighbor])
                queue.append(neighbor)

    return T, best_node


def prune_spurs(T, min_branch_px, root_min_branch_px=None, trunk_row=None):
    """Iteratively remove leaf branches shorter than threshold.

    Uses a lower threshold for root-region nodes (below trunk_row) to preserve
    short but real root segments that arise from woodcut texture.
    """
    if root_min_branch_px is None:
        root_min_branch_px = min_branch_px

    total_pruned = 0
    canopy_pruned = 0
    root_pruned = 0
    changed = True
    while changed:
        changed = False
        leaves = [n for n in T.nodes()
                  if T.out_degree(n) == 0 and T.in_degree(n) == 1]
        for leaf in leaves:
            parent = list(T.predecessors(leaf))[0]
            length = T[parent][leaf].get('length', 0)
            # Use lower threshold for root region
            pos = T.nodes[leaf]['pos']
            is_root = trunk_row is not None and pos[0] > trunk_row
            threshold = root_min_branch_px if is_root else min_branch_px
            if length < threshold:
                T.remove_node(leaf)
                changed = True
                total_pruned += 1
                if is_root:
                    root_pruned += 1
                else:
                    canopy_pruned += 1
    print(f'   Pruned {total_pruned} short spurs (canopy={canopy_pruned} @{min_branch_px}px, root={root_pruned} @{root_min_branch_px}px)')
    print(f'   Remaining: {T.number_of_nodes()} nodes, {T.number_of_edges()} edges')
    return T


def collapse_chains(T):
    """Collapse linear chain segments where a node has exactly 1 child.

    When parent→node→child forms a chain (node has in_degree=1, out_degree=1),
    remove node and connect parent directly to child, concatenating the pixel
    paths along the edges.
    """
    collapsed = 0
    changed = True
    while changed:
        changed = False
        for node in list(T.nodes()):
            if node not in T:
                continue
            if T.in_degree(node) != 1 or T.out_degree(node) != 1:
                continue
            # node is a chain link: parent → node → child
            parent = list(T.predecessors(node))[0]
            child = list(T.successors(node))[0]

            # Concatenate pixel paths (drop duplicate junction point)
            parent_edge_pixels = T[parent][node].get('pixels', [])
            child_edge_pixels = T[node][child].get('pixels', [])
            combined_pixels = list(parent_edge_pixels) + list(child_edge_pixels[1:]) if child_edge_pixels else list(parent_edge_pixels)

            combined_length = T[parent][node].get('length', 0) + T[node][child].get('length', 0)
            # Weighted average radius
            len1 = T[parent][node].get('length', 1)
            len2 = T[node][child].get('length', 1)
            r1 = T[parent][node].get('mean_radius', 0)
            r2 = T[node][child].get('mean_radius', 0)
            combined_radius = (r1 * len1 + r2 * len2) / max(len1 + len2, 1)

            # Remove node, add direct edge
            T.remove_node(node)
            T.add_edge(parent, child, pixels=combined_pixels,
                       length=combined_length, mean_radius=combined_radius)
            collapsed += 1
            changed = True

    print(f'   Collapsed {collapsed} chain segments')
    print(f'   Remaining: {T.number_of_nodes()} nodes, {T.number_of_edges()} edges')
    return T


def filter_outside_mask(T, mask, min_inside_fraction=0.5,
                        root_min_inside_fraction=None, trunk_row=None):
    """Remove branches whose pixel paths mostly fall outside the tree mask.

    For each edge (parent→child), sample points along the pixel path and check
    what fraction are inside the mask (True pixels). Remove nodes where the
    incoming edge has less than min_inside_fraction of its path inside the mask.
    Cascade: removing a node also removes all its descendants.

    Uses a lower threshold for root-region branches (below trunk_row) because
    roots trace along mask edges in the woodcut texture.
    """
    if root_min_inside_fraction is None:
        root_min_inside_fraction = min_inside_fraction

    root_nodes = [n for n in T.nodes() if T.in_degree(n) == 0]
    if not root_nodes:
        return T

    # Find nodes to remove (skip the root — it has no incoming edge)
    nodes_to_remove = set()
    h, w = mask.shape
    for node in T.nodes():
        preds = list(T.predecessors(node))
        if not preds:
            continue  # root node
        parent = preds[0]
        pixels = T[parent][node].get('pixels', [])
        if len(pixels) == 0:
            continue

        # Check what fraction of path pixels are inside the mask
        inside = 0
        total = len(pixels)
        for r, c in pixels:
            ri, ci = int(round(r)), int(round(c))
            if 0 <= ri < h and 0 <= ci < w and mask[ri, ci]:
                inside += 1

        fraction = inside / total if total > 0 else 0

        # Use lower threshold for root region
        pos = T.nodes[node]['pos']
        is_root = trunk_row is not None and pos[0] > trunk_row
        threshold = root_min_inside_fraction if is_root else min_inside_fraction
        if fraction < threshold:
            nodes_to_remove.add(node)

    # Cascade: if a node is removed, all its descendants must go too
    all_to_remove = set()
    for node in nodes_to_remove:
        all_to_remove.add(node)
        all_to_remove.update(nx.descendants(T, node))

    T.remove_nodes_from(all_to_remove)
    print(f'   Removed {len(all_to_remove)} branches outside mask (canopy@{min_inside_fraction:.0%}, root@{root_min_inside_fraction:.0%})')
    print(f'   Remaining: {T.number_of_nodes()} nodes, {T.number_of_edges()} edges')
    return T


def name_branches(T, root_node, mask_shape):
    """Assign hierarchical names based on position and depth."""
    h, w = mask_shape
    names = {}
    parent_map = {}
    tier_counters = defaultdict(int)

    queue = deque([(root_node, 0, None)])
    while queue:
        node, depth, parent_node = queue.popleft()
        if parent_node is None:
            branch_id = 'trunk'
        else:
            pos = T.nodes[node]['pos']
            parent_pos = T.nodes[parent_node]['pos']
            row, col = pos[0], pos[1]

            # Side: left/right relative to parent
            side = 'left' if col < parent_pos[1] else 'right'

            # Tier: based on y position and depth
            if row > h * 0.55:  # below trunk center = roots
                tier = 'root'
            elif depth == 1:
                tier = 'limb'
            elif depth <= 3:
                tier = 'branch'
            else:
                tier = 'twig'

            key = f'{tier}_{side}'
            idx = tier_counters[key]
            tier_counters[key] += 1
            branch_id = f'{tier}_{side}_{idx}'

        names[node] = branch_id
        parent_map[node] = parent_node

        for child in T.successors(node):
            queue.append((child, depth + 1, node))

    return names, parent_map


def build_paths(T, names, img_h, img_w, rdp_epsilon=3.0):
    """Extract each edge's pixel path, simplify with RDP, convert to topology coords."""
    paths = {}

    for node in T.nodes():
        # Get the edge from parent → this node
        preds = list(T.predecessors(node))
        if not preds:
            # Root node: just its position
            pos = T.nodes[node]['pos']
            paths[node] = [pixel_to_topo(pos[0], pos[1], img_h, img_w)]
            continue

        parent = preds[0]
        pixels = T[parent][node].get('pixels', [])

        if len(pixels) < 2:
            pos = T.nodes[node]['pos']
            parent_pos = T.nodes[parent]['pos']
            pixels = [(parent_pos[0], parent_pos[1]), (pos[0], pos[1])]

        pixel_arr = np.array(pixels, dtype=float)

        # RDP simplification
        if len(pixel_arr) > 2:
            simplified = approximate_polygon(pixel_arr, rdp_epsilon)
        else:
            simplified = pixel_arr

        # Convert to topology coordinates
        topo_path = [pixel_to_topo(p[0], p[1], img_h, img_w) for p in simplified]
        paths[node] = topo_path

    return paths


def allocate_leds(T, paths, num_leds):
    """Distribute LEDs proportional to branch arc length."""
    arc_lengths = {}
    for node, path in paths.items():
        length = sum(
            np.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
            for i in range(1, len(path))
        )
        arc_lengths[node] = length

    total = sum(arc_lengths.values())
    if total == 0:
        return {n: max(1, num_leds // len(paths)) for n in paths}

    # 1 LED minimum per branch, rest proportional
    led_counts = {}
    remaining = num_leds - len(paths)
    for node in paths:
        extra = int(round(remaining * arc_lengths[node] / total)) if total > 0 else 0
        led_counts[node] = 1 + max(0, extra)

    # Adjust to exact target
    actual = sum(led_counts.values())
    diff = num_leds - actual
    sorted_nodes = sorted(paths.keys(), key=lambda n: arc_lengths.get(n, 0), reverse=True)
    for node in sorted_nodes:
        if diff == 0:
            break
        if diff > 0:
            led_counts[node] += 1
            diff -= 1
        elif led_counts[node] > 1:
            led_counts[node] -= 1
            diff += 1

    return led_counts


def build_topology_json(T, names, parent_map, paths, led_counts):
    """Build the final Topology JSON."""
    branches = []
    for node in T.nodes():
        parent_node = parent_map[node]
        branches.append({
            'id': names[node],
            'parent': names[parent_node] if parent_node is not None else None,
            'path': paths[node],
            'ledCount': led_counts[node],
        })

    return {
        'name': 'yggdrasil',
        'numLeds': sum(led_counts.values()),
        'branches': branches,
    }


def draw_overlay(mask, T, names, paths, output_path, img_h, img_w):
    """Diagnostic overlay: tree mask + colored branches + junction dots + labels."""
    img = Image.new('RGB', (img_w, img_h), 'black')

    # Draw mask as gray
    mask_arr = np.array(img)
    mask_arr[mask] = [100, 100, 100]
    img = Image.fromarray(mask_arr)
    draw = ImageDraw.Draw(img)

    palette = [
        '#ff0000', '#00ff00', '#0088ff', '#ffaa00', '#ff00ff',
        '#00ffcc', '#ff4444', '#44ff44', '#4488ff', '#ffcc44',
        '#ff44ff', '#44ffcc', '#aa0000', '#00aa00', '#0044aa',
        '#aa8800', '#aa00aa', '#00aa88',
    ]

    for i, node in enumerate(T.nodes()):
        path = paths[node]
        color = palette[i % len(palette)]

        # Convert topology coords back to pixels for drawing
        px_path = []
        for tx, ty in path:
            col = int((tx + 0.85) / 1.7 * img_w)
            row = int((1 - (ty + 0.55) / 1.5) * img_h)
            px_path.append((col, row))

        # Draw branch line
        for j in range(1, len(px_path)):
            draw.line([px_path[j-1], px_path[j]], fill=color, width=4)

        # Draw tip dot
        if px_path:
            x, y = px_path[-1]
            draw.ellipse([x-4, y-4, x+4, y+4], fill=color)

    # Draw junction nodes as white dots
    for node in T.nodes():
        if T.nodes[node].get('is_junction', False):
            pos = T.nodes[node]['pos']
            x, y = int(pos[1]), int(pos[0])
            draw.ellipse([x-3, y-3, x+3, y+3], fill='white')

    img.save(output_path)
    print(f'   Saved overlay to {output_path}')


def _topo_to_pixel(tx, ty, img_w, img_h):
    """Convert topology coords back to pixel coords for drawing."""
    col = int((tx + 0.85) / 1.7 * img_w)
    row = int((1 - (ty + 0.55) / 1.5) * img_h)
    return (col, row)


def draw_overlay_combined(mask, T, names, paths, root_arcs, output_path, img_h, img_w):
    """Diagnostic overlay: canopy branches (colored) + root arcs (warm tones)."""
    img = Image.new('RGB', (img_w, img_h), 'black')

    mask_arr = np.array(img)
    mask_arr[mask] = [100, 100, 100]
    img = Image.fromarray(mask_arr)
    draw = ImageDraw.Draw(img)

    canopy_palette = [
        '#ff0000', '#00ff00', '#0088ff', '#ffaa00', '#ff00ff',
        '#00ffcc', '#ff4444', '#44ff44', '#4488ff', '#ffcc44',
        '#ff44ff', '#44ffcc', '#aa0000', '#00aa00', '#0044aa',
        '#aa8800', '#aa00aa', '#00aa88',
    ]

    # Draw canopy branches
    for i, node in enumerate(T.nodes()):
        path = paths[node]
        color = canopy_palette[i % len(canopy_palette)]
        px_path = [_topo_to_pixel(tx, ty, img_w, img_h) for tx, ty in path]

        for j in range(1, len(px_path)):
            draw.line([px_path[j-1], px_path[j]], fill=color, width=4)
        if px_path:
            x, y = px_path[-1]
            draw.ellipse([x-4, y-4, x+4, y+4], fill=color)

    # Draw canopy junction nodes as white dots
    for node in T.nodes():
        if T.nodes[node].get('is_junction', False):
            pos = T.nodes[node]['pos']
            x, y = int(pos[1]), int(pos[0])
            draw.ellipse([x-3, y-3, x+3, y+3], fill='white')

    # Draw root arcs in warm/earth tones, thicker lines
    root_palette = [
        '#ff6600', '#cc3300', '#ff9933', '#994400', '#ffcc00',
        '#cc6600', '#ff4400', '#dd5500', '#ee7700', '#bb4400',
        '#ff8800', '#aa5500', '#dd3300', '#cc7700', '#ee5500',
    ]
    for i, arc in enumerate(root_arcs):
        color = root_palette[i % len(root_palette)]
        px_path = [_topo_to_pixel(tx, ty, img_w, img_h) for tx, ty in arc['path']]

        for j in range(1, len(px_path)):
            draw.line([px_path[j-1], px_path[j]], fill=color, width=5)
        # Draw tip dot (last point = tip since path is trunk→tip)
        if px_path:
            x, y = px_path[-1]
            draw.ellipse([x-5, y-5, x+5, y+5], fill=color)
            # Draw trunk anchor dot (first point)
            x0, y0 = px_path[0]
            draw.ellipse([x0-3, y0-3, x0+3, y0+3], fill='white')

    img.save(output_path)
    print(f'   Saved overlay to {output_path}')


# ============================================================================
# Root arc extraction (cost-surface path tracing)
# ============================================================================

def extract_root_arcs(mask, trunk_row, img_h, img_w, close_radius=9,
                      rdp_epsilon=10.0, min_tip_dist=300, min_path_px=80):
    """Extract root arcs as clean linear paths from trunk to tip.

    Uses skeleton endpoints in the root region as tip candidates, then traces
    cost-surface shortest paths from each tip back to the trunk anchor.

    Returns list of dicts: {id, parent, path (topo coords)}.
    """
    # 1. Isolate root region and close gaps
    root_mask = np.zeros_like(mask)
    root_mask[trunk_row:, :] = mask[trunk_row:, :]
    root_closed = closing(root_mask, disk(close_radius))

    # 2. Distance transform of closed root mask
    dist = distance_transform_edt(root_closed)

    # 3. Skeletonize root region to find tip endpoints
    root_skel = skeletonize(root_closed)
    # Classify skeleton pixels
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    n_neighbors = convolve(root_skel.astype(int), kernel) * root_skel
    endpoint_mask = (n_neighbors == 1)
    endpoint_coords = list(zip(*np.where(endpoint_mask)))
    print(f'   Root skeleton: {root_skel.sum()} pixels, {len(endpoint_coords)} endpoints')

    # 4. Filter endpoints: keep only those far enough from trunk center
    trunk_center = np.array([trunk_row, img_w / 2.0])
    tips = []
    for r, c in endpoint_coords:
        d = np.sqrt((r - trunk_center[0])**2 + (c - trunk_center[1])**2)
        if d >= min_tip_dist:
            tips.append((r, c, d))

    # Sort by distance (farthest first) and de-duplicate nearby tips
    tips.sort(key=lambda t: -t[2])
    filtered_tips = []
    min_tip_separation = 200  # pixels between distinct root tips
    for r, c, d in tips:
        too_close = False
        for fr, fc, _ in filtered_tips:
            if np.sqrt((r - fr)**2 + (c - fc)**2) < min_tip_separation:
                too_close = True
                break
        if not too_close:
            filtered_tips.append((r, c, d))

    print(f'   {len(tips)} endpoints far enough from trunk, {len(filtered_tips)} after de-duplication')

    # 5. Build cost surface — infinite cost outside mask prevents paths leaving
    cost = 1.0 / (dist + 1.0)
    cost[~root_closed] = 1e6

    # 6. Trunk anchor: closest root-mask pixel to trunk center
    trunk_anchor = (trunk_row, img_w // 2)
    if not root_closed[trunk_anchor[0], trunk_anchor[1]]:
        best_d = np.inf
        for dr in range(-40, 41):
            for dc in range(-40, 41):
                r, c = trunk_anchor[0] + dr, trunk_anchor[1] + dc
                if 0 <= r < img_h and 0 <= c < img_w and root_closed[r, c]:
                    dd = dr * dr + dc * dc
                    if dd < best_d:
                        best_d = dd
                        trunk_anchor = (r, c)

    print(f'   Trunk anchor: row={trunk_anchor[0]}, col={trunk_anchor[1]}')

    # 7. Trace paths from each tip to trunk
    root_arcs = []
    rejected_inside = 0
    rejected_short = 0
    for tip_r, tip_c, tip_dist in filtered_tips:
        tip_rc = (max(0, min(img_h - 1, tip_r)),
                  max(0, min(img_w - 1, tip_c)))

        try:
            path_pixels, path_cost = route_through_array(
                cost, tip_rc, trunk_anchor, fully_connected=True
            )
        except Exception as e:
            print(f'   Warning: path tracing failed for tip ({tip_r},{tip_c}): {e}')
            continue

        path_pixels = np.array(path_pixels)

        # Clip path at mask boundary.
        # Path goes tip→trunk. Reverse to trunk→tip for clipping, then
        # walk from trunk and truncate when we hit a long outside run.
        path_pixels = path_pixels[::-1]  # now trunk→tip

        inside_flags = np.array([
            root_closed[int(r), int(c)]
            if 0 <= int(r) < img_h and 0 <= int(c) < img_w else False
            for r, c in path_pixels
        ])

        # Walk from trunk end. Allow short outside gaps (up to max_gap_px)
        # but stop at the first gap longer than that.
        max_gap_px = 15
        outside_run = 0
        clip_end = len(path_pixels)
        for idx in range(len(path_pixels)):
            if inside_flags[idx]:
                outside_run = 0
            else:
                outside_run += 1
                if outside_run > max_gap_px:
                    clip_end = idx - max_gap_px  # back up to start of gap
                    break

        path_pixels = path_pixels[:clip_end]
        # path_pixels is now trunk→tip (already reversed above)

        if len(path_pixels) < min_path_px:
            rejected_short += 1
            continue

        # Recompute inside flags after clipping
        inside_count = sum(
            1 for r, c in path_pixels
            if 0 <= int(r) < img_h and 0 <= int(c) < img_w and root_closed[int(r), int(c)]
        )
        inside_frac = inside_count / len(path_pixels)
        if inside_frac < 0.80:
            rejected_inside += 1
            continue

        # RDP simplification
        if len(path_pixels) > 2:
            simplified = approximate_polygon(path_pixels.astype(float), rdp_epsilon)
        else:
            simplified = path_pixels.astype(float)

        # Already trunk→tip order
        topo_path = [pixel_to_topo(p[0], p[1], img_h, img_w) for p in simplified]

        side = 'left' if tip_c < img_w / 2 else 'right'

        root_arcs.append({
            'path': topo_path,
            'side': side,
            'tip_rc': (tip_r, tip_c),
        })

    print(f'   Rejected: {rejected_inside} below 80% inside mask, {rejected_short} too short')

    # Name roots by side, ordered by x position
    left_roots = sorted([r for r in root_arcs if r['side'] == 'left'],
                        key=lambda r: r['tip_rc'][1])
    right_roots = sorted([r for r in root_arcs if r['side'] == 'right'],
                         key=lambda r: r['tip_rc'][1])

    named_arcs = []
    for i, r in enumerate(left_roots):
        named_arcs.append({
            'id': f'root_left_{i}',
            'parent': 'trunk',
            'path': r['path'],
        })
    for i, r in enumerate(right_roots):
        named_arcs.append({
            'id': f'root_right_{i}',
            'parent': 'trunk',
            'path': r['path'],
        })

    print(f'   Extracted {len(named_arcs)} root arcs ({len(left_roots)} left, {len(right_roots)} right)')
    return named_arcs


# ============================================================================
# Main
# ============================================================================

def allocate_leds_for_all(canopy_branches, root_branches, num_leds):
    """Distribute LEDs across all branches proportional to arc length."""
    all_branches = canopy_branches + root_branches

    arc_lengths = {}
    for b in all_branches:
        path = b['path']
        length = sum(
            np.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
            for i in range(1, len(path))
        )
        arc_lengths[b['id']] = length

    total = sum(arc_lengths.values())
    if total == 0:
        return {b['id']: max(1, num_leds // len(all_branches)) for b in all_branches}

    # 1 LED minimum per branch, rest proportional
    led_counts = {}
    remaining = num_leds - len(all_branches)
    for b in all_branches:
        extra = int(round(remaining * arc_lengths[b['id']] / total)) if total > 0 else 0
        led_counts[b['id']] = 1 + max(0, extra)

    # Adjust to exact target
    actual = sum(led_counts.values())
    diff = num_leds - actual
    sorted_ids = sorted(arc_lengths.keys(), key=lambda k: arc_lengths[k], reverse=True)
    for bid in sorted_ids:
        if diff == 0:
            break
        if diff > 0:
            led_counts[bid] += 1
            diff -= 1
        elif led_counts[bid] > 1:
            led_counts[bid] -= 1
            diff += 1

    return led_counts


def main():
    parser = argparse.ArgumentParser(description='Extract branch topology from tree mask')
    parser.add_argument('--min-branch-px', type=float, default=40,
                        help='Minimum branch length in pixels (default: 40)')
    parser.add_argument('--num-leds', type=int, default=3000,
                        help='Target number of LEDs (default: 3000)')
    args = parser.parse_args()

    jam_root = Path(__file__).parent.parent
    mask_path = jam_root / 'client' / 'public' / 'yggdrasil-mask.png'

    print('=' * 70)
    print('Branch Topology Extraction Pipeline')
    print('=' * 70)

    # 1. Load mask
    print(f'\n1. Loading mask')
    mask = load_mask(mask_path)
    img_h, img_w = mask.shape
    print(f'   {img_w}x{img_h}, {mask.sum()} tree pixels')

    trunk_row = int(img_h * 0.55)

    # 2. Skeleton + distance transform
    print('\n2. Skeletonizing (with morphological closing)...')
    skel, dist, closed_mask = extract_skeleton(mask, close_radius=7)
    print(f'   {skel.sum()} skeleton pixels')

    # ---- FULL SKELETON PIPELINE (steps 3-12) ----
    # Run the original pipeline on the full skeleton (canopy + roots),
    # then strip root-named branches and replace with root arcs.

    # 3. Classify pixels
    print('\n3. Classifying skeleton pixels...')
    junctions, endpoints = classify_pixels(skel)
    print(f'   {len(junctions)} junction pixels, {len(endpoints)} endpoint pixels')

    # 4. Merge nearby junctions
    print('\n4. Merging nearby junctions (radius=5px)...')
    merged_juncs, pixel_to_cluster = merge_nearby_nodes(junctions, radius=5.0)
    print(f'   {len(junctions)} → {len(merged_juncs)} merged junction nodes')

    # 5. Trace branches
    print('\n5. Tracing branches between nodes...')
    branches = trace_branches(skel, merged_juncs, endpoints, pixel_to_cluster)
    print(f'   {len(branches)} branch segments traced')

    # 6. Build graph
    print('\n6. Building graph...')
    G = build_branch_graph(branches, merged_juncs, endpoints, dist)
    print(f'   {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

    # 7. Extract largest component
    print('\n7. Extracting largest component...')
    G = prepare_tree(G, mask.shape)
    print('\n8. Breaking cycles...')
    G = break_cycles(G)

    # 9. Root at trunk
    print('\n9. Rooting tree at trunk...')
    T, root = root_and_direct(G, mask.shape)

    # 10. Prune short spurs
    print('\n10. Pruning short spurs...')
    T = prune_spurs(T, args.min_branch_px, root_min_branch_px=30, trunk_row=trunk_row)

    # 11. Collapse linear chains
    print('\n11. Collapsing linear chains...')
    T = collapse_chains(T)

    # 12. Filter branches outside mask
    print('\n12. Filtering branches outside mask...')
    T = filter_outside_mask(T, mask, min_inside_fraction=0.5,
                            root_min_inside_fraction=0.35, trunk_row=trunk_row)

    # 13. Name all branches (including skeleton-based roots)
    print('\n13. Naming branches...')
    names, parent_map = name_branches(T, root, mask.shape)
    print(f'   {len(names)} branches named')

    # 14. Build paths
    print('\n14. Simplifying polylines + converting coordinates...')
    paths = build_paths(T, names, img_h, img_w)
    print(f'   {len(paths)} paths')

    # Split into canopy (keep) and skeleton-roots (discard)
    canopy_branches = []
    skeleton_root_count = 0
    for node in T.nodes():
        parent_node = parent_map[node]
        bid = names[node]
        if bid.startswith('root_'):
            skeleton_root_count += 1
            continue
        canopy_branches.append({
            'id': bid,
            'parent': names[parent_node] if parent_node is not None else None,
            'path': paths[node],
        })

    # Fix parent references: any canopy branch whose parent was a root_ branch
    # should re-parent to trunk
    root_ids = {names[n] for n in T.nodes() if names[n].startswith('root_')}
    for b in canopy_branches:
        if b['parent'] in root_ids:
            b['parent'] = 'trunk'

    print(f'   Kept {len(canopy_branches)} canopy branches, dropped {skeleton_root_count} skeleton roots')

    # ---- ROOT PIPELINE (cost-surface path tracing) ----
    print('\n--- ROOT ARC PIPELINE ---')
    print('\n15. Extracting root arcs via cost-surface path tracing...')
    root_arcs = extract_root_arcs(mask, trunk_row, img_h, img_w)

    # ---- MERGE & ALLOCATE LEDs ----
    print('\n--- MERGE & LED ALLOCATION ---')
    print(f'\n16. Allocating {args.num_leds} LEDs across {len(canopy_branches)} canopy + {len(root_arcs)} root branches...')
    led_counts = allocate_leds_for_all(canopy_branches, root_arcs, args.num_leds)

    # Build final topology
    all_branch_dicts = []
    for b in canopy_branches:
        all_branch_dicts.append({
            'id': b['id'],
            'parent': b['parent'],
            'path': b['path'],
            'ledCount': led_counts[b['id']],
        })
    for b in root_arcs:
        all_branch_dicts.append({
            'id': b['id'],
            'parent': b['parent'],
            'path': b['path'],
            'ledCount': led_counts[b['id']],
        })

    topology = {
        'name': 'yggdrasil',
        'numLeds': sum(led_counts.values()),
        'branches': all_branch_dicts,
    }

    # 17. Save
    out_json = jam_root / 'client' / 'src' / 'topology' / 'yggdrasil-branches.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(topology, f, indent=2)
    print(f'\n   Topology saved to {out_json}')

    # 18. Overlay (use canopy tree T + root arcs for drawing)
    print('\n18. Drawing diagnostic overlay...')
    overlay_path = jam_root / 'tools' / 'branch-overlay.png'
    draw_overlay_combined(mask, T, names, paths, root_arcs, overlay_path, img_h, img_w)

    # Stats
    print('\n' + '=' * 70)
    print('RESULTS')
    print('=' * 70)
    n_canopy = len(canopy_branches)
    n_roots = len(root_arcs)
    print(f'Canopy branches: {n_canopy}')
    print(f'Root arcs: {n_roots}')
    print(f'Total branches: {len(topology["branches"])}')
    print(f'LEDs: {topology["numLeds"]}')

    # Depth distribution
    depth_of = {}
    for b in topology['branches']:
        if b['parent'] is None:
            depth_of[b['id']] = 0
        else:
            depth_of[b['id']] = depth_of.get(b['parent'], 0) + 1

    depth_counts = defaultdict(int)
    for d in depth_of.values():
        depth_counts[d] += 1
    for d in sorted(depth_counts):
        print(f'  Depth {d}: {depth_counts[d]} branches')

    max_depth = max(depth_of.values()) if depth_of else 0
    print(f'Max depth: {max_depth}')

    leds = [b['ledCount'] for b in topology['branches']]
    print(f'LEDs per branch: min={min(leds)}, max={max(leds)}, mean={np.mean(leds):.1f}')


if __name__ == '__main__':
    main()
