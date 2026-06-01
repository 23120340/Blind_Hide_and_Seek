from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path as _scipy_shortest_path
from environment import Move
from typing import List, Tuple, Optional

# Generic helpers are now imported from utils.py as requested.
import utils

INF_DIST: int = 10_000

_CARDINAL_MOVES: List[Move] = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
_DELTA: dict = {m: m.value for m in Move}   # Move -> (dr, dc)


class Infrastructure:
    """
    One-time preprocessing + O(1) query interface over a fixed map.

    Usage
    -----
    infra = Infrastructure(map_state, pacman_speed=2)
    infra.preprocess()          # call once, at the very first turn
    d = infra.distance(a, b)    # graph distance
    nb = infra.neighbors(cell)  # walkable neighbours
    """

    def __init__(self, map_state: np.ndarray, pacman_speed: int = 2,
                 use_all_pairs: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Parameters
        ----------
        map_state     : 2-D numpy array, 0 = walkable, 1 = wall.
                        Values other than 0/1 are treated as walls (e.g. -1 fog).
        pacman_speed  : maximum straight-line steps Pacman may take per turn.
        use_all_pairs : if True (default), precompute full V×V distance matrix
                        for O(1) lookups. Memory: worst-case on a fully walkable
                        21×21 grid → 441×441 int32 ≈ 760 KB; on the default
                        Pacman map (216 walkable nodes) → 216×216 int32 ≈ 180 KB.
                        Set False to skip matrix; distance() will raise until set
                        back to True and preprocess() re-run. Useful for very large maps.
        cache_dir     : directory to read/write the precomputed graph cache.
                        If None, uses the same directory as this source file.
                        Pass an empty string "" to disable caching entirely.
        """
        raw = np.array(map_state, dtype=np.int8)

        # Input validation
        if raw.ndim != 2:
            raise ValueError(
                f"map_state must be a 2-D array, got shape {raw.shape}"
            )
        # Treat anything != 0 as wall (handles fog=-1, unexpected values)
        self._raw_map = raw
        self._map: np.ndarray = np.where(raw == 0, np.int8(0), np.int8(1))
        self._height, self._width = self._map.shape
        self._pacman_speed: int = max(1, int(pacman_speed))
        self._use_all_pairs: bool = use_all_pairs

        # Cache: None = use file-adjacent dir, "" = disabled
        from pathlib import Path
        if cache_dir is None:
            self._cache_dir: Optional[Path] = Path(__file__).parent
        elif cache_dir == "":
            self._cache_dir = None          # caching disabled
        else:
            self._cache_dir = Path(cache_dir)

        # Built during preprocess()
        self._cell_to_node: dict[Tuple[int, int], int] = {}
        self._node_to_cell: list[Tuple[int, int]] = []
        self._n_nodes: int = 0
        self._dist_matrix: Optional[np.ndarray] = None   # shape (n, n)
        self._neighbors_cache: dict[Tuple[int, int], List[Tuple[Move, Tuple[int, int]]]] = {} # dict of pos -> list of (move, next_pos)

        self.is_preprocessed: bool = False

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    # ------------------------------------------------------------------
    # Public: preprocess  (call ONCE at turn 1)
    # ------------------------------------------------------------------

    def preprocess(self) -> None:
        """
        Build node index, neighbour cache, and distance matrix.
        Must be called before distance() or neighbors() queries.

        Strategy
        --------
        Uses scipy.sparse.csgraph.shortest_path (Dijkstra, C implementation)
        which is ~7-8x faster than pure-Python BFS on the default 21×21 map.

        Flow:
          1. Build node index (walkable cell → int id)
          2. Build neighbours cache (pure numpy, reused by search/eval)
          3. Build CSR sparse adjacency matrix
          4. Run scipy shortest_path once → int32 distance matrix
          5. Save to .npz cache; load on subsequent runs

        If use_all_pairs=False, steps 3-5 are skipped entirely.

        Memory: O(V²) int32 — default map 216×216 ≈ 180 KB;
                worst-case fully walkable 21×21 → 441×441 ≈ 760 KB.
        """
        self._build_node_index()
        self._build_neighbors_cache()

        if not self._use_all_pairs:
            self.is_preprocessed = True
            return

        # ── Try loading from cache ────────────────────────────────────────
        if self._cache_dir is not None:
            loaded = self._load_cache()
            if loaded is not None:
                self._dist_matrix = loaded
                self.is_preprocessed = True
                return

        # ── Cache miss (or disabled): compute from scratch ────────────────
        self._dist_matrix = self._scipy_all_pairs()

        # ── Save to cache for next run ────────────────────────────────────
        if self._cache_dir is not None:
            self._save_cache(self._dist_matrix)

        self.is_preprocessed = True

    # ------------------------------------------------------------------
    # Public: core query API
    # ------------------------------------------------------------------

    def distance(self, cell_a: Tuple[int, int], cell_b: Tuple[int, int]) -> int:
        """
        Graph distance (number of steps) between two walkable cells. O(1).

        Returns a large sentinel (INF_DIST) if the cells are in different
        connected components, or if either cell is a wall / out-of-bounds.

        Raises
        ------
        RuntimeError        – if preprocess() has not been called.
        NotImplementedError – if use_all_pairs=False.
        """
        self._require_preprocessed()
        if self._dist_matrix is None:
            raise NotImplementedError(
                "distance() requires use_all_pairs=True. "
                "Re-create Infrastructure with use_all_pairs=True and call preprocess()."
            )
        na = self._cell_to_node.get(cell_a)
        nb = self._cell_to_node.get(cell_b)
        if na is None or nb is None:
            return INF_DIST
        return int(self._dist_matrix[na, nb])

    def neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Return list of walkable 4-connected neighbours of *cell*.

        Raises RuntimeError if preprocess() has not been called.
        Returns [] for wall cells or out-of-bounds positions.
        """
        # Unpacks positions from the internal Move/Position transition cache.
        self._require_preprocessed()
        cached = self._neighbors_cache.get(cell, [])
        return [pos for _, pos in cached]

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """True iff pos is in-bounds and not a wall (value 0)."""
        return utils.is_walkable(self._map, pos)

    # ------------------------------------------------------------------
    # Ghost (Hider) moves  –  exactly 1 step per turn
    # ------------------------------------------------------------------

    def get_ghost_actions(self, pos: Tuple[int, int],
                          include_stay: bool = True) -> List[Move]:
        """
        Return all legal Move values for the Ghost at *pos*.
        Uses cached move validation for performance.

        Parameters
        ----------
        include_stay : whether to include Move.STAY.
                       Default True — Ghost spec allows staying put.

        A directional move is legal if the destination cell is walkable.
        """
        self._require_preprocessed()
        cached = self._neighbors_cache.get(pos, [])
        actions = [m for m, _ in cached]
        if include_stay:
            actions.append(Move.STAY)
        return actions

    def get_ghost_transitions(self, pos: Tuple[int, int],
                               include_stay: bool = True) -> List[Tuple[Move, Tuple[int, int]]]:
        """
        High-performance method for search/evaluation.
        Returns List of (Move, resulting_pos) for the Ghost at *pos*.
        """
        self._require_preprocessed()
        cached = self._neighbors_cache.get(pos, [])
        if include_stay:
            # We don't cache STAY results because STAY is always legal
            # and it keeps the cache smaller and map-specific.
            return cached + [(Move.STAY, pos)]
        return cached

    def apply_ghost_action(self, pos: Tuple[int, int], move: Move) -> Tuple[int, int]:
        """
        Apply a Ghost move. Returns new position.
        Uses cached transitions if possible, otherwise falls back to utils.
        """
        if move == Move.STAY:
            return pos
        
        # Check cache for this specific move
        self._require_preprocessed()
        cached = self._neighbors_cache.get(pos)
        if cached:
            for m, npos in cached:
                if m == move:
                    return npos
                    
        return utils.apply_move_once(self._map, pos, move)

    # ------------------------------------------------------------------
    # Pacman (Seeker) moves  –  1..pacman_speed straight-line steps
    # ------------------------------------------------------------------

    def get_pacman_actions(self, pos: Tuple[int, int],
                           emit_all_steps: bool = False) -> List[Tuple[Move, int]]:
        """
        Return all legal (Move, steps) actions for Pacman at *pos*.

        Parameters
        ----------
        emit_all_steps : if False (default), only emit the MAX reachable steps
                         for each direction — reduces branching factor from
                         (4 × speed) to 4, critical for Minimax performance.
                         if True, emit every step count 1..max for each direction
                         (useful for exhaustive search or debugging).

        Rules
        -----
        - Pacman moves up to pacman_speed cells in a straight line per turn.
        - Each intermediate cell must be walkable.
        - If the first cell is blocked → direction is skipped entirely.
        - STAY is always included as (Move.STAY, 1).
        """
        actions: List[Tuple[Move, int]] = []
        for move in _CARDINAL_MOVES:
            max_steps = self._max_straight_steps(pos, move)
            if max_steps == 0:
                continue   # first cell blocked — direction illegal
            if emit_all_steps:
                for s in range(1, max_steps + 1):
                    actions.append((move, s))
            else:
                actions.append((move, max_steps))
        actions.append((Move.STAY, 1))
        return actions

    def apply_pacman_action(
        self, pos: Tuple[int, int], action: Tuple[Move, int]
    ) -> Tuple[int, int]:
        """
        Apply a Pacman (Move, steps) action. Returns final position.

        Walks step by step in the given direction for at most `steps` cells,
        stopping if a wall is encountered. Consistent with environment.py's
        _apply_pacman_move() behaviour.
        """
        return utils.apply_pacman_action(self._map, pos, action, self._pacman_speed)

    # ------------------------------------------------------------------
    # Convenience: max reachable straight-line distance
    # ------------------------------------------------------------------

    def max_straight_steps(self, pos: Tuple[int, int], move: Move) -> int:
        """
        How many steps can Pacman take from *pos* in *move* direction
        before hitting a wall or boundary (capped at pacman_speed)?
        Returns 0 if the first cell is blocked.

        Raises ValueError for Move.STAY — straight-line steps are only
        meaningful for cardinal directions.
        """
        if move not in _CARDINAL_MOVES:
            raise ValueError(
                f"max_straight_steps() requires a cardinal move, got {move!r}"
            )
        return self._max_straight_steps(pos, move)


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_node_index(self) -> None:
        self._cell_to_node.clear()
        self._node_to_cell.clear()
        idx = 0
        for r in range(self._height):
            for c in range(self._width):
                if self._map[r, c] == 0:
                    self._cell_to_node[(r, c)] = idx
                    self._node_to_cell.append((r, c))
                    idx += 1
        self._n_nodes = idx

    def _build_neighbors_cache(self) -> None:
        """Pre-cache neighbour lists for every walkable cell."""
        self._neighbors_cache.clear()
        for cell in self._cell_to_node:
            self._neighbors_cache[cell] = self._compute_neighbors(cell)

    def _compute_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[Move, Tuple[int, int]]]:
        r, c = cell
        result = []
        for move in _CARDINAL_MOVES:
            dr, dc = _DELTA[move]
            nb = (r + dr, c + dc)
            if self.is_valid_position(nb):
                result.append((move, nb))
        return result

    def _scipy_all_pairs(self) -> np.ndarray:
        """
        All-pairs shortest path via scipy (C-level Dijkstra on CSR sparse graph).
        ~7-8x faster than pure-Python BFS on the default 21×21 map.

        Returns int32 matrix (n×n); unreachable pairs stored as INF_DIST.
        int32 is used (not int16) to safely handle larger maps where path
        lengths could exceed int16 max (32_767).
        """
        n = self._n_nodes

        # Build sparse adjacency (lil for construction, csr for computation)
        adj = lil_matrix((n, n), dtype=np.float32)
        for cell, nid in self._cell_to_node.items():
            for _, nb in self._neighbors_cache[cell]:
                adj[nid, self._cell_to_node[nb]] = 1.0
        adj_csr = adj.tocsr()

        # scipy returns float64 with np.inf for unreachable pairs
        dist_f = _scipy_shortest_path(adj_csr, method='D',
                                      directed=False, unweighted=True)

        # Convert to int32: inf → INF_DIST
        dist = np.where(np.isinf(dist_f), INF_DIST, dist_f).astype(np.int32)
        return dist

    def _max_straight_steps(self, pos: Tuple[int, int], move: Move) -> int:
        return utils.max_valid_steps(self._map, pos, move, self._pacman_speed)

    def _map_hash(self) -> str:
        """SHA-256 of the binary map content — used as cache filename key."""
        import hashlib
        return hashlib.sha256(self._map.tobytes()).hexdigest()[:16]

    def _cache_path(self) -> "Path":
        from pathlib import Path
        return self._cache_dir / f"graph_cache_{self._map_hash()}.npz"  # type: ignore[operator]

    def _load_cache(self) -> Optional[np.ndarray]:
        """
        Try to load a previously saved distance matrix.
        Returns the matrix on success, None on any failure (missing file,
        shape mismatch, corrupt data).
        Uses context manager to ensure the NpzFile is closed and file
        descriptors are released even in long-running processes.
        """
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                dist = data["dist_matrix"]
                # Sanity-check: shape must match current node count
                if dist.shape != (self._n_nodes, self._n_nodes):
                    return None
                return dist.astype(np.int32)
        except Exception:
            return None

    def _save_cache(self, dist_matrix: np.ndarray) -> None:
        """
        Persist the distance matrix to disk. Silently skips on any I/O error
        so a read-only filesystem never crashes the agent.
        """
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            np.savez_compressed(self._cache_path(), dist_matrix=dist_matrix)
        except Exception:
            pass   # cache save failure is non-fatal

    def _require_preprocessed(self) -> None:
        if not self.is_preprocessed:
            raise RuntimeError(
                "Infrastructure.preprocess() must be called before this query."
            )

    # ------------------------------------------------------------------
    # Debug / testing helpers
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        """Number of walkable cells (graph nodes)."""
        return self._n_nodes

    def all_walkable_cells(self) -> List[Tuple[int, int]]:
        """Return all walkable (row, col) cells in row-major order."""
        return list(self._node_to_cell)
