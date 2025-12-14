import numpy as np
import math


class State3DQueensBoard:
    """
    State for 3D Queens problem with board constraint.
    Each (i,j) pair has exactly one queen at height k.
    Total queens: Q = N² (one per (i,j) pair).
    """

    def __init__(self, N, heights=None, init_mode="random"):
        """
        Args:
            N: Board size (NxN board, N heights)
            heights: Optional array of shape (N, N) with height k for each (i,j)
                     If None, initialized based on init_mode
            init_mode:
                - "random": random height k for each (i,j)
                - "latin": k = (i + j) mod N
                - "klarner": k = (3i + 5j) mod N (if gcd(N,210)==1)
        """
        self.N = N
        self.Q = N * N
        
        if heights is None:
            if init_mode == "random":
                self.heights = np.random.randint(0, N, size=(N, N))
            elif init_mode == "latin":
                i_grid, j_grid = np.indices((N, N))
                self.heights = (i_grid + j_grid) % N
            elif init_mode == "klarner":
                g = math.gcd(N, 210)
                if g == 1:
                    i_grid, j_grid = np.indices((N, N))
                    self.heights = (3 * i_grid + 5 * j_grid) % N
                else:
                    M = None
                    for m in range(N - 1, 0, -1):
                        if math.gcd(m, 210) == 1:
                            M = m
                            break
                    if M is None:
                        raise ValueError(
                            f"Could not find M < {N} with gcd(M,210)==1 "
                            f"(N={N}, gcd(N,210)={g})."
                        )
                    
                    self.heights = np.zeros((N, N), dtype=int)
                    for i in range(M):
                        for j in range(M):
                            self.heights[i, j] = (3 * i + 5 * j) % M
                    
                    for i in range(N):
                        for j in range(N):
                            if not (i < M and j < M):
                                self.heights[i, j] = np.random.randint(0, N)
            else:
                raise ValueError(f"Unknown init_mode: {init_mode}")
        else:
            heights = np.asarray(heights, dtype=int)
            if heights.shape != (N, N):
                raise ValueError(f"heights must be of shape ({N}, {N}), got {heights.shape}")
            if np.any((heights < 0) | (heights >= N)):
                raise ValueError(f"All heights must be in [0, {N-1}]")
            self.heights = heights.copy()
        
        self._energy = None
    
    def copy(self):
        """Create a deep copy of the state."""
        new_state = State3DQueensBoard(self.N, heights=self.heights.copy())
        new_state._energy = self._energy
        return new_state
    
    def energy(self, recompute=False):
        """Compute energy (number of attacking pairs)."""
        if self._energy is None or recompute:
            self._energy = self._compute_energy()
        return self._energy
    
    def _compute_energy(self):
        """
        Optimized energy computation for board-constrained version.
        Since each (i,j) pair has exactly one queen, we can work directly
        with the heights array and skip same_ij checks.
        """
        N = self.N
        if N < 2:
            return 0
        
        heights = self.heights
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        k_flat = heights.flatten()
        
        di = np.abs(i_flat[:, None] - i_flat[None, :])
        dj = np.abs(j_flat[:, None] - j_flat[None, :])
        dk = np.abs(k_flat[:, None] - k_flat[None, :])
        
        same_ik = (i_flat[:, None] == i_flat[None, :]) & (k_flat[:, None] == k_flat[None, :])
        same_jk = (j_flat[:, None] == j_flat[None, :]) & (k_flat[:, None] == k_flat[None, :])
        
        plane_k_diag = (k_flat[:, None] == k_flat[None, :]) & (di == dj)
        plane_j_diag = (j_flat[:, None] == j_flat[None, :]) & (di == dk)
        plane_i_diag = (i_flat[:, None] == i_flat[None, :]) & (dj == dk)
        
        space_diag = (di == dj) & (dj == dk)
        
        attacked = (
            same_ik
            | same_jk
            | plane_k_diag
            | plane_j_diag
            | plane_i_diag
            | space_diag
        )
        
        upper_triangle = np.triu(attacked, k=1)
        return int(upper_triangle.sum())
    
    def propose_move(self, i, j, new_k):
        """
        Propose moving the queen at (i,j) to a new height new_k.
        
        Args:
            i, j: Board position (i,j)
            new_k: New height for the queen at (i,j)
        
        Returns:
            old_k: Previous height
        """
        if not (0 <= i < self.N and 0 <= j < self.N):
            raise ValueError(f"Invalid position ({i}, {j}) for N={self.N}")
        if not (0 <= new_k < self.N):
            raise ValueError(f"Invalid height {new_k} for N={self.N}")
        
        old_k = self.heights[i, j]
        self.heights[i, j] = new_k
        
        self._energy = None
        
        return old_k
    
    def conflicts_for_position(self, i, j, k=None):
        """
        Optimized conflict computation for a specific position.
        Works directly with heights array without converting to queen positions.
        
        Args:
            i, j: Board position
            k: Height (if None, uses self.heights[i, j])
        
        Returns:
            Number of queens attacking this position
        """
        if k is None:
            k = self.heights[i, j]
        
        N = self.N
        heights = self.heights
        
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        
        qi_flat = i_grid.flatten()
        qj_flat = j_grid.flatten()
        qk_flat = heights.flatten()
        
        di = np.abs(qi_flat - i)
        dj = np.abs(qj_flat - j)
        dk = np.abs(qk_flat - k)
        
        mask = ~((qi_flat == i) & (qj_flat == j))
        
        same_ik = (qi_flat == i) & (qk_flat == k)
        same_jk = (qj_flat == j) & (qk_flat == k)
        plane_k_diag = (qk_flat == k) & (di == dj)
        plane_j_diag = (qj_flat == j) & (di == dk)
        plane_i_diag = (qi_flat == i) & (dj == dk)
        space_diag = (di == dj) & (dj == dk)
        
        attacked = (
            same_ik
            | same_jk
            | plane_k_diag
            | plane_j_diag
            | plane_i_diag
            | space_diag
        )
        
        return int(np.sum(attacked & mask))
