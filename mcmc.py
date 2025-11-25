import os
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml

import math

class State3DQueens:
    def __init__(self, N, Q=None, positions=None, init_mode="latin"):
        """
        init_mode:
            - "latin": k = (i + j) mod N (deterministic Latin-square style)
            - "klarner": k = (3i + 5j) mod N (requires gcd(N, 210) == 1 for non-attacking theorem)
            - "random": fully random distinct 3D positions
        """
        self.N = N
        if Q is None:
            Q = N * N
        self.Q = Q

        if positions is None:
            if init_mode in ("latin", "klarner"):
                if Q != N * N:
                    raise ValueError(
                        f"{init_mode} initialization assumes Q = N^2, "
                        f"got Q={Q}, N^2={N*N}."
                    )

                # Create full (i,j) grid
                i_grid, j_grid = np.indices((N, N))  # shape (N,N)

                if init_mode == "latin":
                    # Latin square: k = (i + j) mod N
                    k_grid = (i_grid + j_grid) % N
                else:  # "klarner"
                    # Klarner construction: k = (3i + 5j) mod N
                    # For gcd(N, 210) == 1 this is a known non-attacking configuration
                    if math.gcd(N, 210) != 1:
                        print(
                            f"[warning] Klarner construction has no guarantee for N={N} "
                            f"(gcd(N, 210) = {math.gcd(N,210)}). Using it anyway."
                        )
                    k_grid = (3 * i_grid + 5 * j_grid) % N

                # Flatten into (Q, 3)
                i_coords = i_grid.ravel()
                j_coords = j_grid.ravel()
                k_coords = k_grid.ravel()
                self.queens = np.stack([i_coords, j_coords, k_coords], axis=1)

            elif init_mode == "random":
                total_cells = N ** 3
                if Q > total_cells:
                    raise ValueError("Q cannot exceed N^3.")

                flat_indices = np.random.choice(total_cells, size=Q, replace=False)
                k_coords = flat_indices % N
                j_coords = (flat_indices // N) % N
                i_coords = (flat_indices // (N * N))
                self.queens = np.stack([i_coords, j_coords, k_coords], axis=1)

            else:
                raise ValueError(f"Unknown init_mode: {init_mode}")
        else:
            positions = np.asarray(positions, dtype=int)
            if positions.shape[1] != 3:
                raise ValueError("positions must be of shape (Q, 3).")
            self.queens = positions
            self.Q = positions.shape[0]

        # Occupancy set
        self.occ_set = set()
        for (i, j, k) in self.queens:
            pos = (int(i), int(j), int(k))
            if pos in self.occ_set:
                raise ValueError("Two queens occupy the same (i,j,k) cell.")
            self.occ_set.add(pos)

        self._energy = None


    def copy(self):
        new_state = State3DQueens(self.N, Q=self.Q, positions=self.queens.copy())
        new_state._energy = self._energy
        return new_state


    def energy(self, recompute=False):
        if self._energy is None or recompute:
            self._energy = self._compute_energy()
        return self._energy

    def _compute_energy(self):
        Q = self.Q
        if Q < 2:
            return 0

        positions = self.queens
        # Note: i, j, k have shape (Q,)
        i = positions[:, 0]
        j = positions[:, 1]
        k = positions[:, 2]

        # Note: di, dj, dk have shape (Q, Q)
        di = np.abs(i[:, None] - i[None, :])
        dj = np.abs(j[:, None] - j[None, :])
        dk = np.abs(k[:, None] - k[None, :])

        # Note: same_ij, same_ik, same_jk have shape (Q, Q)
        same_ij = (i[:, None] == i[None, :]) & (j[:, None] == j[None, :])
        same_ik = (i[:, None] == i[None, :]) & (k[:, None] == k[None, :])
        same_jk = (j[:, None] == j[None, :]) & (k[:, None] == k[None, :])
        
        # Note: plane_k_diag, plane_j_diag, plane_i_diag have shape (Q, Q)
        plane_k_diag = (k[:, None] == k[None, :]) & (di == dj)
        plane_j_diag = (j[:, None] == j[None, :]) & (di == dk)
        plane_i_diag = (i[:, None] == i[None, :]) & (dj == dk)
        
        space_diag = (di == dj) & (dj == dk)

        attacked = (
            same_ij
            | same_ik
            | same_jk
            | plane_k_diag
            | plane_j_diag
            | plane_i_diag
            | space_diag
        )

        upper_triangle = np.triu(attacked, k=1)
        return int(upper_triangle.sum())

    def propose_move(self, q_idx, new_pos):
        i_old, j_old, k_old = self.queens[q_idx]
        i_new, j_new, k_new = new_pos

        old_pos = (int(i_old), int(j_old), int(k_old))
        new_pos_t = (int(i_new), int(j_new), int(k_new))

        self.occ_set.remove(old_pos)
        self.occ_set.add(new_pos_t)

        self.queens[q_idx] = [i_new, j_new, k_new]

        return old_pos

    def conflicts_for_queen(self, q_idx, pos=None):
        if pos is None:
            i, j, k = self.queens[q_idx]
        else:
            i, j, k = pos

        mask = np.ones(self.Q, dtype=bool)
        mask[q_idx] = False
        others = self.queens[mask]
        if others.shape[0] == 0:
            return 0

        i2 = others[:, 0]
        j2 = others[:, 1]
        k2 = others[:, 2]

        di = np.abs(i2 - i)
        dj = np.abs(j2 - j)
        dk = np.abs(k2 - k)

        same_ij = (i2 == i) & (j2 == j)

        same_ik = (i2 == i) & (k2 == k)
        same_jk = (j2 == j) & (k2 == k)

        plane_k_diag = (k2 == k) & (di == dj)
        plane_j_diag = (j2 == j) & (di == dk)
        plane_i_diag = (i2 == i) & (dj == dk)

        space_diag = (di == dj) & (dj == dk)

        attacked = (
            same_ij
            | same_ik
            | same_jk
            | plane_k_diag
            | plane_j_diag
            | plane_i_diag
            | space_diag
        )

        return int(attacked.sum())
