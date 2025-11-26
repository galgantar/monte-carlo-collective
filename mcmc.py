import os
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml

import math

import math


class State3DQueens:
    def __init__(self, N, Q=None, positions=None, init_mode="latin"):
        """
        init_mode:
            - "latin":  k = (i + j) mod N (deterministic Latin-square style)
            - "klarner": k = (3i + 5j) mod N if gcd(N, 210) == 1;
                         otherwise use the largest M < N s.t. gcd(M,210)==1
                         to build an MxM Klarner core and place the rest randomly.
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

                if init_mode == "latin":
                    # Latin square: full N x N
                    i_grid, j_grid = np.indices((N, N))
                    k_grid = (i_grid + j_grid) % N
                    self.queens = np.stack(
                        [i_grid.ravel(), j_grid.ravel(), k_grid.ravel()],
                        axis=1
                    )

                else:  # init_mode == "klarner"
                    g = math.gcd(N, 210)
                    if g == 1:
                        # Standard Klarner construction on full N x N
                        i_grid, j_grid = np.indices((N, N))
                        k_grid = (3 * i_grid + 5 * j_grid) % N
                        self.queens = np.stack(
                            [i_grid.ravel(), j_grid.ravel(), k_grid.ravel()],
                            axis=1
                        )
                    else:
                        # Find largest M < N with gcd(M,210) == 1
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

                        print(
                            f"[warning] Klarner: gcd(N={N},210)={g}≠1; "
                            f"using Klarner core with M={M} (gcd(M,210)=1) "
                            f"and placing remaining queens randomly."
                        )

                        positions = []

                        # 1) Klarner core on M x M, depth M (non-attacking among themselves)
                        for i in range(M):
                            for j in range(M):
                                k = (3 * i + 5 * j) % M
                                positions.append((i, j, k))

                        used = set(positions)
                        core_Q = len(positions)  # M^2
                        remaining = Q - core_Q

                        if remaining < 0:
                            raise ValueError(
                                f"Klarner core size M^2={core_Q} exceeds Q={Q}."
                            )

                        # 2) Place the remaining queens randomly in free 3D cells
                        if remaining > 0:
                            N3 = N ** 3
                            # To avoid building the full list in huge N,
                            # we just sample until we have enough distinct free cells.
                            while len(positions) < Q:
                                i = np.random.randint(0, N)
                                j = np.random.randint(0, N)
                                k = np.random.randint(0, N)
                                pos = (i, j, k)
                                if pos not in used:
                                    used.add(pos)
                                    positions.append(pos)

                        self.queens = np.array(positions, dtype=int)

            elif init_mode == "random":
                # Fully random distinct 3D positions
                total_cells = N ** 3
                if Q > total_cells:
                    raise ValueError(f"Q={Q} cannot exceed N^3={total_cells}.")

                flat_indices = np.random.choice(total_cells, size=Q, replace=False)
                k_coords = flat_indices % N
                j_coords = (flat_indices // N) % N
                i_coords = flat_indices // (N * N)
                self.queens = np.stack([i_coords, j_coords, k_coords], axis=1)

            else:
                raise ValueError(f"Unknown init_mode: {init_mode}")

        else:
            positions = np.asarray(positions, dtype=int)
            if positions.shape[1] != 3:
                raise ValueError("positions must be of shape (Q, 3).")
            self.queens = positions
            self.Q = positions.shape[0]

        # Build occ_set, energy cache etc. here as in your full class
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
