"""Tabular Q-Learner with epsilon-greedy exploration and optional Dyna."""

import random

import numpy as np


class TabularQLearner:
    """Model-free RL via a Q-table.

    Parameters:
        num_states:  Size of the discrete state space.
        num_actions: Number of available actions.
        alpha:       Learning rate.
        gamma:       Discount factor.
        rar:         Random-action rate (exploration probability).
        radr:        Random-action decay rate (``rar *= radr`` each step).
        dyna:        Number of Dyna replay updates per real update (0 = off).
    """

    def __init__(
        self,
        num_states: int = 100,
        num_actions: int = 3,
        alpha: float = 0.2,
        gamma: float = 0.9,
        rar: float = 0.5,
        radr: float = 0.99,
        dyna: int = 0,
    ):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.Q = np.zeros((num_states, num_actions))
        self.s: int = 0
        self.a: int = 0

        # Dyna model
        self.Tc = np.zeros((num_states, num_actions, num_states))
        self.R = np.zeros((num_states, num_actions))
        self._experiences: list[tuple[int, int]] = []
        self._experience_set: set[tuple[int, int]] = set()

    # ── public API ─────────────────────────────────────────────────────

    def querysetstate(self, s: int) -> int:
        """Set the state and return the greedy action (no learning)."""
        self.s = s
        self.a = int(np.argmax(self.Q[s]))
        return self.a

    def query(self, s_prime: int, reward: float) -> int:
        """Update Q-table and return the next action."""
        self._update_q(self.s, self.a, s_prime, reward)

        if self.dyna > 0:
            self._update_dyna_model(self.s, self.a, s_prime, reward)
            self._replay()

        # Epsilon-greedy action selection
        if random.random() < self.rar:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = int(np.argmax(self.Q[s_prime]))

        self.s = s_prime
        self.a = action
        self.rar *= self.radr
        return action

    # ── internals ──────────────────────────────────────────────────────

    def _update_q(self, s: int, a: int, s_prime: int, r: float) -> None:
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (
            r + self.gamma * np.max(self.Q[s_prime])
        )

    def _update_dyna_model(self, s: int, a: int, s_prime: int, r: float) -> None:
        self.Tc[s, a, s_prime] += 1
        self.R[s, a] = (1 - self.alpha) * self.R[s, a] + self.alpha * r
        exp = (s, a)
        if exp not in self._experience_set:
            self._experiences.append(exp)
            self._experience_set.add(exp)

    def _replay(self) -> None:
        if not self._experiences:
            return
        for _ in range(self.dyna):
            s, a = self._experiences[random.randint(0, len(self._experiences) - 1)]
            s_prime = int(np.argmax(self.Tc[s, a]))
            self._update_q(s, a, s_prime, self.R[s, a])
