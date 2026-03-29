"""
Alpha Schedule for ICGAR

This module provides various α(t) schedules for controlling
the trade-off between repair speed and invariance preservation.

α=0: strict invariance (only use tangent gradient)
α=1: no invariance (use full gradient)
"""

import numpy as np


class AlphaScheduler:
    """
    Base class for alpha schedules.

    Provides various schedules for computing α(t) at each iteration.
    """

    def __init__(self, schedule_type='exponential_decay', **params):
        """
        Initialize alpha scheduler.

        Args:
            schedule_type: Type of schedule ('strict', 'constant', 'linear_ramp',
                                     'exponential_decay', 'inverse_decay',
                                     'feedback', 'loss_based', 'cosine')
            **params: Schedule-specific parameters
        """
        self.schedule_type = schedule_type
        self.params = params
        self.history = []

    def __call__(self, t, failed_regions_count=None, failed_regions_prev=None,
                loss=None, loss_history=None):
        """
        Compute alpha at iteration t.

        Args:
            t: Current iteration
            failed_regions_count: Number of failed regions
            failed_regions_prev: Previous number of failed regions
            loss: Current loss value
            loss_history: List of previous loss values

        Returns:
            alpha_t: scalar in [0, 1]
        """
        alpha = self._compute_alpha(
            t, failed_regions_count, failed_regions_prev, loss, loss_history
        )
        alpha = np.clip(alpha, 0.0, 1.0)
        self.history.append(alpha)
        return alpha

    def _compute_alpha(self, t, failed_regions_count, failed_regions_prev,
                   loss, loss_history):
        """Compute alpha based on schedule type."""
        if self.schedule_type == 'strict':
            return self._strict(t)
        elif self.schedule_type == 'constant':
            return self._constant(t)
        elif self.schedule_type == 'linear_ramp':
            return self._linear_ramp(t)
        elif self.schedule_type == 'exponential_decay':
            return self._exponential_decay(t)
        elif self.schedule_type == 'inverse_decay':
            return self._inverse_decay(t)
        elif self.schedule_type == 'feedback':
            return self._feedback(t, failed_regions_count, failed_regions_prev)
        elif self.schedule_type == 'loss_based':
            return self._loss_based(t, loss, loss_history)
        elif self.schedule_type == 'cosine':
            return self._cosine(t)
        else:
            # Default to exponential decay
            return self._exponential_decay(t)

    def _strict(self, t):
        """Strict invariance: always return 0."""
        return 0.0

    def _constant(self, t):
        """Constant alpha: return specified value."""
        return self.params.get('alpha', 0.5)

    def _linear_ramp(self, t):
        """Linear ramp from 0 to max."""
        T_ramp = self.params.get('T_ramp', 100)
        alpha_max = self.params.get('alpha_max', 1.0)
        return min(alpha_max, t / T_ramp)

    def _exponential_decay(self, t):
        """Exponential decay from alpha_0 to 0."""
        tau = self.params.get('tau', 50)
        alpha_0 = self.params.get('alpha_0', 1.0)
        return alpha_0 * (1.0 - np.exp(-t / tau))

    def _inverse_decay(self, t):
        """Inverse decay from 1 to 0 like 1/t."""
        t_0 = self.params.get('t_0', 1)
        return 1.0 / (1.0 + t / t_0)

    def _feedback(self, t, failed_regions_count, failed_regions_prev):
        """Feedback-based: adapt based on repair progress."""
        alpha_0 = self.params.get('alpha_0', 0.5)
        beta = self.params.get('beta', 1.0)

        if failed_regions_prev is None or failed_regions_prev == 0:
            progress = 0.0
        else:
            progress = (failed_regions_prev - failed_regions_count) / failed_regions_prev

        # Alpha decreases when making progress, increases when stuck
        alpha_t = alpha_0 * (1.0 - beta * progress)
        return alpha_t

    def _loss_based(self, t, loss, loss_history):
        """Loss-based: increase alpha when loss plateaus."""
        if loss_history is None or len(loss_history) < 10:
            return self.params.get('alpha_0', 0.1)

        window = self.params.get('window', 10)
        alpha_0 = self.params.get('alpha_0', 0.1)
        alpha_max = self.params.get('alpha_max', 0.9)

        recent_losses = loss_history[-window:]
        loss_std = np.std(recent_losses)
        plateau_threshold = self.params.get('plateau_threshold', 0.01)

        if loss_std < plateau_threshold:
            # Loss plateaus: use higher alpha for more flexibility
            return alpha_max
        else:
            # Loss decreasing: use conservative alpha
            return alpha_0

    def _cosine(self, t):
        """Cosine schedule."""
        T = self.params.get('T', 100)
        alpha_min = self.params.get('alpha_min', 0.0)
        alpha_max = self.params.get('alpha_max', 1.0)
        return alpha_min + 0.5 * (alpha_max - alpha_min) * \
               (1.0 + np.cos(np.pi * t / T))


def compute_alpha(t, schedule_type='exponential_decay', **kwargs):
    """
    Convenience function for computing alpha at iteration t.

    Args:
        t: Current iteration
        schedule_type: Type of schedule
        **kwargs: Schedule parameters

    Returns:
        alpha_t: scalar in [0, 1]
    """
    scheduler = AlphaScheduler(schedule_type, **kwargs)
    return scheduler(t)


# Predefined schedules
STRICT_SCHEDULE = {'schedule_type': 'strict'}

CONSTANT_SCHEDULES = {
    'strict': {'schedule_type': 'strict'},
    'conservative': {'schedule_type': 'constant', 'alpha': 0.1},
    'moderate': {'schedule_type': 'constant', 'alpha': 0.5},
    'permissive': {'schedule_type': 'constant', 'alpha': 0.9},
}

RAMP_SCHEDULES = {
    'linear': {
        'schedule_type': 'linear_ramp',
        'T_ramp': 100,
        'alpha_max': 0.5
    },
    'linear_full': {
        'schedule_type': 'linear_ramp',
        'T_ramp': 200,
        'alpha_max': 1.0
    },
}

DECAY_SCHEDULES = {
    'exponential': {
        'schedule_type': 'exponential_decay',
        'tau': 50,
        'alpha_0': 1.0
    },
    'exponential_slow': {
        'schedule_type': 'exponential_decay',
        'tau': 100,
        'alpha_0': 0.8
    },
    'inverse': {
        'schedule_type': 'inverse_decay',
        't_0': 10
    },
}
