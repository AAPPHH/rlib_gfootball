"""Feature engineering for Google Research Football observations."""

import numpy as np

FEATURE_DIM = 93
OBS_DIM = 115


class FeatureEngineer:
    """Extracts a hand-crafted 93-dim feature vector from raw 115-dim observations.

    Features include ball state, relative positions, goal/keeper geometry,
    nearest teammates/opponents, formation metrics, zone indicators, and
    sticky action encodings.

    Attributes:
        GOAL: Position of the opponent goal.
        OWN_GOAL: Position of the own goal.
    """

    GOAL = np.array([1.0, 0.0], dtype=np.float32)
    OWN_GOAL = np.array([-1.0, 0.0], dtype=np.float32)

    @staticmethod
    def extract(obs: np.ndarray, active_idx: int = None) -> np.ndarray:
        """Extract engineered features from raw observations.

        Supports both single observations (1D) and batched observations (2D).
        Single observations are automatically squeezed back to 1D on return.

        Args:
            obs: Raw observation array, shape ``(115,)`` or ``(B, 115)``.
            active_idx: Index of the active player. If ``None``, detected
                automatically via ``argmax`` over the active-player slice.

        Returns:
            Feature array of shape ``(93,)`` or ``(B, 93)``.
        """
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs.reshape(1, -1)
        B = obs.shape[0]
        obs = (
            obs[:, :115]
            if obs.shape[1] >= 115
            else np.pad(obs, ((0, 0), (0, 115 - obs.shape[1])))
        )
        feat = np.zeros((B, FEATURE_DIM), dtype=np.float32)

        left_pos = obs[:, 0:22].reshape(B, 11, 2)
        left_dir = obs[:, 22:44].reshape(B, 11, 2)
        right_pos = obs[:, 44:66].reshape(B, 11, 2)
        right_dir = obs[:, 66:88].reshape(B, 11, 2)
        ball_pos = obs[:, 88:90]
        ball_z = obs[:, 90:91]
        ball_dir = obs[:, 91:94]
        ball_owned_team = np.argmax(obs[:, 94:97], axis=1) - 1
        game_mode = obs[:, 98:105]
        sticky = obs[:, 105:115]

        if active_idx is None:
            active_idx = np.argmax(obs[:, 97:108], axis=1)
        elif isinstance(active_idx, int):
            active_idx = np.full(B, active_idx)

        bi = np.arange(B)
        active_pos = left_pos[bi, active_idx]
        ball_speed = np.linalg.norm(ball_dir[:, :2], axis=1)

        # Ball state (features 0-6).
        feat[:, 0:2] = ball_pos
        feat[:, 2] = np.clip(ball_z[:, 0], 0, 1)
        feat[:, 3] = np.clip(ball_speed, 0, 2)
        feat[:, 4:6] = ball_dir[:, :2]
        feat[:, 6] = ball_owned_team / 2.0

        # Relative ball position (features 7-10).
        rel_ball = ball_pos - active_pos
        feat[:, 7:9] = rel_ball
        feat[:, 9] = np.clip(np.linalg.norm(rel_ball, axis=1), 0, 2)
        feat[:, 10] = np.arctan2(rel_ball[:, 1], rel_ball[:, 0]) / np.pi

        # Goal geometry (features 11-15).
        goal_vec = FeatureEngineer.GOAL - ball_pos
        dist_goal = np.linalg.norm(goal_vec, axis=1)
        feat[:, 11] = np.clip(dist_goal, 0, 2)
        feat[:, 12] = np.abs(np.arctan2(goal_vec[:, 1], goal_vec[:, 0])) / np.pi
        feat[:, 13] = np.clip(0.088 / (dist_goal + 0.01), 0, 1)
        feat[:, 14] = (dist_goal < 0.35).astype(np.float32)
        feat[:, 15] = np.clip(
            np.linalg.norm(FeatureEngineer.OWN_GOAL - ball_pos, axis=1), 0, 2
        )

        # Keeper geometry (features 16-17).
        right_x = right_pos[:, :, 0]
        keeper_idx = np.argmax(right_x, axis=1)
        keeper_pos = right_pos[bi, keeper_idx]
        keeper_dist = np.linalg.norm(ball_pos - keeper_pos, axis=1)
        feat[:, 16] = np.clip(keeper_dist, 0, 2)
        feat[:, 17] = (
            np.arctan2((ball_pos - keeper_pos)[:, 1], (ball_pos - keeper_pos)[:, 0])
            / np.pi
        )

        # Nearest 5 teammates (features 18-37).
        left_active = np.abs(left_pos[:, :, 0]) > 0.01
        tm_dist = np.linalg.norm(left_pos - active_pos[:, None, :], axis=2)
        tm_dist[bi, active_idx] = 999.0
        tm_dist = np.where(left_active, tm_dist, 999.0)
        tm_idx = np.argsort(tm_dist, axis=1)
        for i in range(5):
            idx = tm_idx[:, i]
            valid = tm_dist[bi, idx] < 100
            feat[:, 18 + i * 4 : 20 + i * 4] = np.where(
                valid[:, None], left_pos[bi, idx] - active_pos, 0
            )
            feat[:, 20 + i * 4 : 22 + i * 4] = np.where(
                valid[:, None], left_dir[bi, idx], 0
            )

        # Nearest 5 opponents (features 38-57).
        right_active = np.abs(right_pos[:, :, 0]) > 0.01
        op_dist = np.linalg.norm(right_pos - active_pos[:, None, :], axis=2)
        op_dist = np.where(right_active, op_dist, 999.0)
        op_idx = np.argsort(op_dist, axis=1)
        for i in range(5):
            idx = op_idx[:, i]
            valid = op_dist[bi, idx] < 100
            feat[:, 38 + i * 4 : 40 + i * 4] = np.where(
                valid[:, None], right_pos[bi, idx] - active_pos, 0
            )
            feat[:, 40 + i * 4 : 42 + i * 4] = np.where(
                valid[:, None], right_dir[bi, idx], 0
            )

        # Formation metrics (features 58-60).
        ball_x = ball_pos[:, 0]
        left_x = left_pos[:, :, 0]
        feat[:, 58] = np.sum((left_x > ball_x[:, None]) & left_active, axis=1) / 11.0
        feat[:, 59] = np.sum((right_x > ball_x[:, None]) & right_active, axis=1) / 11.0
        feat[:, 60] = np.clip((feat[:, 58] - feat[:, 59]) * 2, -1, 1)

        # Closest opponent distance (feature 63).
        opp_dist_active = np.where(
            right_active,
            np.linalg.norm(right_pos - active_pos[:, None, :], axis=2),
            10.0,
        )
        feat[:, 63] = np.clip(np.min(opp_dist_active, axis=1), 0, 1)

        # Offside line (features 67-68).
        sorted_rx = np.sort(right_x, axis=1)
        offside_line = np.maximum(ball_x, sorted_rx[:, 1])
        feat[:, 67] = offside_line
        feat[:, 68] = (
            (active_pos[:, 0] > offside_line) & (ball_owned_team == 0)
        ).astype(np.float32)

        # Zone indicators (features 69-73).
        feat[:, 69:72] = np.column_stack(
            [
                (ball_x > 0.33),
                ((ball_x >= -0.33) & (ball_x <= 0.33)),
                (ball_x < -0.33),
            ]
        )
        feat[:, 72:74] = np.column_stack(
            [(ball_pos[:, 1] > 0.2), (ball_pos[:, 1] < -0.2)]
        )

        # Sticky actions (features 74-84).
        feat[:, 74:76] = sticky[:, 8:10]
        sticky_dir = sticky[:, :8]
        sticky_active = np.any(sticky_dir > 0, axis=1)
        angle = np.argmax(sticky_dir, axis=1) * (2 * np.pi / 8)
        feat[:, 76:78] = np.column_stack(
            [
                np.where(sticky_active, np.cos(angle), 0),
                np.where(sticky_active, np.sin(angle), 0),
            ]
        )
        feat[:, 78:85] = game_mode

        # Reserved (features 85-92).
        feat[:, 85:93] = 0

        return feat[0] if squeeze else feat
