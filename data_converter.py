import time
import numpy as np

def check_campas_action(campas_action_and_shoot: np.ndarray) -> None:
    assert len(campas_action_and_shoot.shape) == 2
    assert campas_action_and_shoot.shape[1] == 4
    print("delta_heading min-max:", campas_action_and_shoot[:, 0].min(), campas_action_and_shoot[:, 0].max())
    print("delta_altitude min-max:", campas_action_and_shoot[:, 1].min(), campas_action_and_shoot[:, 1].max())
    print("delta_speed min-max:", campas_action_and_shoot[:, 2].min(), campas_action_and_shoot[:, 2].max())
    print("shoot min-max:", campas_action_and_shoot[:, 3].min(), campas_action_and_shoot[:, 3].max())

def convert_campas_action_to_discrete(campas_action_and_shoot: np.ndarray) -> np.ndarray:
    """Convert continuous compass actions to discrete action indices

    The recorded actions are continuous Box values that have been normalized:
    - For heading/altitude/speed: continuous_value = (original_value / max_rate) + 1
    - This maps: [-max_rate, 0, +max_rate] → [0, 1, 2] (but as continuous values)
    - We need to discretize these into bins {0, 1, 2}

    Args:
        campas_action_and_shoot: Array of shape (batch, 4) with normalized continuous values
                                 [delta_heading, delta_altitude, delta_speed, shoot]
                                 where first 3 dims are continuous around [0, 1, 2]
                                 and shoot is continuous around {0, 1}

    Returns:
        action_indices: Array of shape (batch,) with discrete indices in [0, 53]
    """
    import gymnasium.spaces as spaces
    from uhtk.spaces.xxx2D import MD2D
    assert len(campas_action_and_shoot.shape) == 2
    assert campas_action_and_shoot.shape[1] == 4

    # Create action space and converter (MultiDiscrete [3, 3, 3, 2])
    action_space = spaces.MultiDiscrete([3, 3, 3, 2])
    action_converter = MD2D(action_space)

    seq_size = campas_action_and_shoot.shape[0]
    action_indices = np.zeros(seq_size, dtype=np.int64)

    # check_campas_action(campas_action_and_shoot)
    # check_campas_action(np.round(campas_action_and_shoot).astype(np.int64))

    for i in range(seq_size):
        # Extract the 4D action: [delta_heading, delta_altitude, delta_speed, shoot]
        action_4d = campas_action_and_shoot[i]

        if (action_4d[0] < 1.0): action_4d[0] = 0.0
        if (action_4d[1] < 1.0): action_4d[1] = 0.0
        if (action_4d[2] < 1.0): action_4d[2] = 0.0
        if (action_4d[0] > 1.0): action_4d[0] = 2.0
        if (action_4d[1] > 1.0): action_4d[1] = 2.0
        if (action_4d[2] > 1.0): action_4d[2] = 2.0
        action_discrete = np.round(action_4d).astype(np.int64)

        # Ensure values are in valid range
        action_discrete[0] = np.clip(action_discrete[0], 0, 2)  # delta_heading
        action_discrete[1] = np.clip(action_discrete[1], 0, 2)  # delta_altitude
        action_discrete[2] = np.clip(action_discrete[2], 0, 2)  # delta_speed
        action_discrete[3] = np.clip(action_discrete[3], 0, 1)  # shoot (threshold at 0.5)

        # Convert to single discrete index
        action_indices[i] = action_converter.action_to_index(action_discrete)

    return action_indices


