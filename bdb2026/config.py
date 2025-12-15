from __future__ import annotations

# Canonical ID columns
GAME_ID = "game_id"
PLAY_ID = "play_id"
FRAME_ID = "frame_id"

# Common role / side columns in the prepared group_input/group_output datasets
PLAYER_ROLE = "player_role"          # e.g., 'Passer', 'Targeted Receiver'
PLAYER_SIDE = "player_side"          # 'Offense' / 'Defense'
PLAYER_POS = "player_position"       # 'WR', etc.
PLAYER_NAME = "player_name"
NFL_ID = "nfl_id"

# Tracking columns (raw)
X = "x"
Y = "y"
S = "s"                              # speed (units depend on dataset; keep consistent)
A = "a"                              # acceleration
DIR = "dir"                          # movement direction (deg)
O = "o"                              # orientation (deg)

# Post-throw ball landing location (from competition engineered outputs)
BALL_LAND_X = "ball_land_x"
BALL_LAND_Y = "ball_land_y"


# Normalized ball landing location (consistent with x_norm/y_norm)
BALL_LAND_X_NORM = "ball_land_x_norm"
BALL_LAND_Y_NORM = "ball_land_y_norm"

# Output/label positions (after attaching df_output)
Y_TRUE_X = "y_true_x"
Y_TRUE_Y = "y_true_y"
Y_TRUE_X_NORM = "y_true_x_norm"
Y_TRUE_Y_NORM = "y_true_y_norm"

# Pass timing
PASS_RELEASE_FRAME = "pass_release_frame"   # if present
TIME_SINCE_THROW = "time_since_throw"       # engineered, seconds

# Normalized coordinates used by notebook
X_NORM = "x_norm"
Y_NORM = "y_norm"

# Engineered kinematics
DX = "dx"
DY = "dy"
SPEED = "speed"                      # derived speed magnitude from dx/dy

# Geometry to landing point
DIST_TO_LAND = "dist_to_land"
BEARING_TO_LAND = "bearing_to_land"  # radians
HEADING = "heading"                  # radians
HEADING_ALIGN_COS = "heading_align_cos"

# Target label for LSTM (your notebook uses converge_rate)
CONVERGE_RATE = "converge_rate"
