# Big Data Bowl 2026: Air Play IQ - Post-Throw Convergence Analysis

## 🏈 Project Overview

This project introduces **Air Play IQ**, a novel metric system that evaluates wide receiver performance during the critical post-throw phase of a passing play. While traditional metrics focus on pre-snap alignment or route running, our approach captures what happens *after* the ball leaves the quarterback's hand—specifically, how efficiently a receiver **converges** toward the ball's landing location.

### The Key Insight

> **"Getting open is only half the battle. Tracking and attacking the ball is what separates elite receivers."**

Our LSTM-based model learns the expected convergence behavior from successful plays (completions), then measures each receiver's deviation from this baseline to create actionable IQ scores.

---

## 📊 Metrics Produced

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **ConvergenceIQ_WR** | Post-throw ball tracking efficiency | `+1.0` = Converged 1 std dev faster than expected |
| **RouteExecIQ** | Pre-throw route execution quality (supporting metric) | `+1.0` = Route was 1 std dev cleaner than cluster average |
| **AirPlayIQ_WR** | Combined overall WR performance | Weighted combination of both components |

### Score Interpretation Guide

| Score Range | Label | What It Means |
|-------------|-------|---------------|
| > +1.5 | Elite | Top ~7% of plays |
| +0.5 to +1.5 | Above Average | Better than typical |
| -0.5 to +0.5 | Average | Expected performance |
| -1.5 to -0.5 | Below Average | Room for improvement |
| < -1.5 | Poor | Bottom ~7% of plays |

---

## 🔬 Methodology: Post-Throw Convergence Analysis

### What is "Convergence Rate"?

Convergence rate measures how quickly a receiver is closing the distance to the ball's landing location at each frame after the throw:

$$\text{ConvergeRate}[t] = \text{DistToLand}[t] - \text{DistToLand}[t+1]$$

- **Positive**: Receiver is getting closer to the catch point (good!)
- **Zero**: No net progress toward the ball
- **Negative**: Receiver is moving away from the ball (bad!)

### The LSTM Model

We train a sequence-to-sequence LSTM model to predict **expected convergence rate** at each frame based on:

#### Input Features (28 total)

**Geometry Features (13):**
| Feature | Description |
|---------|-------------|
| `x_norm`, `y_norm` | Normalized field position |
| `dx`, `dy` | Frame-to-frame displacement |
| `speed` | Velocity magnitude (yards/sec) |
| `dist_to_land` | Distance to ball landing location |
| `bearing_to_land` | Angle from player to ball (radians) |
| `heading` | Direction of movement (radians) |
| `heading_align_cos` | Alignment between movement and target (-1 to 1) |
| `time_since_throw` | Elapsed time since ball release |
| `initial_separation` | Distance at moment of throw (context) |
| `ball_land_x_norm`, `ball_land_y_norm` | Ball landing coordinates |

**Route Embedding Features (15):**

Pre-computed route characteristics from Phase 1 that provide context about the type of route run before the throw.

#### Model Architecture

```
Input (25 frames × 28 features)
    ↓
Masking Layer (handles variable-length sequences)
    ↓
LSTM(64 units, return_sequences=True)
    ↓
Dropout(0.3)
    ↓
LSTM(32 units, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
TimeDistributed Dense(16, relu)
    ↓
TimeDistributed Dense(1, linear)
    ↓
Output: Predicted convergence rate per frame
```

### Training Strategy: Learning from Success

**Critical Decision**: We train the model **only on completed passes**. This teaches the LSTM what "good" convergence looks like. When we then apply the model to all plays, deviations from this learned baseline reveal:

- **Positive residuals**: Receiver tracked the ball *better* than successful plays typically do
- **Negative residuals**: Receiver tracked the ball *worse* than successful plays typically do

---

## 📁 Project Structure

```
Big_Data_Bowl/
├── BDB_2026.ipynb                    # Main notebook: Data exploration, Phase 1 & 2
├── BDB_2026_Phase3_Metrics.ipynb     # Phase 3: Metric calculation & visualization
├── BDB_2026_insights.ipynb           # Additional analysis and insights
│
├── src/
│   ├── phase2_convergence.py         # LSTM model building and training functions
│   ├── data_processing.py            # Data loading and preprocessing
│   ├── feature_engineering.py        # Feature calculation utilities
│   └── phase3_metrics.py             # Metric computation functions
│
├── run_pipeline.py                   # End-to-end pipeline execution
│
├── wr_routes_embeddings.csv          # Phase 1 output: Route features + clusters
├── postthrow_predictions_Completed_Pass.csv  # Phase 2 output: Frame-level predictions
├── convergence_lstm_model_Completed_Pass.h5  # Trained LSTM model
│
├── air_play_iq_metrics_final.csv     # Final player-level metrics
├── air_play_iq_metrics_final_filtered.csv  # Filtered for volume (qualified players)
│
└── supplementary_data.csv            # Original competition supplementary data
```

---

## 🔄 Analysis Pipeline

### Phase 1: Route Execution (Supporting Context)

**Purpose**: Provide route-level context for the post-throw model.

1. Filter to targeted receivers only (`player_role == 'Targeted Receiver'`)
2. Normalize coordinates (offense always moves left → right)
3. Engineer 15 route features:
   - Distance, depth, width
   - Speed statistics (max, avg, std)
   - Acceleration statistics
   - Direction changes
   - Route efficiency
4. K-Means clustering (12 clusters) to group similar route shapes
5. Calculate **RouteExecIQ** based on deviation from cluster centroid

**Output**: `wr_routes_embeddings.csv`

### Phase 2: Post-Throw Convergence Modeling (Core Analysis)

**Purpose**: Model expected ball-tracking behavior and identify exceptional performance.

1. Extract post-throw tracking data for targeted receivers
2. Calculate geometry features (distance, bearing, heading alignment)
3. Merge route embeddings as contextual features
4. **Filter training data to completed passes only**
5. Build padded sequences (max 25 frames)
6. Train LSTM to predict convergence rate
7. Generate predictions for all plays
8. Calculate residuals (actual - predicted)

**Output**: `postthrow_predictions_Completed_Pass.csv`, `convergence_lstm_model_Completed_Pass.h5`

### Phase 3: Metric Creation (Final Scores)

**Purpose**: Transform raw residuals into interpretable IQ scores.

1. **Frame-Level IQ**: Standardize residuals within context groups (route cluster + air yards bin)
2. **Play-Level Aggregation**: Weighted average emphasizing later frames (when ball arrives)
3. **Combined Score**: 
   ```
   AirPlayIQ_WR = f(RouteExecIQ, ConvergenceIQ_WR)
   ```
   Using adaptive weighting where high separation reduces route execution weight

**Output**: `air_play_iq_metrics_final.csv`

---

## 📈 Key Results

### Model Performance

| Metric | Value |
|--------|-------|
| Validation MSE | ~0.02 |
| Validation MAE | ~0.10 yards/frame |
| R² Score | ~0.65 |

### Metric Validation

**ConvergenceIQ correlates with outcomes:**

| Pass Result | Mean ConvergenceIQ | Interpretation |
|-------------|-------------------|----------------|
| Complete (C) | +0.15 | Successful plays show better tracking |
| Incomplete (I) | -0.08 | Failed plays show worse tracking |
| Interception (IN) | -0.12 | Worst tracking on picks |

### Sample Player Rankings

Top performers by AirPlayIQ_WR (qualified by volume):

| Rank | Player | Routes | AirPlayIQ | Catch Rate |
|------|--------|--------|-----------|------------|
| 1 | [Elite WR] | 150+ | +0.85 | 72% |
| 2 | [Elite WR] | 140+ | +0.78 | 68% |
| ... | ... | ... | ... | ... |

---

## 🎯 Use Cases for Teams

### 1. Player Evaluation
- Identify receivers who excel at ball tracking vs. route running
- Find specialists (great on deep balls vs. short routes)
- Track development over time

### 2. Play Design
- Call routes that match receiver strengths
- Use depth ranges where receiver tracks ball best
- Avoid mismatches

### 3. Opponent Scouting
- Predict likely outcomes based on historical IQ scores
- Identify tendencies in opponent WR performance

### 4. QB-WR Chemistry
- Compare metrics with different QBs
- Identify timing/anticipation issues

---

## 🛠️ How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Quick Start

```python
# Run full pipeline
python run_pipeline.py

# Or run notebooks in order:
# 1. BDB_2026.ipynb (Phase 1 & 2)
# 2. BDB_2026_Phase3_Metrics.ipynb (Phase 3)
```

### Using Pre-trained Model

```python
import tensorflow as tf
import pandas as pd

# Load model
model = tf.keras.models.load_model('convergence_lstm_model_Completed_Pass.h5')

# Load predictions
predictions = pd.read_csv('postthrow_predictions_Completed_Pass.csv')

# Load final metrics
metrics = pd.read_csv('air_play_iq_metrics_final.csv')
```

---

## 📋 Data Requirements

**Input Data** (from NFL Big Data Bowl competition):
- `group_input.csv`: Pre-throw tracking data
- `group_output.csv`: Post-throw tracking data  
- `supplementary_data.csv`: Play-level metadata (pass result, route type, yards gained)

**Key Columns Used**:
- `game_id`, `play_id`: Play identifiers
- `nfl_id`, `player_name`: Player identifiers
- `player_role`: Must include "Targeted Receiver"
- `x`, `y`, `s`, `a`, `dir`, `o`: Tracking coordinates
- `ball_land_x`, `ball_land_y`: Ball landing location
- `pass_result`: C (Complete), I (Incomplete), IN (Interception)

---

## 🔮 Future Work

1. **Steam Data Integration**: Scrape Steam for additional game metadata to improve recommendations
2. **Defender Analysis**: Extend model to evaluate closest defender behavior
3. **QB Evaluation**: Model expected throw location and timing
4. **Real-time Application**: Deploy model for live game analysis

---

## 👥 Contributors

Big Data Bowl 2026 Submission Team

---

## 📜 License

This project is for the NFL Big Data Bowl 2026 competition.
