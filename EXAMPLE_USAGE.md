# Example usage in a notebook

```python
from bdb2026 import load_bdb_csvs
from bdb2026.preprocess import normalize_coordinates, add_postthrow_features
from bdb2026.sequences import build_sequences
from bdb2026.models import build_lstm, train_lstm
from bdb2026.metrics import compute_truespeed
from bdb2026.viz import animate_speed_comparison

data = load_bdb_csvs(
    input_path="group_input.csv",
    output_path="group_output.csv",
    supplementary_path="supplementary_data.csv",
)

# Normalize and engineer post-throw features (example)
df_out = normalize_coordinates(data.df_output, offense_left_to_right=True)
df_out = add_postthrow_features(df_out)

# Build sequences (you choose feature_cols and target_col to match your notebook)
feature_cols = ["x_norm","y_norm","dx","dy","speed","dist_to_land","heading_align_cos","time_since_throw"]
X, y, keys = build_sequences(df_out, feature_cols, target_col="converge_rate", max_len=25, id_cols=("game_id","play_id"))

model = build_lstm(n_features=X.shape[-1], hidden_units=64, dropout=0.2, mask_value=0.0)
result = train_lstm(model, X, y, epochs=20, batch_size=64, verbose=1)

# Predict + compute TrueSpeed per play
y_hat = result.model.predict(X)
df_pred = (
    pd.DataFrame(keys, columns=["game_id","play_id"])
      .assign(idx=np.arange(len(keys)))
      .merge(
          pd.DataFrame({
              "idx": np.repeat(np.arange(len(keys)), X.shape[1]),
              "t": np.tile(np.arange(X.shape[1]), len(keys)),
              "actual": y.reshape(-1),
              "pred": y_hat.reshape(-1),
          }),
          on="idx",
      )
)
ts = compute_truespeed(df_pred, actual_col="actual", pred_col="pred", id_cols=("game_id","play_id"), agg="mean")
ts.head()
```
