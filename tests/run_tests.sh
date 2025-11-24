#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$ROOT_DIR/tests/results"
mkdir -p "$RESULTS_DIR"

declare -a files=(
  "test_rl_trading.py"
  "test_rl_hedging.py"
  "test_rl_allocation.py"
  "test_pricing_derivatives.py"
  "test_hedging_derivatives.py"
  "test_signals_stock_prediction.py"
  "test_signals_btc_pca.py"
  "test_signals_eigen_portfolio.py"
  "test_signals_yc_build.py"
  "test_signals_yc_predict.py"
  "test_strategies_btc_ma.py"
  "test_strategies_crypto_allocation.py"
  "test_strategies_rl_sp500.py"
  "test_strategies_nlp_sentiment.py"
)

cd "$ROOT_DIR"

for file in "${files[@]}"; do
  name="${file%.*}"
  out="$RESULTS_DIR/${name}.txt"
  echo "Running $file ..."
  pytest "tests/$file" -q >"$out" || true
done

echo "Reports written to $RESULTS_DIR"
