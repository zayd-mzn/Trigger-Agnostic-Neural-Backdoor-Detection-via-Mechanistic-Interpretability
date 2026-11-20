#!/usr/bin/env bash
# End-to-end pipeline for one domain: build zoo -> fingerprints -> analysis.
# Usage: scripts/run_pipeline.sh <text|image> <tiny|cpu|full>
# Models and fingerprints already on disk are skipped, so it is safe to re-run.
set -euo pipefail
DOMAIN=${1:?domain: text|image}
PROFILE=${2:-cpu}
cd "$(dirname "$0")/.."
python -m bdd.benchmark --domain "$DOMAIN" --profile "$PROFILE"
python -m bdd.fingerprint --domain "$DOMAIN"
python -m bdd.analyze --domain "$DOMAIN"
