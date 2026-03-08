#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Launch sheplet-student with a bundled test project
# Prerequisites: run ./scripts/test_e2e.sh first to create the test-project/
# =============================================================================

# --- Step 0: Environment -----------------------------------------------------

export PATH="$HOME/.cargo/bin:$HOME/miniconda3/envs/ml-env/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:$PATH"
export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_DIR="$PROJECT_ROOT/test-project"
INSTRUCTOR="$PROJECT_ROOT/target/release/sheplet-instructor"
STUDENT="$PROJECT_ROOT/target/release/sheplet-student"

# --- Step 1: Validate prerequisites ------------------------------------------

if [ ! -f "$TEST_DIR/model/model.gguf" ]; then
    echo "Error: $TEST_DIR/model/model.gguf not found."
    echo "Run ./scripts/test_e2e.sh first to create the test project."
    exit 1
fi

# --- Step 2: Build sheplet-student -------------------------------------------

echo ""
echo "=== Building sheplet-student ==="
STEP_START=$SECONDS
cargo build --release -p sheplet-student --manifest-path "$PROJECT_ROOT/Cargo.toml"
TIME_BUILD=$(( SECONDS - STEP_START ))
echo "--- Build completed in ${TIME_BUILD}s ---"

# --- Step 3: Build sheplet-instructor if needed (for bundling) ---------------

if [ ! -f "$INSTRUCTOR" ]; then
    echo ""
    echo "=== Building sheplet-instructor ==="
    cargo build --release -p sheplet-instructor --manifest-path "$PROJECT_ROOT/Cargo.toml"
fi

# --- Step 4: Bundle the test project ----------------------------------------

BUNDLE_PATH="$PROJECT_ROOT/test-project.sheplet"

echo ""
echo "=== Bundling test project ==="
STEP_START=$SECONDS
BUNDLE_OUTPUT=$("$INSTRUCTOR" bundle --project "$TEST_DIR" --output "$BUNDLE_PATH" 2>&1)
echo "$BUNDLE_OUTPUT"
TIME_BUNDLE=$(( SECONDS - STEP_START ))
echo "--- Bundle completed in ${TIME_BUNDLE}s ---"

# Extract fingerprint from bundle output
FINGERPRINT=$(echo "$BUNDLE_OUTPUT" | grep 'Fingerprint:' | head -1 | awk '{print $NF}')

# --- Step 5: Set up student data directory -----------------------------------

STUDENT_DIR="$PROJECT_ROOT/test-student-data"
rm -rf "$STUDENT_DIR"

# --- Step 6: Launch sheplet-student ------------------------------------------

echo ""
echo "============================================"
echo "  sheplet-student is starting!"
echo "  URL: http://127.0.0.1:8420"
echo "  Load the bundle at: $BUNDLE_PATH"
if [ -n "$FINGERPRINT" ]; then
    echo "  Instructor fingerprint: $FINGERPRINT"
fi
echo "  Press Ctrl+C to stop"
echo "============================================"
echo ""

exec "$STUDENT" --dir "$STUDENT_DIR" --port 8420
