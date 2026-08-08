"""Path resolution and stub imports for the load suite.

The suite lives inside SonicSightBackend/ and reuses the project's own
generated stubs and config constants, exactly as replay_client.py does, so
the harness cannot drift from the server's wire contract.
"""

import os
import sys

LOADTEST_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(LOADTEST_DIR)
SRC_DIR = os.path.join(BACKEND_ROOT, "src")
RESULTS_DIR = os.path.join(LOADTEST_DIR, "results")

# docs/ lives beside the three repositories at the workspace root.
WORKSPACE_ROOT = os.path.dirname(BACKEND_ROOT)
NFR_TARGETS_YAML = os.path.join(WORKSPACE_ROOT, "docs", "nfr", "nfr_targets.yaml")
BACKEND_PROTO = os.path.join(BACKEND_ROOT, "sonicsight.proto")
MOBILE_PROTO = os.path.join(
    WORKSPACE_ROOT, "SonicSightMobile", "app", "src", "main", "proto", "sonicsight.proto"
)

for _p in (BACKEND_ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def import_stubs():
    """Import the project's generated stubs and shared constants."""
    import sonicsight_pb2            # noqa: F401
    import sonicsight_pb2_grpc       # noqa: F401
    from config import AUD_RATE, IMG_SIZE, FRAME_RATE  # noqa: F401

    return sonicsight_pb2, sonicsight_pb2_grpc
