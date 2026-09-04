import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Triage tests exercise the real routing table and env-configurable target URLs.
# These settings keep pydantic-settings constructible without a live database.
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017/growthengine_test")
os.environ.setdefault("OPERATOR_ADMIN_KEY", "test-operator-key")
