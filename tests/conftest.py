import sys
from pathlib import Path

# Add both project root and src to sys.path for pytest and script imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
