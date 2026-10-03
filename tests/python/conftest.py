import os
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# torch_geometric triggers a torch.jit deprecation warning on import.
warnings.filterwarnings("ignore", category=FutureWarning)
