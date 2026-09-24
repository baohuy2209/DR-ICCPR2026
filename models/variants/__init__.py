"""Paper architectural ablation variants (V0 through V3).

Includes:
- V0: Global Context baseline (ResNet-50 + Softmax CE)
- V1: Local MIL architecture (+ EfficientNet-B0 Dual Pooling)
- V2: Global Context enrichment (+ Non-Local + Quadrant Tokens)
- V3: Ordinal CLM (+ Cumulative Link Model with clog-log link and hybrid QWK loss)
"""

from drdg.models.variants.v0 import V0BaselineModel, build_v0_model
from drdg.models.variants.v1 import V1LocalMILModel, build_v1_model
from drdg.models.variants.v2 import V2GlobalContextModel, build_v2_model
from drdg.models.variants.v3 import V3OrdinalCLMModel, build_v3_model

__all__ = [
    "V0BaselineModel",
    "build_v0_model",
    "V1LocalMILModel",
    "build_v1_model",
    "V2GlobalContextModel",
    "build_v2_model",
    "V3OrdinalCLMModel",
    "build_v3_model",
]
