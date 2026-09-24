"""Output heads for diabetic retinopathy severity classification and ordinal regression.

Includes:
- SoftmaxCEHead: Categorical Cross-Entropy baseline
- SoftmaxQWKHead: Multi-task CE + continuous QWK
- CORALHead: Consistent Rank Logits ordinal regression
- CORNHead: Conditional Ordinal Regression Neural Network
- CumulativeLinkModelQWKHead: Cumulative Link Model with clog-log link and hybrid QWK loss
"""

from drdg.models.heads.clm import CumulativeLinkModelQWKHead
from drdg.models.heads.coral import CORALHead
from drdg.models.heads.corn import CORNHead
from drdg.models.heads.softmax import SoftmaxCEHead, SoftmaxQWKHead

__all__ = [
    "CumulativeLinkModelQWKHead",
    "CORALHead",
    "CORNHead",
    "SoftmaxCEHead",
    "SoftmaxQWKHead",
]
