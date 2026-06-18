from .cloud_adapters import get_cloud_adapter
from .local_adapters import get_local_adapter

__all__ = ["get_cloud_adapter", "get_local_adapter"]
