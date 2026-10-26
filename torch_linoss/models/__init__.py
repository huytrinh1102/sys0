from torch_linoss.models.linoss import LinOSS, LinOSSLayer
from torch_linoss.models.linoss_cuda import LinOSSCuda, LinOSSCudaLayer

MODELS = {"ops": LinOSS, "cuda": LinOSSCuda}

__all__ = ["MODELS", "LinOSS", "LinOSSCuda", "LinOSSCudaLayer", "LinOSSLayer"]
