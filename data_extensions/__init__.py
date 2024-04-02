from torch.utils.cpp_extension import load
load_utils = load(name="load_utils", sources=["data_extensions/load_data.cpp"], verbose=True)

__all__ = ["load_utils"]