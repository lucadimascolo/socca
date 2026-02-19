# Installation
```{warning}
**``socca``** has been built using `python=3.12`. Although the installation is not bound to this specific version and different releases have been used successfully in the past, it is recommended to use the same version to avoid any compatibility issues.
```

To install **``socca``**, you can either clone the repository and install it locally, or install it directly from the GitHub repository. In the first case, you can run the following commands:

```
git clone https://github.com/lucadimascolo/socca.git
cd socca
python -m pip install .
```



If you plan to modify the source code or contribute to the project, you may prefer an editable installation:

```
python -m pip install -e .
```

Alternatively, you can install **``socca``** directly from the repository without cloning it locally:

```
python -m pip install git+https://github.com/lucadimascolo/socca.git
```

This will download and install the latest version of **``socca``** as well as all the required dependencies. Once the installation is completed, you should be ready to use **``socca``** to crunch your data.

```{note}
All dependencies required by **``socca``** are standard Python packages and should not cause major conflicts. Nevertheless, it is recommended to install **``socca``** inside a virtual environment (`venv`, `conda`, etc.) to avoid potential issues with existing packages.
```

## Intel-based chips

On Intel-based machines, a conflict between Intel OpenMP (`libiomp5`) — bundled with PyTorch via MKL — and LLVM OpenMP (`libomp`) — used by JAX/XLA — may cause an error at import time. The recommended fix is to install the CPU-only version of PyTorch, which does not ship with MKL and therefore _should_ avoid the conflict entirely.

The cleanest way to handle this automatically is to use [`uv`](https://github.com/astral-sh/uv) instead of `pip`. The **``socca``** package already includes the necessary configuration, so the following commands are sufficient:

```
uv pip install socca
```

or, if cloning the repository locally:

```
git clone https://github.com/lucadimascolo/socca.git
cd socca
uv pip install .
```

`uv` will automatically pull the CPU-only PyTorch wheel from the PyTorch index. If you prefer to stick with `pip`, you can manually install the CPU-only version of PyTorch before installing **``socca``**:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install socca
```