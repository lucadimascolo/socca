# Installation
```{warning}
<code>socca</code> has been built using `python=3.12`. Although the installation is not bound to this specific version and different releases have been used successfully in the past, it is recommended to use the same version to avoid any compatibility issues.
```

To install <code>socca</code>, you can either clone the repository and install it locally, or install it directly from the GitHub repository. In the first case, you can run the following commands:

```
git clone https://github.com/lucadimascolo/socca.git
cd socca
python -m pip install .
```

If you plan to modify the source code or contribute to the project, you may prefer an editable installation:

```
python -m pip install -e .
```

Alternatively, you can install <code>socca</code> directly from the repository without cloning it locally:

```
python -m pip install git+https://github.com/lucadimascolo/socca.git
```

This will download and install the latest version of <code>socca</code> as well as all the required dependencies. Once the installation is completed, you should be ready to use <code>socca</code> to crunch your data.

```{note}
All dependencies required by <code>socca</code> are standard Python packages and should not cause major conflicts. Nevertheless, it is recommended to install <code>socca</code> inside a virtual environment (`venv`, `conda`, etc.) to avoid potential issues with existing packages.
```