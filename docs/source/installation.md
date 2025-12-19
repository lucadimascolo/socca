# Installation

To install <code>socca</code>, you can either clone the repository and install it locally or install it directly from the repository. In the first case, you can run the following commands:

```
git clone https://github.com/lucadimascolo/socca.git
cd socca
python -m pip install .
```
If you want to 

Alternatively, you can install <code>socca</code> directly from the repository:

```
python -m pip install git+https://github.com/lucadimascolo/socca.git
```

This will download and install the latest version of <code>socca</code> as well as all the required dependencies. Once the installation is completed, you should be ready to get <code>socca</code> to crunch your data.

All the dependencies required by <code>socca</code> are standard packages and should not case any major conflicts. Still, it is recommended to install <code>socca</code> within a virtual environment to avoid any potential issues.


```{warning}
<code>socca</code> was built using python=3.12. Although the installation is not bound to this specific version and different releases have been known to work fine, it is recommended to use the same version to avoid any compatibility issues.
```