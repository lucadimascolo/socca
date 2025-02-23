# Models

As shown in the [Quickstart](./quickstart.md) tutorial, the `socca` library is built around the concept of model components. These are classes that define the functional form of the model and the priors for its parameters. In this section, you can find detailed description of the available model components and their parameters.

If you have installed <code>socca</code> from source, it might be good to take a look at the available models as follows:
```{code-block} python
>>> import socca
>>> socca.models.zoo()
```

To take a look at the parameters of a specific model, you can then use the instance method `print_params()`.

```{toctree}
:caption: Read more
./models/zoo.md
./models/link.md
```