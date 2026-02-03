# FAQ
This section tries to answer some common questions about **``socca``**.
 
## Why **``socca``**?
As already mentioned in the [introduction](./index.md), the core design of **``socca``** is inspired by existing and more mature tools [``galfit``](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html), [`astrophot`](https://github.com/Autostronomy/AstroPhot), and [`pysersic`](https://github.com/pysersic/pysersic), which have clearly demonstrated the power and flexibility of the modular approach to image fitting.

Still, my day-to-day use of these libraries made me realize the lack of some (or a combination of) features that have been essential to analysis workflow. The most important among these are the native _JAX integration_, a direct _interface to nested sampling algorithms_ for enabling accurate Bayesian model selection, and a flexible framework for defining _complex prior probabilities_.

**``socca``** has been built to address all this, with the core idea of providing a library that is simple, modular, and extensible.

## Why "socca"?
I ([@lucadimascolo](https://github.com/lucadimascolo)) started developing this library while working at Observatoire de la Côte d'Azur in Nice, France. _Socca_ is a traditional dish from Nice made from chickpea flour, water, olive oil, and salt (source: [Wikipedia](https://en.wikipedia.org/wiki/Socca_(food)), as well as all the socca I personally ate at Chez Thérésa at the Marché aux Fleurs). I chose the name "_socca_" to acknowledge the roots of the project in Nice and the lively scientific environment that shaped its early development. In a way, though, the name also conveys the idea of simplicity in modelling astronomical data: minimal ingredients, no unnecessary complexity, and a result that is both robust and satisfying—much like socca itself.