# Priors

Since <code>socca</code> integrates Bayesian samplers for performing posterior exploration, it is necessary to define priors for the parameters of the models we aim to fit. In <code>socca</code>, priors are assigned directly to the individual parameters of each model component. Overall, there are three options: fixed values, probability distributions, and functional forms. 

## Fixed values
This is the simplest case and it is achieved by setting a given parameter equal to a float. The corresponding parameter will then be fixed to that value during the sampling process. For instance, assuming to have a component `comp` with a parameter `a` and that we would like to fix `a=0.00`, we can simply do:
```python
comp.a = 0.00
```

## Probability distributions
If we are interested in sampling a given parameter from a probability distribution, we can use 

<code>socca</code> integrates some common distributions in the form of simple wrappers around <code>scipy.stats</code> distributions.

## Functional constraints
Functional constraints are particularly useful when considering multi-term models, as they allow for the definition of complex relationships between the parameters of different components.