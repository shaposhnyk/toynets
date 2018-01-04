# Toy Neural Network implementations

## Files
- testnet01.py - naive implementation of one hidden layer NN on ReLUs
- testnet02.py - naive implementation of two hidden layer NN on ReLUs
- trainer.py - trains testnet01 on linear input (SGD)
- trainersin.py - trains testnet01 on sinusoidal input (SGD)

# Perfs 
- TN1x10 + linear input - convergence 7k iterations
- TN1x10 + sinusoidal input - convergence 700k iterations 
- TN1 + sinusoidal fq input - convergence ???k iterations 

# TODOs

# Findings
- minibatches does not help on independent samples, which corresponds to the theory