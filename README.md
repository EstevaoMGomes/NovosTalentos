# NovosTalentos

by Estêvão Gomes ([@EstevaoMGomes](https://github.com/EstevaoMGomes))

This project is a Stellarator Coil Optimizer of alpha particles via differentiable JAX code and was developed as the research 
work for the New Talents in Physics Fellowship, awarded by the [Calouste Gulbenkian Foundation](https://gulbenkian.pt/en/).

The project was developed under the supervision of professor Rogério Jorge ([@rogeriojorge](https://github.com/rogeriojorge)).

## Repository Organization

## Abstract
In magnetic confinement fusion, one of the most promising approaches that allows steady state
operation with no disruptions is the stellarator. A stellarator consists of electromagnetic coils
that create a twisted magnetic field that needs to be optimized to confine a high-performing
plasma. Such optimization is performed over a large set of parameters, typically of the order of
several hundred or more. Furthermore, the target magnetic field is usually a fixed one, that has
been previously obtained using another optimization based on the ideal MHD equations. With
this work, we trace particles directly in the corresponding Biot-Savart magnetic fields stemming
from a set of coils and optimize them to yield a small fraction of lost particles outside of the
confinement region. Furthermore, to replace the need for hundreds of simulations per
optimization step, we make use of automatic differentiation by implementing the guiding-center
equations, magnetic field solver, and optimization routines in JAX. This allows us to streamline
optimization efforts, and create a specialized, but very fast, numerical tool, to optimize force-free
stellarator equilibria. As force-free equilibria are usually the first step in determining the
viability of a device, such optimizations will be able to guide future designs based on ideal
MHD equilibria.

## How to use the repository
After cloning the repository:
```
git clone https://github.com/EstevaoMGomes/NovosTalentos.git
```
The easiest way to run an example script is to create a conda environment:
```
conda create --name myenv --file spec-file.txt    maybe (python=3.12.2)
```
where "myenv" is the name you want your environment to have and "spec-file.txt" can be dependenciesGPU.txt or dependenciesCPU.txt, whether you want to run the scripts in CPU or GPU.
