# idaes-mvo

This Python repository contains generalized optimization formulations (written in the Algebraic Modeling Language Pyomo) to solve process family design problems. The methodology itself has been combined into a Python package, for ease of use. This problem can be solved in a variety of ways - currently, we can solve this problem by using a discretization method or through embedded machine learning surrogates. 

## Problem Statement

Process systems is a broad term that captures a variety of system based on chemical and mechanical processes. Some are small and relatively simple, like air conditioning units, while others are exceptionally large and complex, such as a nuclear power plant. Traditionally, process systems engineers focus on optimizing the complex, non-linear equations that govern the physics of these systems to find ideal operating conditions and schedules and overall designs of the systems so as to minimize expenses while meeting customers and company requirements.

Manufacturing process systems is rarely considered in the context of optimization in process systems engineering. When designing a singular instance of a system, the context of the deployment and system itself are used to determine optimal decisions (e.g., operations, schedules, designs). Manufacturing schemes for mass-deployment are rarely considered, however they are key area for saving capital and deployment costs as it is not likely a process system will only be deployed once. 

For example, take a carbon capture system that is to be deployed at a carbon-emitting factory in New York. A team of engineers will have the decide each possible optimization variable for this particular setting. This is time consuming; additionally, manufacturing one-off designs is incredibly expensive when compared to standardized manfuacturing lines.
Consider now that this carbon capture system is also being deployed at sites in Los Angeles, Boston, and Seattle. Rather than viewing each deployment as a singular, isolated optimization problem we propose designing all carbon capture systems simultaneously in a way that leads to standardization at the manufacturing levels.

In process family design, we aim to develop an optimization-backed manufacturing scheme for deploying a "family" of process systems. We design elements of these systems commonly, desinging a platform of standaradized units from which elements of the system must select. We design other elements of the system uniquely, aiming to balance the trade-offs between standardization and customization.

_For a more comprehensive problem statement, please see in the papers directory the document called focapo_invited_2023_stinchfield_preliminiary.pdf_

## Optimization Techniques

We pose a general problem, coined "process family design", and have developed four subsequent optimization based techniques to solve it. This package and repository demonstrates all two of the four approaches, as detailed below.

### Approach 1: Discretization

Searching the continuous design space is impractical for complex, non-linear systems. Our first approach involved discretizing across the design space, simulating, and making optimal selections among these discrete design choices. In addition, the formulation shares the structure and properties of the P-median optimization formulation.

_See the discretized.py file in the example directory_

### Approach 2: Embedded Machine Learning Surrogates

Constraining our decisions to discrete designs limits our ability to fully customize the manufacturing scheme; additionally, we have no guarantees that the optimal design is among the discrete options we pre-selected. Our second approach involved training and embedding machine learning surrogates to replace the complex, non-linear system of equations that govern the physics of these models. This allowed us to search among a surrogated continuous design space.  

_See the surrogates.py file in the examples directory_

