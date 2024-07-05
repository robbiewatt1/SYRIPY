![plot](./example_image.png)
# SYRIPY (Synchrotron radiation in Python) 
SYRIPY is a differentiable and GPU accelerated python package for performing 
statistical inference on synchrotron radiation based diagnostics. SYRIPY 
includes three modules: 
- Particle tracking
- Liénard–Wiechert solver
- Fourier optics propagation

allowing you to fully simulate the generation and detection of synchrotron
radiation (from the initial beam parameters to the expected intensity 
profile on a detector). 
SYRIPY is based on the PyTorch library. 
This allows the automatic calculation of gradients with respect to input beam 
parameters, which is useful for many statistical inference methods. 
An overview of the package and benchmarks can be found in the following <https://journals.iucr.org/s/issues/2024/02/00/tv5053/index.html>.
## Installing SYRIPY
SYRIPY can be installed using pip
```bash
git clone https://github.com/robbiewatt1/SYRIPY
pip install ./SYRIPY
```

## SYRIPY Examples
Some example scripts / notebooks demonstrating how to use SYRIPY can be found 
in the examples folder.


## Citation
```
@article{watt2024differentiable,
  title={A differentiable simulation package for performing inference of synchrotron-radiation-based diagnostics},
  journal={Journal of Synchrotron Radiation},
  volume={31},
  number={2},
  year={2024},
  publisher={International Union of Crystallography}
}
```
