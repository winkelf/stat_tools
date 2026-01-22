# Error Propagation and Linear Fit (Python)

This script implements a **linear least-squares fit with error propagation**, based on material from a statistics course.  
It computes the fit parameters, their uncertainties, the propagated error band, and validates the result using a **Monte Carlo simulation**.

> **Note:** Variable names, comments, and outputs are currently written in Argentinean Spanish.

---

## Features

- Linear fit \( y = a_1 + a_2 x \) with constant uncertainty `sigma`
- Analytical calculation of:
  - Fit parameters (`a1`, `a2`)
  - Covariance matrix
  - Uncertainties of the parameters
- Error propagation to estimate the uncertainty on \( y(x) \)
- Visualization of:
  - Data with error bars
  - Best-fit line
  - Correct and incorrect (for comparison) error bands
- Monte Carlo simulation to validate error propagation
- Histogram of Monte Carlo results compared to a Gaussian distribution

---

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

Install dependencies with:

```bash
pip install numpy scipy matplotlib

