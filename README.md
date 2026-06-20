# ARFF POLYNOMIAL REGRESSOR
This project generates a multi variate polynomial regressor.
* The target attribute must be numeric.
* The nominal attributes will be internally one-hot encoded.

It has 2 operational modes:
* "create_model"(default)
  * Calculate the regression
  * Plot the prediticions
  * Returns the polynomial
* "test_orders"
  * Tests several different orders to help to find a good balance between complexity and accuracy.

The predition is calculated this way:
* For each instance (data row):
  * Generate the model without this instance
  * Apply the model to this instance to get the predicted output

# How to install (Linux instructions)
* Clone the project and enter in the folder
```bash
git clone https://github.com/sergio-schmiegelow/arff_polynomial_regressor.git
cd arff_polynomial_regressor/
```

* Create a python virtual environment and activate it.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

* Install the dependencies
```bash
pip install -r requirements.txt 
```

 * Deactivate the virtual environment
```bash
deactivate
```

# How to run (Linux instructions)
* Activate the virtual environment (just once for all runnings)
```bash
source .venv/bin/activate
```

* Show the command line options
```bash
python3 arff_polynomial_regressor.py 
```

* Execute the order test (optional)
(using normalization in this example)
```bash
python3 arff_polynomial_regressor.py -m test_orders -f mydata.arff -t my_target_attribute_name -r
```
It wil generate a plot of error vs polynomial order

* Execute the regression with the desired order
(using order 3 and normalization in this example)
```bash
python3 arff_polynomial_regressor.py -f mydata.arff -t my_target_attribute_name -n 3 -r
```
It wil generate a plot with the preditions vs real values and the polynomial.

# Why use normalization
Some values get real big when elevated to powers. Ex: 2000000^6 = 6e37.
When adding those values with small values in the polynomyal they go beyond the variable bit depth and become meaningless.
The normalization transforms all ranges in the 0.0 to 1.0 range before processing.

