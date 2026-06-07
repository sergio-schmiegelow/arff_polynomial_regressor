#TODO - Option to split in several regressions for each nominal value combination
#TODO - Export the model to a file and load it later for predictions

import argparse
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------
def sanity_check(args):
    original_df, meta = load_arff_to_dataframe(args.arff_file)
    #Check that the target column exists
    if args.target_attribute not in meta.names():
        return f"Target attribute '{args.target_attribute}' not found in ARFF file. Available attributes: {meta.names()}"
    #Check for zeros when using having negative powers
    if args.max_negative_order > 0:
        numeric_df, _, _ = separate_numeric_nominal_target(original_df, meta, args.target_attribute)
        if (numeric_df == 0).any().any():
            return "Negative powers are specified but there are zero values in the numeric features (produces division by zero). Please remove or handle zero values before using negative powers."
        #Check for normalization when using negative powers
        if args.normalize:
            return "Negative powers can't be used with normalization (produces division by zero). Please disable normalization or negative powers."
#-------------------------------------------------------------------------
def load_arff_to_dataframe(file_path):
    '''Loads ARFF file and returns a DataFrame and metadata.
    inputs:
        file_path: path to the ARFF file
    outputs:        df: DataFrame containing the data
        meta: metadata from the ARFF file
    '''
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    df = df.map(lambda x: x.decode() if isinstance(x, bytes) else x)
    return df, meta
#-------------------------------------------------------------------------
def separate_numeric_nominal_target(df, meta, target_col):
    '''Separates numeric and nominal columns, and the target column.
    inputs:
        df: DataFrame containing the data
        meta: metadata from the ARFF file
        target_col: name of the target column
    outputs:
        numeric_df: DataFrame containing numeric columns
        nominal_df: DataFrame containing nominal columns
        target_df: DataFrame containing the target column
    '''
    nominal_cols = [col[0] for col in zip(meta.names(), meta.types()) if col[1] == 'nominal' and col[0] != target_col]
    nominal_df = df[nominal_cols]
    numeric_df = df.drop(columns=nominal_cols + [target_col])
    target_df = df[target_col]
    return numeric_df, nominal_df, target_df
#-------------------------------------------------------------------------
def normalize_numeric_features(numeric_df):
    '''Normalizes numeric features using min-max scaling.
    inputs:
        numeric_df: DataFrame containing numeric columns
    outputs:
        normalized_df: DataFrame containing normalized numeric columns
        norm_params: dictionary containing normalization parameters (min and max for each column)
    '''
    normalized_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
    norm_params = {col: {'min': numeric_df[col].min(), 'max': numeric_df[col].max()} for col in numeric_df.columns}
    return normalized_df, norm_params
#-------------------------------------------------------------------------
def create_polynomial_features(numeric_df, max_positive_order, max_negative_order):
    '''Creates polynomial features up to the specified order.
    inputs:
        numeric_df: DataFrame containing numeric columns
        max_positive_order: maximum positive order of polynomial features
        max_negative_order: maximum negative order of polynomial features
    outputs:
        numeric_df: DataFrame containing polynomial features
    '''
    new_numeric_df = pd.DataFrame()
    numeric_cols = numeric_df.columns
    for p in range(1, max_positive_order + 1):
        for col in numeric_cols:
            if p == 1:
                new_numeric_df[col] = numeric_df[col]
            else:
                new_numeric_df[f'{col}^{p}'] = numeric_df[col] ** p
    for n in range(1, max_negative_order + 1):
        for col in numeric_cols:
            new_numeric_df[f'{col}^(-{n})'] = numeric_df[col] ** (-n)
    return new_numeric_df
#-------------------------------------------------------------------------
def one_hot_encode_nominal(nominal_df):
    '''Performs one-hot encoding on nominal columns.
    inputs:
        nominal_df: DataFrame containing nominal columns
    outputs:
        one_hot_df: DataFrame containing one-hot encoded columns
    '''
    return pd.get_dummies(nominal_df, dtype=int, drop_first=False)
#-------------------------------------------------------------------------
def preprocess_data(original_df, meta, target_col, max_positive_order, max_negative_order, normalize):
    '''Preprocesses the data by separating numeric and nominal features, creating polynomial features, and one-hot encoding nominal features.
    inputs:
        original_df: DataFrame containing the data
        meta: metadata from the ARFF file
        target_col: name of the target column
        max_positive_order: maximum positive order of polynomial features
        max_negative_order: maximum negative order of polynomial features
    outputs:
        processed_df: DataFrame containing the preprocessed data
    '''
    numeric_df, nominal_df, target_df = separate_numeric_nominal_target(original_df, meta, target_col)
    numeric_df, norm_data = normalize_numeric_features(numeric_df) if normalize else (numeric_df, None)
    print(f'DEBUG - min/max normalized data:\n{numeric_df.describe().loc[["min", "max"]]}')
    numeric_df = create_polynomial_features(numeric_df, max_positive_order, max_negative_order)
    one_hot_encoded = one_hot_encode_nominal(nominal_df)
    processed_df = pd.concat([numeric_df, one_hot_encoded, target_df], axis=1)
    print(f'DEBUG - min/max all data:\n{processed_df.describe().loc[["min", "max"]]}')
    return processed_df, norm_data
#-------------------------------------------------------------------------
def train_and_evaluate_full_df(df, target_col):
    '''Trains a linear regression model on the entire DataFrame and evaluates it.
    inputs:
        df: DataFrame containing the preprocessed data
        target_col: name of the target column
    outputs:        model: trained LinearRegression model
        mse: mean squared error of the model on the training data
        r2: R^2 score of the model on the training data
    '''
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    rms = root_mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f"Root Mean Squared Error: {rms}")
    print(f"R^2 Score: {r2}")
    return model, rms, r2 
#-------------------------------------------------------------------------
def train_and_evaluate_single_line(df, target_col, test_index):
    '''Trains a linear regression model on all rows except the test row and evaluates it on the test row.
    inputs:
        df: DataFrame containing the preprocessed data
        target_col: name of the target column
        test_index: index of the row to use for testing
    outputs:
        actual: actual value of the target column for the test row
        pred: predicted value of the target column for the test row
        error: absolute error between actual and predicted values
    '''
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    train_df = df.drop(index=test_index)
    test_df = df.iloc[test_index].to_frame().T
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]
    actual = y_test.values[0]
    error = np.abs(actual - pred)
    return actual, pred, error
#-------------------------------------------------------------------------
def test_each_row(processed_df, target_col):
    '''Tests the model on each row by training on all other rows and evaluating on the test row.
    inputs:
        processed_df: DataFrame containing the preprocessed data
        target_col: name of the target column
    outputs:
        actuals: list of actual values for each test row
        predictions: list of predicted values for each test row
        errors: list of absolute errors for each test row
    '''
    num_rows = len(processed_df)
    actuals, predictions, errors = [], [], []
    for rowIdx in range(num_rows):
        actual, pred, error = train_and_evaluate_single_line(processed_df, target_col, rowIdx)
        actuals.append(actual)
        predictions.append(pred)
        errors.append(error)
        print(f"Row {rowIdx}: Actual={actual}, Predicted={pred}, Error={error}")
    return actuals, predictions, errors
#-------------------------------------------------------------------------
def generate_regression_equation(model, feature_names, target_col, norm_data):
    '''Generates a human-readable regression equation from the model coefficients.
    inputs:
        model: trained LinearRegression model
        feature_names: list of feature names
        target_col: name of the target column
        norm_data: dictionary containing normalization parameters (min and max for each column) if normalization was applied
    outputs:
        equation: string representing the regression equation
    '''
    coefficients = model.coef_
    intercept = model.intercept_
    equation = f"{target_col} = {intercept}"
    for coef, name in zip(coefficients, feature_names):
        if norm_data and name in norm_data:
            min_val = norm_data[name]['min']
            max_val = norm_data[name]['max']
            equation += f" + ({coef} * "
            if min_val != 0:
                equation += f"(({name} - {min_val}) / {max_val - min_val}))"
            else:
                equation += f"({name} / {max_val - min_val}))"
        else:
            equation += f" + ({coef} * {name})"
    return equation
#-------------------------------------------------------------------------
def test_polynomial_orders(df, meta, target_col, max_order, max_negative_order, normalize):
    '''Tests different polynomial orders and plots the mean squared error for each order.
    inputs:
        df: DataFrame containing the preprocessed data
        meta: metadata for the ARFF file
        target_col: name of the target column
        max_order: maximum polynomial order to test
        max_negative_order: maximum negative polynomial order to test
        normalize: whether to normalize numeric features
    outputs:
        None (plots the results)
    '''
    positive_orders = list(range(0, max_order + 1))
    negative_orders = list(range(0, max_negative_order + 1))
    mse_grid = np.zeros((len(negative_orders), len(positive_orders)))

    for i, negative_order in enumerate(negative_orders):
        for j, positive_order in enumerate(positive_orders):
            if negative_order == 0:
                print(f"\nTesting positive order={positive_order}, no negative powers")
            else:
                print(f"\nTesting positive order={positive_order}, negative order=-{negative_order}")
            processed_df, norm_data = preprocess_data(df, meta, target_col, positive_order, negative_order, normalize)
            actuals, predictions, errors = test_each_row(processed_df, target_col)
            rms = root_mean_squared_error(actuals, predictions)
            print(f"Root Mean Squared Error for pos={positive_order}, neg=-{negative_order}: {rms}")
            mse_grid[i, j] = rms

    print(f'DEBUG - MSE Grid:\n{mse_grid}')
    if max_negative_order == 0:
        print("\nNegative powers were not tested, only positive powers.")



    #plot a 2D graph with x-axis as the positive order and y-axis as the negative order, and the color representing the MSE
    fig, ax = plt.subplots()
    im = ax.imshow(mse_grid, origin='lower', aspect='auto', cmap='viridis',
                   extent=[positive_orders[0] - 0.5, positive_orders[-1] + 0.5,
                           negative_orders[0] - 0.5, negative_orders[-1] + 0.5])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Root Mean Squared Error')
    ax.set_xlabel('Positive Order')
    ax.set_ylabel('Negative Order')
    ax.set_title('MSE vs Positive/Negative Polynomial Order')
    ax.set_xticks(positive_orders)
    ax.set_yticks(negative_orders)
    ax.set_yticklabels(["0" if n == 0 else f"-{n}" for n in negative_orders])

    for i, negative_order in enumerate(negative_orders):
        for j, positive_order in enumerate(positive_orders):
            ax.text(j, i, f"{mse_grid[i, j]:.3f}", ha='center', va='center', color='white', fontsize=8)

    plt.show()
#-------------------------------------------------------------------------
def parse_arguments():
    '''Parses command-line arguments for the script.
    inputs:
        None (reads from command line)
    outputs:        
        args: parsed arguments
    '''
    parser = argparse.ArgumentParser(description='Multipolynomial regression on ARFF data.')
    parser.add_argument('-f', '--arff_file', required=True, help='Input ARFF file')
    parser.add_argument('-t', '--target_attribute', required=True, help='Attribute to be predicted')
    parser.add_argument('-m', '--mode', choices=['create_model', 'test_orders'], default='create_model', help='Operation mode')
    parser.add_argument('-n', '--max_order', type=int, default=5, help='Max (positive) polynomial order')
    parser.add_argument('-N', '--max_negative_order', type=int, default=0, help='Max (negative) polynomial order')
    parser.add_argument('-r', '--normalize', action='store_true', help='Normalize numeric features')
    return parser.parse_args()
#-------------------------------------------------------------------------
def main():
    args = parse_arguments()

    res = sanity_check(args)
    if res:
        print(res)
        quit()
    original_df, meta = load_arff_to_dataframe(args.arff_file)

    if args.mode == 'test_orders':
        test_polynomial_orders(original_df, meta, args.target_attribute, args.max_order, args.max_negative_order, args.normalize)
        return

    processed_df, norm_data = preprocess_data(original_df, meta, args.target_attribute, args.max_order, args.max_negative_order, args.normalize)
    actuals, predictions, errors = test_each_row(processed_df, args.target_attribute)
    print(f"Mean Absolute Error: {np.mean(errors)}")

    model, mse, r2 = train_and_evaluate_full_df(processed_df, args.target_attribute)
    equation = generate_regression_equation(model, processed_df.drop(args.target_attribute, axis=1).columns, args.target_attribute, norm_data)
    print("Regression Equation:")
    print(equation)

    plt.scatter(actuals, predictions)
    plt.xlabel(f'Actual {args.target_attribute}')
    plt.ylabel(f'Predicted {args.target_attribute}')
    plt.title(f'Actual vs Predicted {args.target_attribute}')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.show()


if __name__ == '__main__':
    main()



