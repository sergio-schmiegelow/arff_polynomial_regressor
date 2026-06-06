import argparse
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------
def load_arff_to_dataframe(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    df = df.map(lambda x: x.decode() if isinstance(x, bytes) else x)
    return df, meta
#-------------------------------------------------------------------------
def separate_numeric_nominal_target(df, meta, target_col):
    nominal_cols = [col[0] for col in zip(meta.names(), meta.types()) if col[1] == 'nominal' and col[0] != target_col]
    nominal_df = df[nominal_cols]
    numeric_df = df.drop(columns=nominal_cols + [target_col])
    target_df = df[target_col]
    return numeric_df, nominal_df, target_df
#-------------------------------------------------------------------------
def create_polynomial_features(numeric_df, order):
    numeric_cols = numeric_df.columns
    for p in range(2, order + 1):
        for col in numeric_cols:
            numeric_df[f'{col}^{p}'] = numeric_df[col] ** p
    return numeric_df
#-------------------------------------------------------------------------
def one_hot_encode_nominal(nominal_df):
    return pd.get_dummies(nominal_df, dtype=int)
#-------------------------------------------------------------------------
def preprocess_data(original_df, meta, target_col, order):
    numeric_df, nominal_df, target_df = separate_numeric_nominal_target(original_df, meta, target_col)
    numeric_df = create_polynomial_features(numeric_df, order)
    one_hot_encoded = one_hot_encode_nominal(nominal_df)
    processed_df = pd.concat([numeric_df, one_hot_encoded, target_df], axis=1)
    return processed_df
#-------------------------------------------------------------------------
def train_and_evaluate_full_df(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    return model, mse, r2 
#-------------------------------------------------------------------------
def train_and_evaluate_single_line(df, target_col, test_index):
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
def generate_regression_equation(model, feature_names, target_col):
    coefficients = model.coef_
    intercept = model.intercept_
    equation = f"{target_col} = {intercept:.4f}"
    for coef, name in zip(coefficients, feature_names):
        equation += f" + ({coef:.4f} * {name})"
    return equation
#-------------------------------------------------------------------------
def test_polynomial_orders(df, meta, target_col, max_order):
    orders = []
    mse_values = []
    for order in range(1, max_order + 1):
        print(f"\nTesting Polynomial Order: {order}")
        processed_df = preprocess_data(df, meta, target_col, order)
        actuals, predictions, errors = test_each_row(processed_df, target_col)
        mse = mean_squared_error(actuals, predictions)
        print(f"Mean Squared Error for Order {order}: {mse}")
        orders.append(order)
        mse_values.append(mse)
    plt.plot(orders, mse_values, marker='o')
    plt.xlabel('Polynomial Order')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Polynomial Order')
    plt.xticks(orders)
    plt.grid()
    plt.show()
#-------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='Multipolynomial regression on ARFF data.')
    parser.add_argument('-f', '--arff_file', required=True, help='Input ARFF file')
    parser.add_argument('-t', '--target_attribute', required=True, help='Attribute to be predicted')
    parser.add_argument('-m', '--mode', choices=['create_model', 'test_orders'], default='create_model', help='Operation mode')
    parser.add_argument('-M', '--max_order', type=int, default=6, help='Max order for test_orders')
    return parser.parse_args()
#-------------------------------------------------------------------------
def main():
    args = parse_arguments()
    original_df, meta = load_arff_to_dataframe(args.arff_file)

    if args.mode == 'test_orders':
        test_polynomial_orders(original_df, meta, args.target_attribute, args.max_order)
        return

    processed_df = preprocess_data(original_df, meta, args.target_attribute, args.max_order)
    actuals, predictions, errors = test_each_row(processed_df, args.target_attribute)
    print(f"Mean Absolute Error: {np.mean(errors)}")

    model, mse, r2 = train_and_evaluate_full_df(processed_df, args.target_attribute)
    equation = generate_regression_equation(model, processed_df.drop(args.target_attribute, axis=1).columns, args.target_attribute)
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



