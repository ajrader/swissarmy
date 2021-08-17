# Dependencies: scikit-learn (pip install --upgrade scikit-learn),
# Import packages

# Import installed packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations, product
from numpy.random import RandomState
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, \
    BaggingRegressor, BaggingClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNet, Lasso, LassoLars
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
# try: #0.16.0+ features
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import LinearSVR


# except ImportError:
#    print("Consider upgrading scikit-learn")
# Import local packages


# module and the new CV iterations. See http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection


def impute_categorical(dataframe: pd.DataFrame, variable_list: list, value_for_missing: str = "Missing",
                       inplace: bool = False) -> pd.DataFrame:
    """All NaN values in dataframe are replaced with "Missing"

    Parameters
    ----------
    dataframe: pandas dataframe

    variable_list: list
        A list of all the categorical columns to replace

    value_for_missing: string, optional (default="Missing")
    """
    # Change from list to dictionary where each key is a column and the value
    # is the value_for_missing. Creating this is required. Otherwise all missing
    # values in the entire dataframe would be filled with the value_for_missing
    values = dict(zip(variable_list, [value_for_missing] * len(variable_list)))
    if inplace:
        dataframe.fillna(values, inplace=True)
    else:
        return dataframe.fillna(values)


def bin_categorical(X, columns_to_bin="all", min_percent_of_total=0.05,
                    replacement_value="Other", rows_to_scan=5000, test=False):
    """Any categories that have less than min_percent_of_total are replaced with
    the replacement_value. Works inplace.

    Parameters
    ----------
    X: pandas dataframe

    columns_to_bin: string or list, optional (default="all")
        If "all", imputes all object columns with missing values. Other
        strings are interpreted to represent column names and you can
        pass a list of column names to impute.

    min_percent_of_total: float, optional (default=0.05)
        For each category, calculates what percent it makes up of the
        total. If it is less than or equal to min_percent_of_total, the
        value is replaced with the replacement_value

    replacement_value: string, optional (default="Other")
        Replaces small categories with this value

    test: boolean, optional (default=False)
        If True, does not affect the data. Instead prints counts of each
        variable and the percent threshold needed to bin it.
    """
    # Handles convenience args
    if rows_to_scan is None or rows_to_scan == "all":
        rows_to_scan = X.shape[0]
    else:
        if rows_to_scan > X.shape[0]:
            rows_to_scan = X.shape[0]

    if columns_to_bin == "all":
        columns_to_bin = [cols for cols in X.columns if X.dtypes[cols] == "object"]
    elif type(columns_to_bin) == str:
        columns_to_bin = [columns_to_bin]

    # Keeps track of the values to keep. Values not included are binned.
    # By keeping track of only the values to keep, we only need to scan
    # a small part of the dataset.
    for variable in columns_to_bin:
        counts = X[variable][:rows_to_scan].value_counts()
        variables_to_not_bucket_index = counts > rows_to_scan * min_percent_of_total
        large_counts = counts.index[variables_to_not_bucket_index]
        if test:
            counts = pd.DataFrame(counts, columns=["counts"])
            counts["percent"] = counts / counts.sum()
            counts["bucketed?"] = -variables_to_not_bucket_index
            counts["cumsum"] = counts["percent"].cumsum()
            print(counts)
            return counts
        else:
            X.loc[-X[variable].isin(large_counts), variable] = replacement_value


# This is to add the categorical variables to the model to help in imputation
def get_number_of_unique_values(X: pd.DataFrame, columns: str = "all", rows_to_scan: int = 10000,
                                objects_only: bool = False, return_series: bool = False,
                                skip_nans: bool = True):
    """Returns a Series with the number of unique values of each object column
    sorted from least to most.

    Parameters
    ----------
    :param X: pandas df

    :param columns: list or string, optional (default="all")
        Gets unique values for list of columns or a single column.
        If "all" uses all columns.

    :param rows_to_scan: integer or 'all', optional (default=10000)
        If 'all' uses entire df. Else, uses at max rows_to_scan.

    :param objects_only: boolean, optional (default="False")
        If true, only object type columns are analyzed

    :param return_series: boolean, optional (default="False")
        If True, returns Series. If False only prints it.

    :param skip_nans: boolean, optional (default = True)
        if True, don't count nulls as a 
    """
    if skip_nans:
        print("skip_nans not implemented yet")

    if rows_to_scan > X.shape[0] or rows_to_scan == "all":
        rows_to_scan = X.shape[0]
    unique_counts = pd.Series()

    if columns == "all":
        columns = X.columns
    elif type(columns) == str:
        columns = [columns]

    for variables in columns:
        if not objects_only or X.dtypes[variables] == "object":
            list_of_unique_values = X[variables][:rows_to_scan].unique()
            number_of_unique_values = len(list_of_unique_values)
            #             if skip_nans and np.isnan(list_of_unique_values).any():
            #                 number_of_unique_values -= 1
            unique_counts[variables] = number_of_unique_values

    unique_counts.sort_values()
    pd.set_option('display.max_rows', len(X))
    print(unique_counts)
    pd.set_option('display.max_rows', 0)

    if return_series:
        return unique_counts


def n_valid_rows(n, X):
    """ Return n or len(X), whichever is shorter"""
    if n > X.shape[0]:
        return X.shape[0]
    return n


def transform_categorical(X, y, col_name):
    """
    Returns a dataframe of mappings for categorical variables where
    each category is mapped to the mean response variable.
    """
    temp = pd.DataFrame(pd.crosstab(X[col_name], y).apply(lambda x: x[1] / float(x.sum()), axis=1))
    temp.columns = [str(col_name) + "_num"]
    temp[col_name] = temp.index
    return temp


def add_transformed_categorical(X, y, col_name, drop_original=True):
    """
    Return X where each category in the categorical variables is replaced with
    the mean of the category.
    """
    X = pd.merge(X, transform_categorical(X, y, col_name), how="left", on=col_name)
    if drop_original:
        X.drop([col_name], axis=1, inplace=True)
    return X


def dummy_variables(X, columns_to_dummy, drop_one_column=False,
                    rows_to_scan=10000, special_column_rules=None, dummy_na=False):
    """
    Replace categorical columns with dummy variables. Return df and list of base
    variables.

    Parameters
    ----------
    X: pandas dataframe

    columns_to_dummy: list
            List of columns to transform

    drop_one_column: boolean, optional (default=False)
        Avoid the dummy variable trap in linear models. Should be
        false for tree-based methods.


    rows_to_scan: integer, optional (default=10000)
        Finds the most popular column in the first rows_to_scan number
        of rows. Drops this column if drop_one_column is True and if
        special_column_row_rules is blank.

    special_column_rules: collection of tuples (default=None)
        Override rows_to_scan option for variables by passing a list or
        dictionary of tuples. The first element of each tuple must be
        the column name. If the second element is an int, that many
        rows will be scanned and the most popular class will be used.
        If the second element is None, all the rows will be scanned
        and the most popular element will be used. If the second element
        is a string, the dummy variable that corresponds to that string
        name will be dropped.

    dummy_na: boolean, optional (default=False)
        Add a column to indicate NaNs, if False NaNs are ignored.
    """

    # Make a list of columns to drop for dummies
    if drop_one_column:
        if special_column_rules:
            special_column_rules = dict(special_column_rules)
        else:
            special_column_rules = {}
        base_columns = []
        for col in columns_to_dummy:
            rows = special_column_rules.get(col, rows_to_scan)
            if (rows == "all") or (rows is None):
                base_columns.append(str(col) + "_" + str(X[col].value_counts().index[0]))
            elif type(rows) == str:
                base_columns.append(str(col) + "_" + rows)
            else:
                rows = n_valid_rows(rows, X) - 1
                base_columns.append(str(col) + "_" + str(X.loc[:rows, col].value_counts().index[0]))

    # Include dummies in dataframe
    X = pd.get_dummies(X, prefix=columns_to_dummy, dummy_na=dummy_na, columns=columns_to_dummy)

    # Drop most represented variable from each column
    if drop_one_column:
        X.drop(base_columns, axis=1, inplace=True)

    return X  # , base_columns


# Impute numeric with mean
def impute_with_mean(X, columns_to_impute='all',
                     keep_dummies=True,
                     impute_inf=True,
                     rows_to_scan=10000):
    """Use the mean of each variable to impute the data

    Parameters
    ----------
    X: pandas dataframe
        The input samples.

    columns_to_impute: string or list, optional (default="all")
        If "all", imputes all columns with missing values. Other
        strings are interpreted to represent column names and you can
        pass a list of column names to impute.

    impute_inf: boolean, optional (default=True)
        If True, also imputes inf values with the mean.

    keep_dummies: boolean, optional (default=True)
        For all columns returns a column with a 0 if a valid number and
        a 1 if it was NaN. Columns have the same name with "_d"
        appended.

    rows_to_scan: integer, optional (default=10000)
        Calculates the mean based off of these many examples. If None,
        uses the entire dataset.
    """
    # Enable flexibility of columns to impute
    if columns_to_impute == "all":
        columns_to_impute = [cols for cols in X.columns if X.dtypes[cols] != "object"]
    elif type(columns_to_impute) == str:
        columns_to_impute = [columns_to_impute]

    # Enable rows_to_scan functionality
    if rows_to_scan > X.shape[0] or rows_to_scan is None:
        rows_to_scan = X.shape[0]

    values = dict(zip(columns_to_impute, X[columns_to_impute][:rows_to_scan].mean()))

    if keep_dummies:
        dummy_missing = X[columns_to_impute].isnull().astype("int")
        dummy_columns = ["%s_d" % s for s in columns_to_impute]
        dummy_missing.columns = dummy_columns
        X = pd.concat([X.fillna(values), dummy_missing], axis=1)
    else:
        if impute_inf:
            X = X.replace(np.inf, values)
        X = X.fillna(values)
    return X


def hist_of_numeric(X):
    """Histogram of all numeric variables of df"""
    fig = plt.figure(figsize=(10, 3))
    for col in get_numeric(X):
        print(col)
        X[col].hist(bins=50)
        plt.show()


def fix_numeric_outliers(X, variable, min_value=None, max_value=None, values_to_skip=["Missing"]):
    """All values outside the range are set to the nearest valid value.

    Parameters
    ----------
    X: pandas dataframe

    variable: string
        The column name.

    min_value: integer or float, optional (default=None)
        Values less than this will become this value

    max_value: integer or float, optional (default=None)
        Values greater than this will become this value

    values_to_skip: list, optional (default=["Missing"])
        Any value contained in this list will be skipped.
        Useful for strings such as "Missing" which are
        always viewed as greater than the max numeric
        value.

    TODO
    ----
    Infer the best range
    """
    if values_to_skip is None:
        values_to_skip = True
    else:
        temp_mask = ~X[variable].isin(values_to_skip)

    if min_value is not None:
        values_below_min = (X[variable] < min_value) & temp_mask
        X.loc[values_below_min, variable] = min_value

    if max_value is not None:
        values_above_max = (X[variable] > max_value) & temp_mask
        X.loc[values_above_max, variable] = max_value

    return X


def get_columns_with_nulls(X, columns_to_scan="all", rows_to_scan=100000):
    """Returns a list of columns that contain nulls in a dataframe

    Parameters
    ----------
    X: pandas dataframe

    [Not implemented] columns_to_scan: string or list, optional (default="all")
        Pass a list of the columns to scan

    [Not implemented] rows_to_scan: integer or None (default=100000)
        Will scan at most this many rows. If None, scans them all.

    TODO: Add a piece that scans for nulls in the first few rows

    updated on June.9th 2015:

    rows_to_scan: int or float or 'all', default=100000
    If int, only check the first rows_to_scan rows.
    If float, only check the first rows_to_scan fraction of all the rows.
    If 'all', check all the rows.

    columns_to_scan: string or list, optional (default="all")
    If "all", check all columns. Other
    strings are interpreted to represent column names and you can
    pass a list of column names to check.

    """
    rows_to_scan = get_rows_to_scan(rows_to_scan, X.shape[0])

    columns_to_scan = get_list_of_columns_to_check(columns_to_scan, X.columns)
    mask = np.array(X[columns_to_scan][:rows_to_scan].count() < rows_to_scan)
    return list(np.array(columns_to_scan)[mask])


def perfect_collinearity_test(X, min_rows="infer", max_rows=None):
    """X is a pandas dataframe.
    max_rows: Most rows the model will use for a variable
    """
    # Sets the minimum number of rows to start with.
    if min_rows == "infer":
        rows_to_use = 2 * X.shape[1]
        if rows_to_use > X.shape[0]:
            rows_to_use = X.shape[0]
    else:
        rows_to_use = min_rows

    # Sets the maximum number of rows to use.
    if max_rows is None:
        max_rows = X.shape[0]

    columns_in_dataframe = X.columns

    # Template for printing even columns
    template = "{0:%s}{1:13}{2:16}" % len(max(X.columns, key=lambda x: len(x)))

    # Series to save results
    results = pd.Series()

    # Runs a regression of every x against all other X variables.
    # Starts with a small dataset and if R^2 == 1, doubles the size
    # of the dataset until greater than max_rows.
    for temp_y_variable in columns_in_dataframe:
        rows_to_use_base = rows_to_use
        while True:
            X_master = X[:rows_to_use_base]
            temp_X_variables = [col for col in columns_in_dataframe if col != temp_y_variable]
            y_temp = X_master[temp_y_variable]
            X_temp = X_master[temp_X_variables]
            lin_model = LinearRegression()
            lin_model.fit(X_temp, y_temp)
            R_2 = lin_model.score(X_temp, y_temp)
            if R_2 != 1 and R_2 >= 0 or rows_to_use_base >= max_rows:
                if R_2 == 1:
                    print("")
                    print(temp_y_variable + ": PERFECT COLLINEARITY ********")
                    temp_series = pd.Series(lin_model.coef_, index=temp_X_variables)
                    print(list(temp_series[temp_series.round(9) != 0].index))
                    print("")
                else:
                    print(template.format(temp_y_variable, " VIF = " + str(round((1.0 / (1.0 - R_2)), 1)),
                                          "R^2 = " + str(round(R_2, 4))))
                results[temp_y_variable] = R_2
                break
            rows_to_use_base += rows_to_use_base
            if rows_to_use_base > X.shape[0]:
                rows_to_use_base = X.shape[0]
    return results


def perfect_collinearity_test_simple(X, min_rows="infer", max_rows=None):
    """X is a pandas dataframe.
    max_rows: Most rows the model will use for a variable

    This simper version is used in get_initial_analysis
    """
    # Sets the minimum number of rows to start with.
    if min_rows == "infer":
        rows_to_use = 2 * X.shape[1]
        if rows_to_use > X.shape[0]:
            rows_to_use = X.shape[0]
    else:
        rows_to_use = min_rows

    # Sets the maximum number of rows to use.
    if max_rows is None:
        max_rows = X.shape[0]

    columns_in_dataframe = X.columns

    # Series to save results
    results = pd.Series()

    # Runs a regression of every x against all other X variables.
    # Starts with a small dataset and if R^2 == 1, doubles the size
    # of the dataset until greater than max_rows.
    for temp_y_variable in columns_in_dataframe:
        rows_to_use_base = rows_to_use
        while True:
            X_master = X[:rows_to_use_base]
            temp_X_variables = [col for col in columns_in_dataframe if col != temp_y_variable]
            y_temp = X_master[temp_y_variable]
            X_temp = X_master[temp_X_variables]
            lin_model = LinearRegression()
            lin_model.fit(X_temp, y_temp)
            R_2 = lin_model.score(X_temp, y_temp)
            if R_2 != 1 and R_2 >= 0 or rows_to_use_base >= max_rows:
                results[temp_y_variable] = R_2
                break
            rows_to_use_base += rows_to_use_base
            if rows_to_use_base > X.shape[0]:
                rows_to_use_base = X.shape[0]
    return results


def graph_feature_importances(model, feature_names, autoscale=True, headroom=0.05, width=10, summarized_columns=None):
    """
    Graphs the feature importances of a random decision forest using a horizontal bar chart.
    Probably works but untested on other sklearn.ensembles.

    Parameters
    ----------
    ensemble = Name of the ensemble whose features you would like graphed.
    feature_names = A list of the names of those featurs, displayed on the Y axis.
    autoscale = True (Automatically adjust the X axis size to the largest feature +.headroom) / False = scale from 0 to 1
    headroom = used with autoscale, .05 default
    width=figure width in inches
    summarized_columns = a list of column prefixes to summarize on, for dummy variables (e.g. ["day_"] would summarize all day_ vars
    """

    if autoscale:
        x_scale = model.feature_importances_.max() + headroom
    else:
        x_scale = 1

    feature_dict = dict(zip(feature_names, model.feature_importances_))

    if summarized_columns:
        # some dummy columns need to be summarized
        for col_name in summarized_columns:
            # sum all the features that contain col_name, store in temp sum_value
            sum_value = sum(x for i, x in feature_dict.iteritems() if col_name in i)

            # now remove all keys that are part of col_name
            keys_to_remove = [i for i in feature_dict.keys() if col_name in i]
            for i in keys_to_remove:
                feature_dict.pop(i)
            # lastly, read the summarized field
            feature_dict[col_name] = sum_value

    results = pd.Series(feature_dict.values(), index=feature_dict.keys())
    results.sort(axis=1)
    results.plot(kind="barh", figsize=(width, len(results) / 4), xlim=(0, x_scale))


def printall(X, max_rows=10):
    from IPython.display import display, HTML
    display(HTML(X.to_html(max_rows=max_rows)))


def get_categorical(X):
    """Return list of object dtypes variables"""
    return list(X.columns[X.dtypes == "object"])


def get_numeric(X):
    """Return list of numeric dtypes variables"""
    return X.dtypes[X.dtypes.apply(lambda x: str(x).startswith(("float", "int", "bool")))].index.tolist()


def describe_categorical(X):
    from IPython.display import display, HTML
    Y = get_categorical(X)
    if len(Y) > 0:
        display(HTML(X[Y].describe().to_html()))
        print("%d rows x %d columns of categorical variables." % X[Y].shape)
    else:
        print("There are no categorical variables in this DataFrame.")


###############################################################################
# will add new things here

class DataFrameConverter(BaseEstimator, TransformerMixin):
    """
    Deprecated. Use DataFrameTransformer instead.
    """

    def __init__(self, columns=None):
        self.columns = list(columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns=self.columns)
        X = X.convert_objects(
            convert_numeric=True)  # If there are inconsistencies, stick this in the fit method and save
        # the dtypes, then apply the dtypes in the transform method when the data
        # is being read in. Just sample the data when inferring the dtypes for speed
        return X


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    Use in a pipeline that has custom transformers if the
    pipeline is fed into a gridsearch object. Needed for
    scikit-learn 0.15.x and earlier.

    constructor: DataFrameTransformer(df)
    """

    def __init__(self, X_columns, X_dtypes):
        """
        X_columns = X.columns
        X_dtypes = X.dtypes
        """
        self.X_columns = X_columns
        self.X_dtypes = X_dtypes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns=list(self.X_columns))
        for col, dtype in zip(X, self.X_dtypes):
            X[col] = X[col].astype(dtype)
        return X


def get_list_of_columns_to_check(columns_to_check, all_columns):
    if type(columns_to_check) == str:
        if columns_to_check == 'all':
            columns_to_check = list(all_columns)
        else:
            if columns_to_check in all_columns:
                columns_to_check = [columns_to_check]
            else:
                columns_to_check = []
    else:
        columns_to_check = [col for col in columns_to_check if col in all_columns]
    return columns_to_check


def get_rows_to_scan(rows_to_scan, max_row):
    if type(rows_to_scan) == int:
        if rows_to_scan > max_row:
            rows_to_scan = max_row
    elif type(rows_to_scan) == float and rows_to_scan <= 1 and rows_to_scan >= 0:
        rows_to_scan = int(rows_to_scan * max_row)
    else:
        rows_to_scan = max_row
    return rows_to_scan


def get_columns_with_all_nulls(X, columns_to_check='all', rows_to_scan='all'):
    """Returns a list of columns that have no available value in a dataframe

    Parameters
    ----------
    X: pandas dataframe

    columns_to_check: string or list, optional (default="all")
        If "all", check all columns. Other
        strings are interpreted to represent column names and you can
        pass a list of column names to check.

    rows_to_scan: int or float or 'all', default='all'
        If int, only check the first rows_to_scan rows.
        If float, only check the first rows_to_scan fraction of all the rows.
        If 'all', check all the rows.
    """
    rows_to_scan = get_rows_to_scan(rows_to_scan, X.shape[0])
    columns_to_check = get_list_of_columns_to_check(columns_to_check, X.columns)
    mask = np.array(X[columns_to_check][:rows_to_scan].count() == 0)
    return list(np.array(columns_to_check)[mask])


def get_columns_not_all_nulls(X, columns_to_check='all', rows_to_scan='all'):
    """Returns a list of columns that have at least one available value in a dataframe

    Parameters
    ----------
    X: pandas dataframe

    columns_to_check: string or list, optional (default="all")
        If "all", check all columns. Other
        strings are interpreted to represent column names and you can
        pass a list of column names to check.

    rows_to_scan: int or float or 'all', default='all'
        If int, only check the first rows_to_scan rows.
        If float, only check the first rows_to_scan fraction of all the rows.
        If 'all', check all the rows.
    """
    columns_to_check = get_list_of_columns_to_check(columns_to_check, X.columns)
    remove_columns = get_columns_with_all_nulls(X, columns_to_check, rows_to_scan)
    return list(set(columns_to_check) - set(remove_columns))


def get_percentage_of_nulls(X, columns_to_check='all', rows_to_scan='all', only_nulls=True, deci=None):
    """Returns a dict of percentage of null values in each column

    Parameters
    ----------
    X: pandas dataframe

    columns_to_check: string or list, optional (default="all")
        If "all", check all columns. Other
        strings are interpreted to represent column names and you can
        pass a list of column names to check.

    rows_to_scan: int or float or 'all', default='all'
        If int, only check the first rows_to_scan rows.
        If float, only check the first rows_to_scan fraction of all the rows.
        If 'all', check all the rows.

    only_nulls: boolean (default=True)
        If True, only return values for columns that have at least one missing value.
    deci: None or int (default=None)
        If int, the number of decimals will be set by this number.

    """
    rows_to_scan = get_rows_to_scan(rows_to_scan, X.shape[0])
    columns_to_check = get_list_of_columns_to_check(columns_to_check, X.columns)

    percentage = {}
    for col in columns_to_check:
        if not (only_nulls and X[col][:rows_to_scan].count() == rows_to_scan):
            temp = 1 - X[col][:rows_to_scan].count() / float(rows_to_scan)
            if deci != None:
                percentage[col] = round(temp, deci)
            else:
                percentage[col] = temp

    return percentage


def get_percentage_of_nulls_pd(X, columns_to_check='all', rows_to_scan='all', only_nulls=True, deci=None):
    """Returns a pandas dataframe of percentage of null values in each column

    Parameters
    ----------
    X: pandas dataframe

    columns_to_check: string or list, optional (default="all")
        If "all", check all columns. Other
        strings are interpreted to represent column names and you can
        pass a list of column names to check.

    rows_to_scan: int or float or 'all', default='all'
        If int, only check the first rows_to_scan rows.
        If float, only check the first rows_to_scan fraction of all the rows.
        If 'all', check all the rows.

    only_nulls: boolean (default=True)
        If True, only return values for columns that have at least one missing value.
    deci: None or int (default=None)
        If int, the number of decimals will be set by this number.

    """
    per_null_dict = get_percentage_of_nulls(X, deci=deci, columns_to_check=columns_to_check,
                                            rows_to_scan=rows_to_scan, only_nulls=only_nulls)

    per_null_table = pd.DataFrame.from_dict(per_null_dict, orient='index')
    per_null_table.columns = ['Percentage']
    per_null_table.sort_values(by='Percentage', ascending=False, inplace=True)
    return per_null_table


def get_value_counts(X, columns, cate_cap=30):
    """
    returns a dictionary, key is columns name, values is also a dictionary showing counts of each category

    Parameters
    --------------

    X: pandas dataframe

    columns: a list of column names

    cate_cap: int. default=30
       If number of categories is larger than this number, then value of the column name will be "There are more than 30 categories. Please check this column"
    --------------


    """
    counts = {}
    for col in columns:
        temp = dict(X[col].value_counts())
        temp['NaN'] = X[col].isnull().sum()
        if len(temp) > cate_cap:
            counts[col] = 'There are more than %d categories. Please check this column.' % cate_cap
        else:
            counts[col] = temp
    return counts


def get_value_counts_pd(X, columns, cate_cap=30):
    """
    returns a pandas dataframe showing counts of each category of each column

    Parameters
    --------------

    X: pandas dataframe

    columns: a list of column names

    cate_cap: int. default=30
       If number of categories is larger than this number, then "Too many categories" will be showed
    --------------


    """
    count_dict = get_value_counts(X, columns=columns, cate_cap=cate_cap)
    idx_tuple = []
    value_counts = []
    for col in count_dict.keys():
        if type(count_dict[col]) == str:
            idx_tuple += [(col, col)]
            value_counts += ['Too many categories']
        else:
            temp = [[col] * len(count_dict[col]), count_dict[col].keys()]
            idx_tuple += list(zip(*temp))
            value_counts += count_dict[col].values()
    multiidx = pd.MultiIndex.from_tuples(idx_tuple, names=['column', 'category'])
    counts_df = pd.DataFrame(value_counts, columns=['counts'], index=multiidx)
    return counts_df


def get_initial_analysis(X, y=None, tX=None, cate_cap=30, random_state=None, rows_to_scan=10000, sample_size=3000,
                         problem_type='infer'):
    """
    print some initial information of the dataframe including

    shape of the data, column names, numeric columns, description of numeric columns, categorical columns, value_counts of categorical
    columns, percentage of missing values, value_counts of y, counts of missing values of y, initial benchmark of a simple model,
    perfect collinearity, and other potential issues.

    Parameters
    --------------

    X: pandas dataframe, training data

    y: pandas series or np array. True labels of the training data X. Optional, default=None.
       If not None, then there will be some analyses related to y

    tX: pandas dataframe, testing data. Optional, default=None
       If not None, then there will be some analyses related to testing data

    random_state: int. Default = None. used to set the seed of sampling.

    rows_to_scan: int or float or 'all', default='all'
        If int, only check the first rows_to_scan rows.
        If float, only check the first rows_to_scan fraction of all the rows.
        If 'all', check all the rows.
        This can be used when data is large.

    sample_size: int. Should be less than rows_to_scan.
        This determins the size of sample used to get an initial benchmark using a simple model.

    cate_cap: int. default=30
       The analysis will include the value_counts of the categorical columns.
       If number of categories is larger than this number, then the value_counts of this column will be suppressed.

    problem_type: string. default='infer' (automatically decide the problem type)
       other options: 'regression', 'classification'

    --------------

    """
    from numpy.random import RandomState
    random_generator = RandomState(random_state)
    rows_to_scan = get_rows_to_scan(rows_to_scan, X.shape[0])
    sample_idx = random_generator.choice(range(X.shape[0]), rows_to_scan, replace=False)
    sampleX = X.iloc[sample_idx]

    print('For training data:')
    print('\n')
    print('Shape of the original data: ', X.shape)
    print('\n')
    print('Rows to scan for initial analysis: ', rows_to_scan)
    print('\n')
    print('Columns', list(sampleX.columns))
    print('\n')
    nu = get_numeric(sampleX)
    print('Numeric columns: ', nu)
    print('Description of numeric columns: ')
    from IPython.display import display, HTML
    if len(nu) > 0:
        display(HTML(sampleX[nu].describe().to_html()))
    print('\n')
    print('Categorical columns: ', list(get_categorical(sampleX)))
    if len(list(get_categorical(sampleX))) > 0:
        print('Value counts of categorical columns: ')
        count_dict = get_value_counts(sampleX, get_categorical(sampleX), cate_cap)
        idx_tuple = []
        value_counts = []
        for col in count_dict.keys():
            if type(count_dict[col]) == str:
                idx_tuple += [(col, col)]
                value_counts += ['Too many categories']
            else:
                temp = [[col] * len(count_dict[col]), count_dict[col].keys()]
                idx_tuple += list(zip(*temp))
                value_counts += count_dict[col].values()
        multiidx = pd.MultiIndex.from_tuples(idx_tuple, names=['column', 'category'])
        counts_df = pd.DataFrame(value_counts, columns=['counts'], index=multiidx)
        display(HTML(counts_df.to_html()))
        print("\n")
    print('Percentage of missing values (columns not displayed have no missing values): ')

    per_null_dict = get_percentage_of_nulls(sampleX, deci=3)
    per_null_table = pd.DataFrame.from_dict(per_null_dict, orient='index')
    per_null_table.columns = ['Percentage']
    per_null_table.sort_values(by='Percentage', ascending=False, inplace=True)
    display(HTML(per_null_table.to_html()))

    if y is not None:
        sampley = y.iloc[sample_idx]
        if problem_type == 'infer':
            prob_type = get_problem_type(y)
        if prob_type == 'classification':
            value_counts_y = dict(sampley.value_counts())
            value_counts_y_df = pd.DataFrame(value_counts_y.values(), columns=['Counts'], index=value_counts_y.keys())
            print('Value counts of y: ', )
            display(HTML(value_counts_y_df.to_html()))
        print()
        print('Correlation coefficients with numeric columns: ')
        corrs = {}
        for col in get_numeric(sampleX):
            temp_mask = np.isfinite(sampleX[col]) & np.isfinite(sampley)
            if temp_mask.sum() > 0:
                corrs[col] = np.corrcoef(sampleX[col][temp_mask], sampley[temp_mask])[0][1]
            else:
                corrs[col] = 'No available observations'
        corrs_df = pd.DataFrame(corrs.values(), index=corrs.keys(), columns=['Correlation coefficient'])
        display(HTML(corrs_df.to_html()))
        print("\n")
        print('Benchmark on a small sample:')
        if sample_size > rows_to_scan:
            sample_size = rows_to_scan
        print('Sample size: ', sample_size)
        sampleX_temp = sampleX[:sample_size]
        sampleyy = sampley[:sample_size]
        temp_mask = sampleyy.notnull()
        sampleX_temp = sampleX_temp[temp_mask]
        sampleyy = sampleyy[temp_mask]
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("null", RemoveAllNull()),
                         ("cat", ConvertCategorical(get_categorical(sampleX_temp))),
                         ("imp", ImputeData(columns_to_impute='all'))])
        trans_sampleX_temp = pipe.fit_transform(sampleX_temp)

        print('Problem type: ', prob_type)
        if sample_size >= 2000:
            if prob_type == 'classification':
                rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=random_state)
                rf.fit(trans_sampleX_temp, sampleyy)
                from sklearn.metrics import roc_auc_score
                dummy_y = pd.get_dummies(sampleyy)
                print('c-stat of simple random forest: ', roc_auc_score(dummy_y, rf.oob_decision_function_))
            else:
                rf = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1, random_state=random_state)
                rf.fit(trans_sampleX_temp, sampleyy)
                from sklearn.metrics import mean_squared_error
                print('MSE of simple random forest: ', mean_squared_error(sampleyy, rf.oob_prediction_))
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(trans_sampleX_temp, sampleyy, test_size=.25,
                                                                random_state=random_state)
            if prob_type == 'classification':
                rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
                rf.fit(X_train, y_train)
                from sklearn.metrics import roc_auc_score
                dummy_y = pd.get_dummies(y_test)
                print('c-stat of simple random forest: ', roc_auc_score(dummy_y, rf.predict_proba(X_test)))
            else:
                rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)
                rf.fit(X_train, y_train)
                from sklearn.metrics import mean_squared_error
                print('MSE of simple random forest: ', mean_squared_error(y_test, rf.predict(X_test)))

        feature_names = list(trans_sampleX_temp.columns)
        importance_name = [feature_names[i] for i in (np.argsort(rf.feature_importances_))]
        importance_value = np.sort(rf.feature_importances_)
        importance = pd.DataFrame(importance_value, index=importance_name, columns=['Importance'])
        importance.plot(kind='barh', title='Feature Importance', figsize=(7, importance.shape[0] / 3))

    if tX is not None:
        rows_to_scan = get_rows_to_scan(rows_to_scan, tX.shape[0])
        sample_idx = random_generator.choice(range(tX.shape[0]), rows_to_scan)
        sampletX = tX.iloc[sample_idx]
        print('===============================================================')
        print('For testing data:')
        print('\n')
        print('Shape of the data: ', np.shape(tX))
        print("\n")
        print('Rows to scan: ', rows_to_scan)
        print("\n")
        print('Columns:', list(tX.columns))
        print("\n")
        print('Numeric columns: ', get_numeric(tX))

        from scipy import stats
        diff_dist = []
        for col in get_numeric(sampletX):
            if stats.ks_2samp(sampleX[col], sampletX[col])[1] < 0.05:
                diff_dist += [col]
        print('Columns having different distribution in training and testing (at 0.05 critical value): ')
        print(diff_dist)
        print('\n')

        print('Categorical columns: ', list(get_categorical(sampletX)))
        if len(list(get_categorical(sampletX))) > 0:
            print('Value counts of categorical columns: ')
            count_dict = get_value_counts(sampletX, get_categorical(sampletX), cate_cap)
            idx_tuple = []
            value_counts = []
            for col in count_dict.keys():
                if type(count_dict[col]) == str:
                    idx_tuple += [(col, col)]
                    value_counts += ['Too many categories']
                else:
                    temp = [[col] * len(count_dict[col]), count_dict[col].keys()]
                    idx_tuple += list(zip(*temp))
                    value_counts += count_dict[col].values()
            multiidx = pd.MultiIndex.from_tuples(idx_tuple, names=['column', 'category'])
            counts_df = pd.DataFrame(value_counts, columns=['counts'], index=multiidx)
            display(HTML(counts_df.to_html()))
            print('\n')
        print('Percentage of missing values (columns not displayed have no missing values): ')
        per_null_dict = get_percentage_of_nulls(sampletX, deci=3)
        per_null_table = pd.DataFrame.from_dict(per_null_dict, orient='index')
        per_null_table.columns = ['Percentage']
        per_null_table.sort_values(by='Percentage', ascending=False, inplace=True)
        display(HTML(per_null_table.to_html()))
        print()

    print('Potential issues: ')
    print('Columns with all missing values: ', get_columns_with_all_nulls(sampleX))
    unique_value = []
    for col in sampleX.columns:
        temp = sampleX[col].unique()
        temp = [s for s in temp if str(s) != 'nan']
        if len(temp) == 1:
            unique_value += [col]
    print('Columns having only one value: ', unique_value)

    from sklearn.pipeline import Pipeline
    pipe2 = Pipeline([("null", RemoveAllNull()),
                      ("cat", ConvertCategorical(get_categorical(sampleX_temp))),
                      ("imp", ImputeData(columns_to_impute='all'))])
    trans_sampleX = pipe2.fit_transform(sampleX)
    res = perfect_collinearity_test_simple(trans_sampleX)
    print('Perfect collinearity: ', list(res[res == 1].index))

    if tX is not None:
        null_diff = set(get_columns_with_nulls(sampletX, 'all', 'all')) - set(
            get_columns_with_nulls(sampleX, 'all', 'all'))
        print('Columns having missing values in testing but having not in training: ', list(null_diff))
        col_diff_1 = set(sampleX.columns) - set(sampletX.columns)
        col_diff_2 = set(sampletX.columns) - set(sampleX.columns)
        print('Columns in training but not in testing: ', list(col_diff_1))
        print('Columns in testing but not in training: ', list(col_diff_2))

    return None


class ImputeData(BaseEstimator, TransformerMixin):
    def __init__(self,
                 method='mean',
                 columns_to_impute='auto',
                 keep_dummies=True,
                 impute_inf=True,
                 rows_to_scan='all'):
        """
        A class that can be inserted into a pipeline.

        This will impute the missing values of data in selected columns with different methods.
        This supports creating dummy columns and imputing inf values.
        When method is related to numeric columns(mean, median, max), then only numeric columns in the selected columns will be imputed.

        Parameters
        ----------
        X: pandas dataframe

        columns_to_impute: string or list, optional (default="all")
            If "all", check all columns. Other
            strings are interpreted to represent column names and you can
            pass a list of column names to check.

            If 'auto', will automatically impute the columns having missing values


        rows_to_scan: int or float or 'all', default='all'
            This is the number of rows to use to compute the mean, median, max value that will be imputed with
            If int, only check the first rows_to_scan rows.
            If float, only check the first rows_to_scan fraction of all the rows.
            If 'all', check all the rows.
            When method is bfill and ffill, this parameter will not take effect.

        keep_dummies: boolean (default=True)
            If True, create new columns indicating the missing values with names like 'originalname_d'.

        impute_inf: boolean (default=True)
            If True, treat the inf values as missing values.

        method: str (default='mean')
            The method to impute. Methods can be 'mean', 'ffill', 'bfill', 'max', 'median', 'mode'(most frequen value)
            There are also model based methods. They are 'knn' and 'linear_reg'

        returns a pandas dataframe
        """
        self.columns_to_impute = columns_to_impute
        self.rows_to_scan = rows_to_scan
        self.impute_inf = impute_inf
        self.keep_dummies = keep_dummies
        self.method = method

    def fit(self, X, y=None):
        X_temp = X.copy()
        if self.columns_to_impute == 'auto':
            X_temp = X_temp.replace(np.inf, np.nan)
            self.columns_to_impute_in = get_numeric(X_temp)
        else:
            self.columns_to_impute_in = get_list_of_columns_to_check(self.columns_to_impute, X.columns)
            self.columns_to_impute_in = [col for col in self.columns_to_impute_in if col in get_numeric(X_temp)]
            X_temp[self.columns_to_impute_in] = X_temp[self.columns_to_impute_in].replace(np.inf, np.nan)

        self.rows_to_scan_in = get_rows_to_scan(self.rows_to_scan, X.shape[0])

        if self.method in ['mean', 'median', 'max']:
            if self.method == 'mean':
                self.values = dict(
                    zip(self.columns_to_impute_in, X_temp[self.columns_to_impute_in][:self.rows_to_scan_in].mean()))
            elif self.method == 'median':
                self.values = dict(
                    zip(self.columns_to_impute_in, X_temp[self.columns_to_impute_in][:self.rows_to_scan_in].median()))
            elif self.method == 'max':
                self.values = dict(
                    zip(self.columns_to_impute_in, 1 + X_temp[self.columns_to_impute_in][:self.rows_to_scan_in].max()))
        elif self.method == 'mode':
            temp = np.array(X_temp[self.columns_to_impute_in][:self.rows_to_scan_in].mode())[0]
            self.values = dict(zip(self.columns_to_impute_in, temp))
        elif self.method == 'knn':
            self.models = {}
            self.columns_available = list(set(X_temp.columns) - set(get_columns_with_nulls(X_temp, 'all', 'all')))
            self.columns_available = list(set(self.columns_available) - set(self.columns_to_impute))
            availableX = X_temp[self.columns_available][:self.rows_to_scan_in]
            for col in self.columns_to_impute_in:
                tempy = X_temp[col][:self.rows_to_scan_in]
                temp_mask = tempy.notnull()
                tempy = tempy[temp_mask]
                tempX = availableX[temp_mask]
                self.models[col] = KNeighborsRegressor()
                self.models[col].fit(tempX, tempy)
        elif self.method == 'linear_reg':
            self.models = {}
            self.columns_available = list(set(X_temp.columns) - set(get_columns_with_nulls(X_temp, 'all', 'all')))
            self.columns_available = list(set(self.columns_available) - set(self.columns_to_impute))
            availableX = X_temp[self.columns_available][:self.rows_to_scan_in]
            for col in self.columns_to_impute_in:
                tempy = X_temp[col][:self.rows_to_scan_in]
                temp_mask = tempy.notnull()
                tempy = tempy[temp_mask]
                tempX = availableX[temp_mask]
                self.models[col] = LinearRegression()
                self.models[col].fit(tempX, tempy)
        return self

    def transform(self, X, y=None):
        X_temp = X.copy()
        if self.impute_inf:
            X_temp[self.columns_to_impute_in] = X_temp[self.columns_to_impute_in].replace(np.inf, np.nan)
        if self.keep_dummies:
            temp = pd.DataFrame(index=X_temp.index)
            for col in self.columns_to_impute_in:
                temp[(col + '_d')] = X_temp[col].isnull().astype("int")
            X_temp = pd.concat([X_temp, temp], axis=1)
        if self.method in ['bfill', 'ffill']:
            index = X_temp.index
            X_temp = X_temp.reindex(np.random.permutation(X_temp.index))
            X_temp[self.columns_to_impute_in] = X_temp[self.columns_to_impute_in].fillna(method='ffill')
            X_temp[self.columns_to_impute_in] = X_temp[self.columns_to_impute_in].fillna(method='bfill')
            X_temp = X_temp.reindex(index)
        elif self.method in ['mean', 'median', 'max', 'mode']:
            X_temp = X_temp.fillna(self.values)
        elif self.method in ['knn', 'linear_reg']:
            availableX = X_temp[self.columns_available]
            for col in self.columns_to_impute_in:
                temp_mask = X_temp[col].isnull()
                if temp_mask.sum() > 0:
                    X_temp.ix[temp_mask, col] = self.models[col].predict(availableX[temp_mask])

        return X_temp


class RescaleData(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_trans='all', method='standard', rows_to_scan='all'):
        """
        A class that can be inserted into pipeline.
        This will rescale the selected columns of the data using different methods.

        Parameters:
        ------------
        X: pandas dataframe

        columns_to_trans: string or list, optional (default="all")
            If "all", check all columns. Other
            strings are interpreted to represent column names and you can
            pass a list of column names to check.

        rows_to_scan: int or float or 'all', default='all'
            This is the number of rows to use to compute the mean, max value, etc..that will be used to do rescaling
            If int, only check the first rows_to_scan rows.
            If float, only check the first rows_to_scan fraction of all the rows.
            If 'all', check all the rows.

        method: str. default='standard'. Others can be '2_std','max_min','2_norm','1_norm','origin'.
        'origin' means returning the original data without doing rescaling.

        ------------

        returns a pandas dataframe

        """
        self.columns_to_trans = columns_to_trans
        self.method = method
        self.rows_to_scan = rows_to_scan

    def fit(self, X, y=None):
        if self.method != 'origin':
            self.columns_to_trans_in = get_list_of_columns_to_check(self.columns_to_trans, X.columns)
            self.rows_to_scan_in = get_rows_to_scan(self.rows_to_scan, X.shape[0])
            ratio = float(X.shape[0]) / self.rows_to_scan_in
            if self.method == 'standard':
                self.coef1 = X[self.columns_to_trans_in][:self.rows_to_scan_in].mean()
                self.coef2 = X[self.columns_to_trans_in][:self.rows_to_scan_in].std()
            elif self.method == '2_std':
                self.coef1 = X[self.columns_to_trans_in][:self.rows_to_scan_in].mean()
                self.coef2 = X[self.columns_to_trans_in][:self.rows_to_scan_in].std() * 2
            elif self.method == 'max_min':
                self.coef1 = X[self.columns_to_trans_in][:self.rows_to_scan_in].min()
                self.coef2 = X[self.columns_to_trans_in][:self.rows_to_scan_in].max() - self.coef1
            elif self.method == '2_norm':
                self.coef2 = np.sqrt(ratio * np.square(X[self.columns_to_trans_in][:self.rows_to_scan_in]).sum())
                self.coef1 = 0
            elif self.method == '1_norm':
                self.coef2 = (np.abs(X[self.columns_to_trans_in][:self.rows_to_scan_in])).sum() * ratio
                self.coef1 = 0
        return self

    def transform(self, X, y=None):
        if self.method == 'origin':
            return X
        else:
            X_temp = X.copy()
            self.coef2[self.coef2 == 0] = 1.0
            X_temp[self.columns_to_trans_in] = (X_temp[self.columns_to_trans_in] - self.coef1) / self.coef2

            return X_temp


class LogTrans(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_trans='all', trans_flag=True):
        """
        A class that can be inserted into pipeline.
        This will do log(x+1) transformation on selected columns of data

        Parameters
        -----------------
        columns_to_trans: string or list, optional (default="all")
            If "all", check all columns. Other
            strings are interpreted to represent column names and you can
            pass a list of column names to check.
            If "auto", do transformation with all numeric columns in the data.

        trans_flag: boolean, default = True.
            Decide whether to do transformation or not
        -----------------

        returns a pandas dataframe

        """
        self.columns_to_trans = columns_to_trans
        self.trans_flag = trans_flag

    def fit(self, X, y=None):
        if self.columns_to_trans == 'auto':
            self.columns_to_trans_in = get_numeric(X[list(X.columns)])
        else:
            self.columns_to_trans_in = get_list_of_columns_to_check(self.columns_to_trans, X.columns)
            temp_numeric = get_numeric(X[self.columns_to_trans_in])
            temp_not_numeric = set(self.columns_to_trans_in) - set(temp_numeric)
            if len(temp_not_numeric) > 0:
                raise Exception('Columns ' + str(list(temp_not_numeric)) + ' are not numeric!')
        return self

    def transform(self, X, y=None):
        X_temp = X.copy()
        if self.trans_flag:
            X_temp[self.columns_to_trans_in] = np.log(1 + X_temp[self.columns_to_trans_in])
        return X_temp


def impute_data(X, method='mean',
                columns_to_impute='all',
                keep_dummies=True,
                impute_inf=True,
                rows_to_scan='all'):
    """Returns a DateFrame whose missing values will be imputed

    Parameters
    ----------
    X: pandas dataframe

    columns_to_impute: string or list, optional (default="all")
        If "all", check all columns. Other
        strings are interpreted to represent column names and you can
        pass a list of column names to check.
        Will only include numeric columns.

    rows_to_scan: int or float or 'all', default='all'
        This is the number of rows to use to compute the mean, median, max value that will be imputed with
        If int, only check the first rows_to_scan rows.
        If float, only check the first rows_to_scan fraction of all the rows.
        If 'all', check all the rows.
        When method is bfill and ffill, this parameter will not take effect.

    keep_dummies: boolean (default=True)
        If True, create new columns indicating the missing values with names like 'originalname_d'.

    impute_inf: boolean (default=True)
        If True, treat the inf values as missing values.

    method: str (default='mean')
        The method to impute. Methods can be 'mean', 'ffill', 'bfill', 'max', 'median'

    returns a pandas dataframe

    """

    # get the correct number of rows and column names
    rows_to_scan = get_rows_to_scan(rows_to_scan, X.shape[0])
    columns_to_impute = get_list_of_columns_to_check(columns_to_impute, X.columns)
    if impute_inf:
        X_temp = X.copy()
        X_temp[columns_to_impute] = X_temp[columns_to_impute].replace(np.inf, np.nan)
        X = X_temp

    columns_to_impute = get_columns_with_nulls(X, columns_to_impute, rows_to_scan='all')
    columns_to_impute = get_numeric(X[columns_to_impute])

    if keep_dummies:
        temp = pd.DataFrame(index=X.index)
        for col in columns_to_impute:
            temp[(col + '_d')] = X[col].isnull().astype("int")
        X = pd.concat([X, temp], axis=1)

    if method in ['bfill', 'ffill']:
        index = X.index
        X = X.reindex(np.random.permutation(X.index))
        X[columns_to_impute] = X[columns_to_impute].fillna(method=method)
        X = X.reindex(index)
    elif method == 'mean':
        values = dict(zip(columns_to_impute, X[columns_to_impute][:rows_to_scan].mean()))
        X = X.fillna(values)
    elif method == 'median':
        values = dict(zip(columns_to_impute, X[columns_to_impute][:rows_to_scan].median()))
        X = X.fillna(values)
    elif method == 'max':
        values = dict(zip(columns_to_impute, 1 + X[columns_to_impute][:rows_to_scan].max()))
        X = X.fillna(values)
    return X


class ConvertCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns="all", method='factorize', rows_to_scan='all'):
        """
        A class that can be inserted into a pipeline.

        This will convert selected categorical columns to numeric columns using different methods.
        Note that this will DISCARD the categorical columns that are not selected.
        The original categorical columns that are selected will also be DISCARDED.
        (Only keep numeric columns after the converting.)


        (If method = 'dummy', then there will be a column indicating missing values. Column name will be 'column_category')
        (If method = 'factorize', 'group_means', 'value_counts', there will be a category for missing values.)

        Column name will be 'column_f'

        Parameters
        ---------
        X: Pandas dataframe

        categorical_columns: a list of column names, the columns should be categorical. If "all", converts all.

        method: str, default = 'factorize'
        Others can be 'value_counts', 'group_means', 'dummy'

        rows_to_scan: int or float or 'all', default='all'
            This is the number of rows to scan to get related values for categories
            If int, only check the first rows_to_scan rows.
            If float, only check the first rows_to_scan fraction of all the rows.
            If 'all', check all the rows.

        returns a pandas dataframe
        """
        self.method = method
        self.categorical_columns = categorical_columns
        self.rows_to_scan = rows_to_scan

    def fit(self, X, y=None):
        if self.categorical_columns == "all":
            self.categorical_columns = get_categorical(X)
        self.map_values = {}
        self.dummy_values = {}
        self.na_values = {}
        rows_to_scan_in = get_rows_to_scan(self.rows_to_scan, X.shape[0])
        X_temp = X[:rows_to_scan_in].copy()
        for col in self.categorical_columns:
            if self.method == 'factorize':
                map_values = X_temp[col].unique()  # Can sample data here for speed
                self.map_values[col] = {key: index for index, key in enumerate(map_values)}
            elif self.method == 'value_counts':
                self.map_values[col] = dict(X_temp[col].value_counts())
                self.na_values[col + "_f"] = X_temp[col].isnull().sum()
            elif self.method == 'group_means':
                yy = y[:rows_to_scan_in].copy()
                self.map_values[col] = dict(pd.crosstab(X_temp[col], yy).apply(lambda x: x[1] / float(x.sum()), axis=1))
                self.na_values[col + "_f"] = yy[X_temp[col].isnull()].mean()
            elif self.method == 'dummy':
                self.dummy_values[col] = X_temp[col].unique()
        return self

    def transform(self, X, y=None):
        X_temp = X.copy()
        if self.method in ['factorize', 'value_counts', 'group_means']:
            for col in self.categorical_columns:
                X_temp[str(col) + "_f"] = X_temp[col].map(self.map_values[col], "ignore")
                X_temp[str(col) + "_f"] = X_temp[col].map(self.map_values[col], "ignore")
        elif self.method == 'dummy':
            for col in self.categorical_columns:
                for cat in self.dummy_values[col]:
                    if str(cat) == 'nan':
                        X_temp[str(col) + '_' + str(cat)] = X_temp[col].isnull().astype(int)
                    else:
                        X_temp[str(col) + '_' + str(cat)] = (X_temp[col] == cat).astype(int)
        if self.method in ['value_counts', 'group_means']:
            X_temp = X_temp.fillna(self.na_values)
        X_temp = X_temp[get_numeric(X_temp)]
        # Fill all remaining null values with -1
        X_temp = X_temp.fillna(-1)
        return X_temp


class FixNumericOutlier(BaseEstimator, TransformerMixin):
    def __init__(self, criteria_coef=('percentile', 5), fill_with='nearest_value',
                 method='both', columns_to_fix='all', rows_to_scan='all'):
        """
        A class that can be inserted into a pipeline.

        This will fix the outliers in numeric columns of the dataframe. Columns with all nulls or categorical values will not be changed.
        Missing values will remain missing. Inf values will be fixed.

        Parameters
        ---------
        X: Pandas dataframe

        criteria_coef: tuple, default=(percentile, 5). (if 'all', then will not fix anything)
            first entry is criteria, can be 'percentile' and 'sd'
            second entry is coef:  integer
            when criteria='percentile', the top coef percentile extreme values will be taken as outliers.
            when criteria='sd', the values which is coef standard deviations far away from the mean will be taken as outliers.

        fill_with: str, default='nearest_value'. Others can be 'missing'

        method: str, default='both', which means fixing both two sides.
            Others can be 'upper', which means only fixing upper side, and 'lower', which means only fixing lower side.

        columns_to_fix: string or list, optional (default="all")
            If "all", check all columns. Other
            strings are interpreted to represent column names and you can
            pass a list of column names to check.
            If 'auto', will automatically fix numeric columns

        rows_to_scan: int or float or 'all', default='all'
            This is the number of rows to scan to get boundry of outliers
            If int, only check the first rows_to_scan rows.
            If float, only check the first rows_to_scan fraction of all the rows.
            If 'all', check all the rows.

        returns a pandas dataframe
        """
        self.criteria_coef = criteria_coef
        self.fill_with = fill_with
        self.method = method
        self.columns_to_fix = columns_to_fix
        self.rows_to_scan = rows_to_scan

    def fit(self, X, y=None):
        if self.criteria_coef != 'all':

            self.max_val = {}
            self.min_val = {}
            self.coef = self.criteria_coef[1]
            self.criteria = self.criteria_coef[0]
            rows_to_scan_in = get_rows_to_scan(self.rows_to_scan, X.shape[0])

            if self.columns_to_fix == 'auto':
                self.columns_to_fix_in = get_numeric(X)
            else:
                self.columns_to_fix_in = get_list_of_columns_to_check(self.columns_to_fix, X.columns)

            temp_numeric = get_numeric(X[self.columns_to_fix_in])
            temp_not_numeric = set(self.columns_to_fix_in) - set(temp_numeric)
            if len(temp_not_numeric) > 0:
                raise Exception('Columns ' + str(list(temp_not_numeric)) + ' are not numeric!')
            X_temp = X[:rows_to_scan_in].copy()
            for col in self.columns_to_fix_in:
                temp = X_temp[col][np.isfinite(X_temp[col])]
                if self.criteria == 'percentile':
                    self.max_val[col] = np.percentile(temp, 100 - self.coef)
                    self.min_val[col] = np.percentile(temp, self.coef)
                elif self.criteria == 'sd':
                    self.max_val[col] = np.mean(temp) + self.coef * np.std(temp)
                    self.min_val[col] = np.mean(temp) - self.coef * np.std(temp)
        return self

    def transform(self, X, y=None):
        if self.criteria_coef != 'all':
            X_temp = X.copy()
            if self.fill_with == 'nearest_value':
                if self.method == 'upper':
                    for col in self.columns_to_fix_in:
                        X_temp.loc[X_temp[col] > self.max_val[col], col] = self.max_val[col]
                elif self.method == 'lower':
                    for col in self.columns_to_fix_in:
                        X_temp.loc[X_temp[col] < self.min_val[col], col] = self.min_val[col]
                else:
                    for col in self.columns_to_fix_in:
                        X_temp.loc[X_temp[col] > self.max_val[col], col] = self.max_val[col]
                        X_temp.loc[X_temp[col] < self.min_val[col], col] = self.min_val[col]
            elif self.fill_with == 'missing':
                if self.method == 'upper':
                    for col in self.columns_to_fix_in:
                        X_temp.loc[X_temp[col] > self.max_val[col], col] = np.nan
                elif self.method == 'lower':
                    for col in self.columns_to_fix_in:
                        X_temp.loc[X_temp[col] < self.min_val[col], col] = np.nan
                else:
                    for col in self.columns_to_fix_in:
                        X_temp.loc[X_temp[col] > self.max_val[col], col] = np.nan
                        X_temp.loc[X_temp[col] < self.min_val[col], col] = np.nan
            return X_temp
        else:
            return X


class AddInteraction(BaseEstimator, TransformerMixin):
    def __init__(self, add_list=None, degree=2):
        """
        A class that can be inserted into pipeline.

        This will create interaction columns.

        Parameters
        ---------
        X: Pandas dataframe

        add_list: tuple or list.
        If None, it will be a list of column names of all the numeric columns of X
        If list, please insert the column names among which you want to build interactions. For example, ['x1', 'x3', 'x4']
        If tuple, please insert the specific interaction you want to create. For example, (('x1', 'x2'), ('x1', 'x2', 'x3'), ('x4', 'x1'))

        degree: int or list of integers. default=2
        This will only take effect when add_list is a list.
        For example, add_list = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        degree = [3, 5]
        Then the new interaction terms will be all the three-way and five_way interaction among the six columns.
        Values larger than the number of terms or smaller than 2 will be ignored.

        Note: please only include numeric columns.
        New columns names will be the related column names with delimiter asterisk

        returns a pandas dataframe
        """
        self.add_list = add_list
        self.degree = degree

    def fit(self, X, y=None):
        if type(self.degree) == int:
            self.degree_in = [self.degree]
        else:
            self.degree_in = self.degree
        if self.add_list is None:
            temp_list = ()
            self.degree_in = [deg for deg in self.degree_in if deg >= 2 and deg <= len(get_numeric(X))]
            for deg in self.degree_in:
                temp_list += tuple(combinations(get_numeric(X), deg))
            self.add_list_in = temp_list
        elif type(self.add_list) == list:
            temp_list = ()
            self.degree_in = [deg for deg in self.degree_in if deg >= 2 and deg <= len(self.add_list)]
            for deg in self.degree_in:
                temp_list += tuple(combinations(self.add_list, deg))
            self.add_list_in = temp_list
        else:
            self.add_list_in = self.add_list
        return self

    def transform(self, X, y=None):
        X_temp = X.copy()
        for comb in self.add_list_in:
            temp = X_temp[comb[0]].copy()
            col = comb[0]
            for i in range(1, len(comb)):
                temp *= X_temp[comb[i]]
                col = col + '*' + comb[i]
            X_temp[col] = temp
        return X_temp


class DimensionReduction(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, method='pca', rows_to_scan='all'):
        """
        A class that can be inserted into pipeline.

        This will do dimension reduction on the data.
        Please only bring in the data with all numeric columns without missing values and inf values.

        Parameters:
        -------------
        method: str. default='pca'. Others can be 'lda'

        rows_to_scan: int or float or 'all', default='all'
            This is the number of rows to scan to train
            If int, only check the first rows_to_scan rows.
            If float, only check the first rows_to_scan fraction of all the rows.
            If 'all', check all the rows.

        n_components: int or float. default=2 (if 'all', then will not do any reduction)
        If method='lda', the max value is the number of classes minus 1
        If method='pca', n_components can be float. For example, if it's 0.95, then select the number of components such that the amount
            of variance that needs to be explained is greater than 0.95
        -------------

        returns a pandas dataframe whose shape is (n_sample, n_components)
        """
        self.n_components = n_components
        self.method = method
        self.rows_to_scan = rows_to_scan

    def fit(self, X, y=None):
        rows_to_scan_in = get_rows_to_scan(self.rows_to_scan, X.shape[0])
        if self.n_components != 'all':
            if self.method == 'pca':
                self.model = PCA(n_components=self.n_components)
            elif self.method == 'lda':
                self.model = LinearDiscriminantAnalysis(n_components=self.n_components)
            self.model.fit(X[:rows_to_scan_in], y[:rows_to_scan_in])
        return self

    def transform(self, X, y=None):
        if self.n_components == 'all':
            return X
        else:
            temp = self.model.transform(X)
            temp = pd.DataFrame(temp, index=X.index)
            col_names = ['comp_' + str(i) for i in range(temp.shape[1])]
            temp.columns = col_names
            return temp


class NumberGeneratorForModelPredictor():
    def __init__(self, random_state):
        """
        This class is used as a random number generator for ModelPredictor in grid search.

        Parameter:
        -----------
        random_state: int. Used to set the seed.
        -----------
        """
        self.random_state = random_state
        self.random_generator = RandomState(self.random_state)

    def rvs(self):
        return self.random_generator.uniform()

    def init_again(self):
        self.random_generator = RandomState(self.random_state)


class ModelPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, modelwithparams=None, random_number=-1, problem_type='infer'):
        """
        A class that can be inserted into pipeline.

        This will provide prediction or prediction of probability of each class by using a model with a certain setting of arguments.

        ----------
        Parameters:

        modelwithparams: a list with two entries. The first is a string denoting the model to use. The second is a dict,
            which is a parameter distribution of the related model. Default = None, which means a random forest classifier with default setting.
            For example, para_RF = {'n_estimators': [200], "max_depth": range(2,20),"max_features": np.arange(0.1,0.7,0.1)}
            modelwithparams = ['RF', para_RF].

        random_number: a integer which is from 1 to 9999, used to pick the specific parameter setting from the parameter distribution.
            When this class is used in a grid search, the random_number will be in the report of the grid search, and can be used to reproduce the parameter setting (by using the function get_para_from_dict or just pass the number into this class).

        problem_type: string. optional. default='infer', which means automatically determining the problem type.
            Others can be 'classification', 'regression'

        ----------

        returns a numpy array, with shape n_sample*n_classes

        model name list:

        for classification:

        'RF': random forest classifier
        'LR': logistic regression model
        'LDA': 'LinearDiscriminantAnalysis': linear discriminant analysis
        'QDA': 'QuadraticDiscriminantAnalysis': quadratic discriminant analysis
        'SVC': support vector machine (very expensive to compute when n_sample is large)
        'KNN': K nearest neighbor
        'GaussianNB': Gaussian Naive Bayes
        'MultiNB': Multinomial Naive Bayes
        'BagTree': Bagging Trees
        'GBC': Gradient Boosted Trees
        'Cali_linear_SVC': calibrated SVM with linear kernel
        'ExTree': Extremely randomized trees

        for regression:

        'RF': random forest regressor
        'LinearReg': ordinary linear regression
        'Ridge': ridge regression
        'Lasso': lasso regression
        'ElasticNet': Linear regression with combined L1 and L2 priors as regularizer
        'KNN': K nearest regressor
        'LassoLars': lasso regression useing lars
        'KernelRidge': ridge regression using kernel (likely to have memory error)
        'BagTree': Bagging Trees regressor
        'GBR': Gradient Boosted Trees regressor
        'LinearSVR': support vector machine regressor using linear kernel
        'ExTree': Extremely randomized trees regressor

        A simple example in pipeline and gridsearch:

        pipe = Pipeline([("null", eda.RemoveAllNull()),
                 ("cat", eda.ConvertCategorical(['x12', 'x13'])),
                 ("imp", eda.ImputeData(columns_must_impute='x4')),
                 ("modelpred", eda.ModelPredictor())])

        param_RF = {'n_estimators': [200], "max_depth": range(2,20), "max_features": np.arange(0.1,0.7,0.1)}
        param_LR = {'penalty': ['l1', 'l2'], 'C':np.logspace(-2,0,20)}

        rand_num = eda.NumberGeneratorForModelPredictor(422)

        params = {
            'cat__method': ['factorize', 'value_counts', 'group_means'],
            'imp__method': ['mean', 'max', 'median', 'ffill'],
            'modelpred__modelwithparams': [ ['RF', param_RF],['LR', param_LR]],
            'modelpred__random_number': rand_num
        }

        search = RandomizedSearchCV(pipe, param_distributions=paras, n_iter=5,
                                    random_state=42, cv=5,scoring='roc_auc', verbose=2)

        Note: please make lists as values in the dict of parameter distribution. (like n_estimators in para_RF in the example)
        Note: before running another search.fit, please run rand_num.init_again() to guarantee same random number generated.
        Note: random_states for grid search and for number generator are independent. The first controls the parameter other than the
              random number, and the random_state for number generator just controls the random number.

        """
        self.modelwithparams = modelwithparams
        self.oldpara = self.modelwithparams
        self.random_number = random_number
        self.flag = True
        self.problem_type = problem_type

    def fit(self, X, y):
        if self.problem_type == 'infer':
            prob_type = get_problem_type(y)
        else:
            prob_type = self.problem_type
        if self.oldpara is not None:
            if self.oldpara[0] != self.modelwithparams[0]:
                self.flag = True
                self.oldpara = self.modelwithparams
            elif self.oldpara[1] != self.modelwithparams[1]:
                self.flag = True
                self.oldpara = self.modelwithparams
        if self.flag:
            if self.modelwithparams == None:

                self.model_name = "RF"
                if prob_type == 'classification':
                    self.model = RandomForestClassifier()
                else:
                    self.model = RandomForestRegressor()
            else:
                if self.random_number == -1:
                    self.random_number = np.random.choice(range(1, 10000)) / 10000.0
                self.model_name = self.modelwithparams[0]
                para_dict = self.modelwithparams[1]
                self.para = get_params_from_dict(para_dict, self.random_number)
                if prob_type == 'classification':
                    if self.model_name == 'RF':
                        self.model = RandomForestClassifier(**self.para)
                    elif self.model_name == 'LR':
                        self.model = LogisticRegression(**self.para)
                    elif self.model_name == 'LDA':
                        self.model = LinearDiscriminantAnalysis(**self.para)
                    elif self.model_name == 'QDA':
                        self.model = QuadraticDiscriminantAnalysis(**self.para)
                    elif self.model_name == 'SVC':
                        self.model = SVC(**self.para)
                    elif self.model_name == 'KNN':
                        self.model = KNeighborsClassifier(**self.para)
                    elif self.model_name == 'GaussianNB':
                        self.model = GaussianNB(**self.para)
                    elif self.model_name == 'MultiNB':
                        self.model = MultinomialNB(**self.para)
                    elif self.model_name == 'BagTree':
                        self.model = BaggingClassifier(**self.para)
                    elif self.model_name == 'GBC':
                        self.model = GradientBoostingClassifier(**self.para)
                    elif self.model_name == 'Cali_linear_SVC':
                        self.model = CalibratedClassifierCV(LinearSVC(**self.para), 'isotonic', cv=5)
                    elif self.model_name == 'ExTree':
                        self.model = ExtraTreesClassifier(**self.para)
                else:
                    if self.model_name == 'RF':
                        self.model = RandomForestRegressor(**self.para)
                    elif self.model_name == 'LinearReg':
                        self.model = LinearRegression(**self.para)
                    elif self.model_name == 'Ridge':
                        self.model = Ridge(**self.para)
                    elif self.model_name == 'ElasticNet':
                        self.model = ElasticNet(**self.para)
                    elif self.model_name == 'Lasso':
                        self.model = Lasso(**self.para)
                    elif self.model_name == 'LassoLars':
                        self.model = LassoLars(**self.para)
                    elif self.model_name == 'KernelRidge':
                        self.model = KernelRidge(**self.para)
                    elif self.model_name == 'LinearSVR':
                        self.model = LinearSVR(**self.para)
                    elif self.model_name == 'KNN':
                        self.model = KNeighborsRegressor(**self.para)
                    elif self.model_name == 'BagTree':
                        self.model = BaggingRegressor(**self.para)
                    elif self.model_name == 'ExTree':
                        self.model = ExtraTreesRegressor(**self.para)
                    elif self.model_name == 'GBR':
                        self.model = GradientBoostingRegressor(**self.para)

            self.flag = False

        nullcol = get_columns_with_nulls(X, 'all', 'all')
        if len(nullcol) > 0:
            raise Exception('Columns ' + str(list(nullcol)) + ' have missing values!')
        self.model.fit(X, y)
        return self

    def predict_proba(self, X, y=None):
        nullcol = get_columns_with_nulls(X, 'all', 'all')
        if len(nullcol) > 0:
            raise Exception('Columns ' + str(list(nullcol)) + ' have missing values!')
        return self.model.predict_proba(X)

    def predict(self, X, y=None):
        nullcol = get_columns_with_nulls(X, 'all', 'all')
        if len(nullcol) > 0:
            raise Exception('Columns ' + str(list(nullcol)) + ' have missing values!')
        return self.model.predict(X)


def get_params_from_dict(param_dict, random_number):
    """
    A function used to reproduce the specific parameter setting.

    Parameter:
    ----------
    param_dict:  a dict of parameter distribution

    random_number: a number from 0 to 1 used to pick the specific parameter setting
    ----------

    returns a dict of a parameter setting


    """
    temp = list(product(*param_dict.values()))
    param_choose = temp[int(random_number * len(temp))]
    param = {}
    for i in range(len(param_choose)):
        param[list(param_dict.keys())[i]] = param_choose[i]
    return param


class RemoveAllNull(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_check='all'):
        """
        A class that can be inserted into pipeline.
        This will remove the columns with all missing values in the columns_to_check

        Parameter:
        ----------
        columns_to_check: string or list, optional (default="all")
            If "all", check all columns. Other
            strings are interpreted to represent column names and you can
            pass a list of column names to check.
        -----------

        returns a pandas dataframe
        """
        self.columns_to_check = columns_to_check

    def fit(self, X, y=None):
        self.columns_to_remove = get_columns_with_all_nulls(X, self.columns_to_check)
        return self

    def transform(self, X, y=None):
        X_temp = X.copy()
        for col in X_temp.columns:
            if col in self.columns_to_remove:
                X_temp.pop(col)
        return X_temp


def report_grid_score(grid_scores, n_top=3):
    """ Report best scores from grid search """
    top_scores = sorted(grid_scores, key=lambda x: x[1], reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def report_grid_score_with_params(grid_scores, n_top=3, model_key=None, random_number_key=None):
    """ Report best scores from grid search
        Also display the specific parameter setting using the random_number(when we use ModelPredictor)

        Parameters:
        -----------
        model_key: string. The name of the modelwithpara argument for the class ModelPredictor in grid search parameter distribution.

        random_number_key: string. The name of the random_number argument for the class ModelPredictor in grid search parameter distribution.
        -----------

        A simple example:
        br.report_grid_score_with_para(search.grid_scores_, n_top=10, model_key='modelpred__modelwithparams', random_number_key='modelpred__random_number')

    """
    top_scores = sorted(grid_scores, key=lambda x: x[1], reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("Parameters of the model: ",
              get_params_from_dict(score.parameters[model_key][1], score.parameters[random_number_key]))
        print("\n")


def report_grid_score_detail(random_search, charts=True):
    """Input fit grid search estimator. Returns df of scores with details"""
    df_list = []

    for line in random_search.grid_scores_:
        results_dict = dict(line.parameters)
        results_dict["score"] = line.mean_validation_score
        results_dict["std"] = line.cv_validation_scores.std()
        df_list.append(results_dict)

    result_df = pd.DataFrame(df_list)
    result_df = result_df.sort_values(by="score", ascending=False)

    if charts:
        for col in get_numeric(result_df):
            if col not in ["score", "std"]:
                plt.scatter(result_df[col], result_df.score)
                plt.title(col)
                plt.show()

        for col in get_categorical(result_df):
            cat_plot = result_df.score.groupby(result_df[col]).mean()
            cat_plot.sort()
            cat_plot.plot(kind="barh", xlim=(.5, None), figsize=(7, cat_plot.shape[0] / 2))
            plt.show()
    return result_df


def cross_val_pred_both(model, X_train, y_train, X_test, cv=5, n_class=2, problem_type='infer'):
    """
    get crossvalidated predictions for both training and testing data, using the model
    The model should be set with proper parameters before.

    Parameters
    ----------
    model: the model used to fit and predict. Should be set with proper parameters.

    X_train: training data

    y_train: true label of training data

    X_test: testing data

    cv: number of folds to do cross validation. default=5

    n_class: number of classes. default=2

    problem_type: string. optional. default='infer', which means automatically determining the problem type.
            Others can be 'classification', 'regression'

    ----------

    When cv=1, just fit the whole training data and predict the test data.
    We can specify the problem_type as 'regression' when we want to predict labels in classification problems.

    The prediction for training data is from cross validation
    The prediction for testing data is the average of the models fitted by different folds of training data

    returns two numpy arrays. Shape is (n_sample, n_class). first is for training data, second is for testing data
    if problem_type is 'regression', will just return two numpy arrays with shape (n_sample)

    """
    if problem_type == 'infer':
        problem_type = get_problem_type(y_train)
    if problem_type == 'classification':
        pred_train = np.zeros((len(y_train), n_class))
        pred_test = np.zeros((len(X_test), n_class))
    else:
        pred_train = np.zeros(len(y_train))
        pred_test = np.zeros(len(X_test))

    if cv > 1:
        kfold = KFold(len(X_train), n_folds=cv)

        if problem_type == 'classification':
            for train_index, test_index in kfold:
                model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
                pred_train[test_index] = model.predict_proba(X_train.iloc[test_index])
                pred_test = pred_test + model.predict_proba(X_test)
        else:
            for train_index, test_index in kfold:
                model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
                pred_train[test_index] = model.predict(X_train.iloc[test_index])
                pred_test = pred_test + model.predict(X_test)

        pred_test = pred_test / float(cv)
    elif cv == 1:
        if problem_type == 'classification':
            model.fit(X_train, y_train)
            pred_train = model.predict_proba(X_train)
            pred_test = model.predict_proba(X_test)
        else:
            model.fit(X_train, y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
    return pred_train, pred_test


def get_prediction_from_models(grid_scores, model, model_pick_list, X_train, y_train, X_test,
                               cv=5, n_class=2, print_process=False, problem_type='infer'):
    """
    get crossvalidated predictions for both training and testing data, using the selected models(also can be a pipeline) in grid search.

    Parameters
    -----------
    grid_scores: grid_scores_ attribute from grid search

    model: a new model which is the same as the one used to do grid search. DO NOT set parameters of the parameter distribution for grid search.

    model_pick_list: a list of integers. The integers stand for the number of rank in the report of grid search

    X_train: training data

    y_train: true label of training data

    X_test: testing data

    cv: number of folds to do cross validation. default=5

    n_class: number of classes. default=2

    problem_type: string. optional. default='infer', which means automatically determining the problem type.
        Others can be 'classification', 'regression'

    -----------

    The prediction for training data is from cross validation
    The prediction for testing data is the average of the models fitted by different folds of training data
    Will automatically remove the prediction for class 0.
    The column name of prediction for model 2 class 3 is 'prob_model2_class3'

    returns two pandas dataframes, first is for training data, second is for testing data


    """
    if problem_type == 'infer':
        problem_type = get_problem_type(y_train)
    pred_train = pd.DataFrame(index=X_train.index)
    pred_test = pd.DataFrame(index=X_test.index)
    top_scores = sorted(grid_scores, key=lambda x: x[1], reverse=True)
    number = 0
    total = len(model_pick_list)
    for rank_no in model_pick_list:
        number += 1
        if print_process:
            print('Processing Model ' + str(number) + ' of ' + str(total))
        para = top_scores[(rank_no - 1)].parameters
        model.set_params(**para)
        prob_train, prob_test = cross_val_pred_both(model, X_train, y_train, X_test, cv, n_class, problem_type)
        if problem_type == 'classification':
            for i in range(1, n_class):
                pred_train['model' + str(rank_no) + '_class' + str(i)] = prob_train[:, i]
                pred_test['model' + str(rank_no) + '_class' + str(i)] = prob_test[:, i]
        else:
            pred_train['model' + str(rank_no)] = prob_train
            pred_test['model' + str(rank_no)] = prob_test
    return pred_train, pred_test


def get_problem_type(y):
    """
    returns a string denoting the type of problem

    Parameter:
    y: pandas series
    """
    if len(y.unique()) > 50:
        return 'regression'
    else:
        return 'classification'


class TopFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, max_features='all', method="RandomForest", problem_type="infer", rows_to_scan='all'):
        """
        this can be inserted into pipeline in scikit-learn
        it will retain several most important features of data using different methods
        The order of columns will be the descending order of importance

        Parameters:
        ----------------
        max_features: float or int or  'all', default = 'all'
            If float , then this percent of total columns will be kept
            If int , then this number of total columns will be kept
            If 'all', return the original data

        method: string or skearn model with feature_importances_ attribute.
            default = 'RandomForest'
            can be 'RandomForest' or 'ExtraTree' with smart settings
            (or You can also specify your own model)

        problem_type: string. default='infer', which means automatically determine the problem type
            Valid options: "regression", "classification", "infer"

        rows_to_scan: int or float or 'all', default='all'
            This is the number of rows to scan to train
            If int, only check the first rows_to_scan rows.
            If float, only check the first rows_to_scan fraction of all the rows.
            If 'all', check all the rows.
        ---------------
        returns a pandas dataframe

        """
        self.method = method
        self.problem_type = problem_type
        self.max_features = max_features
        self.rows_to_scan = rows_to_scan

    def fit(self, X, y):
        if self.max_features != 'all':
            rows_to_scan_in = get_rows_to_scan(self.rows_to_scan, X.shape[0])
            if type(self.max_features) == float:
                self.max_features_in = int(self.max_features * X.shape[1]) + 1
            else:
                self.max_features_in = self.max_features
            if self.problem_type == "infer":
                prob_type = get_problem_type(y)
            else:
                prob_type = self.problem_type
            if prob_type == "regression":
                if self.method == "RandomForest":
                    self.model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=2, bootstrap=False,
                                                       n_jobs=-1)
                elif self.method == "ExtraTree":
                    self.model = ExtraTreesRegressor(n_estimators=1000, max_features=3, max_depth=3, bootstrap=False,
                                                     n_jobs=-1)
                else:
                    self.model = self.method
            else:
                if self.method == "RandomForest":
                    self.model = RandomForestClassifier(n_estimators=1000, max_features=3, max_depth=2, bootstrap=False,
                                                        n_jobs=-1)
                elif self.method == "ExtraTree":
                    self.model = ExtraTreesClassifier(n_estimators=1000, max_features=3, max_depth=3, bootstrap=False,
                                                      n_jobs=-1)
                else:
                    self.model = self.method
            self.model.fit(X[:rows_to_scan_in], y[:rows_to_scan_in])
            results = pd.Series(self.model.feature_importances_, X.columns).sort_values(ascending=False)
            self.important_features = results.index[:self.max_features_in]
        return self

    def transform(self, X, y=None):
        if self.max_features != 'all':
            return X[self.important_features]
        else:
            return X


class PermutationImportance():
    def __init__(self, method="RandomForest", problem_type="infer", rows_to_scan="all"):
        """
        Trains a model. Records score of model on out of sample data. For each feature,
        permute feature and predict out of sample data. Record new score. Collect all
        scores. Features where a permutation results in the largest decrease in accuracy
        are best.

        This class can be used in TopFeatures

        Parameters
        -------------
        method: string or sklearn model with feature_importance_ attribute
            Use "RandomForest" or "ExtraTrees" for smart defaults or pass your own
            estimator (having predict function for regression or predict_proba for classification)
        problem_type: string. default='infer', which means automatically determine the problem type
            Valid options: "regression", "classification", "infer"
        rows_to_scan: int or float or 'all', default='all'
            This is the number of rows to scan to train
            If int, only check the first rows_to_scan rows.
            If float, only check the first rows_to_scan fraction of all the rows.
            If 'all', check all the rows.
        -------------

        attribute: feature_importances_ , showing the importance of each columns, the larger values mean more importances

        """
        self.feature_importances_ = None
        self.method = method
        self.problem_type = problem_type
        self.rows_to_scan = rows_to_scan

    def fit(self, X, y):
        rows_to_scan_in = get_rows_to_scan(self.rows_to_scan, X.shape[0])
        if self.problem_type == "infer":
            prob_type = get_problem_type(y)
        else:
            prob_type = self.problem_type
        if prob_type == "regression":
            if self.method == "RandomForest":
                self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
            elif self.method == "ExtraTree":
                self.model = ExtraTreesRegressor(n_estimators=100, n_jobs=-1)
            else:
                self.model = self.method
        else:
            if self.method == "RandomForest":
                self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            elif self.method == "ExtraTree":
                self.model = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
            else:
                self.model = self.method
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X[:rows_to_scan_in], y[:rows_to_scan_in], test_size=.25)
        self.model.fit(X_train, y_train)
        temp_importance = []
        if self.problem_type == "regression":
            from sklearn.metrics import mean_squared_error
            base_score = mean_squared_error(y_test, self.model.predict(X_test))

            for col in X.columns:
                X_test_permuted = X_test.copy()
                temp = X_test_permuted[col]
                temp = temp.reindex(np.random.permutation(temp.index))
                temp.index = X_test_permuted.index
                X_test_permuted.loc[:, col] = temp
                temp_score = mean_squared_error(y_test, self.model.predict(X_test_permuted)) / base_score
                temp_importance.append(temp_score)
        else:
            from sklearn.metrics import roc_auc_score
            base_score = roc_auc_score(pd.get_dummies(y_test), self.model.predict_proba(X_test))

            for col in X.columns:
                X_test_permuted = X_test.copy()
                temp = X_test_permuted[col]
                temp = temp.reindex(np.random.permutation(temp.index))
                temp.index = X_test_permuted.index
                X_test_permuted.loc[:, col] = temp
                temp_score = base_score / roc_auc_score(pd.get_dummies(y_test),
                                                        self.model.predict_proba(X_test_permuted))
                temp_importance.append(temp_score)

        self.feature_importances_ = temp_importance
        return self


class SimulPermutationImportance():
    def __init__(self, method="RandomForest", n_random_feature_ratio=5, problem_type="infer", rows_to_scan="all"):
        """
        For each variable, copy variable n_random_feature_ratio times. Permute
        each copy. Train model. For all permuted copies, calculate average importance and
        standard deviation of importance. Compare importance of actual variable (not permuted)
        with the mean of the permuted versions. Test for statistical significance. Rank variables
        based on the highest significance.

        This class can be used in TopFeatures

        Parameters
        -------------
        method: string or sklearn model with feature_importance_ attribute
            Use "RandomForest" or "ExtraTrees" for smart defaults or pass your own
            estimator that has a feature_importances_ attribute
        problem_type: string. default='infer', which means automatically determine the problem type
            Valid options: "regression", "classification", "infer"
        rows_to_scan: int or float or 'all', default='all'
            This is the number of rows to scan to train
            If int, only check the first rows_to_scan rows.
            If float, only check the first rows_to_scan fraction of all the rows.
            If 'all', check all the rows.
        n_random_feature_ratio: int. default=5.
        -------------

        attribute: feature_importances_ , showing the importance of each columns, the larger values mean more importances

        """
        self.feature_importances_ = None
        self.method = method
        self.problem_type = problem_type
        self.rows_to_scan = rows_to_scan
        self.n_random_feature_ratio = n_random_feature_ratio

    def fit(self, X, y):
        rows_to_scan_in = get_rows_to_scan(self.rows_to_scan, X.shape[0])
        if self.problem_type == "infer":
            prob_type = get_problem_type(y)
        else:
            prob_type = self.problem_type
        if prob_type == "regression":
            if self.method == "RandomForest":
                self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
            elif self.method == "ExtraTree":
                self.model = ExtraTreesRegressor(n_estimators=100, n_jobs=-1)
            else:
                self.model = self.method
        else:
            if self.method == "RandomForest":
                self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            elif self.method == "ExtraTree":
                self.model = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
            else:
                self.model = self.method
        sampleX = X[:rows_to_scan_in].copy()
        sampley = y[:rows_to_scan_in].copy()
        for col in X.columns:
            for i in range(self.n_random_feature_ratio):
                temp = sampleX[col]
                temp = temp.reindex(np.random.permutation(temp.index))
                temp.index = sampleX.index
                sampleX.loc[:, col + '_pemu_' + str(i)] = temp

        self.model.fit(sampleX, sampley)
        results = pd.Series(self.model.feature_importances_, sampleX.columns)
        temp_importance = []
        for col in X.columns:
            temp_a = results.loc[col]
            temp_b = results.loc[[col + '_pemu_' + str(i) for i in range(self.n_random_feature_ratio)]]
            temp_z_score = (temp_a - temp_b.mean()) / temp_b.std()
            from scipy.stats import norm
            temp_p_value = norm.sf(abs(temp_z_score))
            temp_importance.append(1 - temp_p_value)

        self.feature_importances_ = temp_importance
        return self


class ConvertToDict(BaseEstimator, TransformerMixin):
    # class to convert a dollar amount to a float
    def __init__(self, columns_to_fix=[], convert_dict={'Y': 1, 'N': 0}):
        """
        A class that can be inserted into a pipeline

        This will convert the list of columns that are input to this class

        Parameters
        ----------
        X: Pandas dataframe

        columns_to_fix: a list of columns to convert in the input DF.
        convert_dict: a dictionary of how to convert the input DF; default is Y==1,N==0

        returns a pandas dataframe
        """
        self.columns_to_fix = columns_to_fix
        self.convert_dict = convert_dict

    def fit(self, X, y=None):
        # my_convert_dict = {'Y':1,'N': 0}
        if type(self.columns_to_fix) == 'str':
            self.columns_to_fix = [self.columns_to_fix]

        else:
            self.columns_to_fix = self.columns_to_fix
        self.convert_dict = self.convert_dict
        return self

    def transform(self, X, y=None):
        X_temp = X.copy()

        X_temp[self.columns_to_fix] = X[self.columns_to_fix].replace(self.convert_dict)

        return X_temp


# class to drop a list of columns
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=[]):
        """
        A class that can be inserted into a pipeline

        This will drop columns that are listed as input to this class

        Parameters
        ----------
        X: Pandas dataframe


        columns_to_drop: a list of columns to remove from the input DF.

        returns a pandas dataframe

        TODO make this so that it could be based upon the pct missing from a training dataset; pct_missing_thresh because a parameter
        """
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        self.columns_to_drop = self.columns_to_drop
        return self

    def transform(self, X, y=None):
        X_temp = X.copy()
        for col in X_temp.columns:
            if col in self.columns_to_drop:
                X_temp.pop(col)

        return X_temp


class DummyEncodeColumn(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_fix=[], rows_to_scan='all', keep_dummies=False):
        """
        A class that can be inserted into a pipeline

        This will replace a given list of columns with label_encoded values;

        Parameters
        ----------
        X: Pandas dataframe

        keep_dummies: boolean (default = False)
            If False then the original dummied column and one of the one-hot-encoded (dummified) columns will be dropped.

        Note that new categories/values in test data will be set as -1.

        returns a pandas dataframe
        """
        self.columns_to_fix = columns_to_fix
        self.rows_to_scan = rows_to_scan
        self.keep_dummies = keep_dummies

    def fit(self, X, y=None):
        # self.map_values = {}
        self.dummy_values = {}
        # self.na_values={}

        rows_to_scan_in = get_rows_to_scan(self.rows_to_scan, X.shape[0])
        # self.columns_to_fix_in = get_list_of_columns_to_check(self.columns_to_fix, X.columns)
        X_temp = X[:rows_to_scan_in].copy()
        # apply labelEncoder to each column in this list
        for col in self.columns_to_fix:
            self.dummy_values[col] = X_temp[col].unique()
        return self

    def transform(self, X, y=None):
        X_temp = X.copy()
        original_cols = list(X_temp.columns)
        """
        for col in self.columns_to_fix:
            tmp = pd.get_dummies(X[col],prefix=str(col))
            X_temp = pd.concat([X_temp,tmp],axis=1)
            # now drop the columns that I don't want if so flagged
            if not self.keep_dummies:
                X_temp.drop([col,tmp.columns[-1]],axis=1,inplace=True)
         """
        for col in self.columns_to_fix:
            for cat in self.dummy_values[col]:
                cat_col = str(col) + '_' + str(cat)
                if str(cat) == 'nan':
                    X_temp[cat_col] = X_temp[col].isnull().astype(int)
                else:
                    X_temp[cat_col] = (X_temp[col] == cat).astype(int)
                # append the new column name
                original_cols.append(cat_col)
            # if keep_dummies is false remove the original column name and the last category
            if not self.keep_dummies:
                original_cols.remove(str(col))
                original_cols.pop()  # note pop removes the last element from a list
        X_temp = X_temp[original_cols]

        return X_temp


class LabelEncodeColumn(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_fix=[], rows_to_scan='all'):
        """
        A class that can be inserted into a pipeline

        This will replace a given list of columns with label_encoded values;

        Parameters
        ----------
        X: Pandas dataframe


        Note that new categories/values in test data will be set as -1.

        returns a pandas dataframe
        """
        self.columns_to_fix = columns_to_fix
        self.rows_to_scan = rows_to_scan

    def fit(self, X, y=None):
        self.map_values = {}
        self.dummy_values = {}
        self.na_values = {}

        rows_to_scan_in = get_rows_to_scan(self.rows_to_scan, X.shape[0])
        # self.columns_to_fix_in = get_list_of_columns_to_check(self.columns_to_fix, X.columns)
        X_temp = X[:rows_to_scan_in].copy()
        # apply labelEncoder to each column in this list
        for col in self.columns_to_fix:
            map_values = X_temp[col].unique()
            map_values.sort()
            self.map_values[col] = {key: index for index, key in enumerate(map_values)}
        return self

    def transform(self, X, y=None):
        X_temp = X.copy()
        original_cols = list(X_temp.columns)
        for col in self.columns_to_fix:
            X_temp[str(col) + "_le"] = X_temp[col].map(self.map_values[col], "ignore")
            X_temp[str(col) + "_le"] = X_temp[str(col) + "_le"].fillna(-1)
            # fill all remaining null values with -1
            # remove the original column name and add the new column name
            original_cols.remove(str(col))
            original_cols.append(str(col + "_le"))
        X_temp = X_temp[original_cols]

        return X_temp


def summarize_dataframe(df, columns_to_check='all', show_progress=False, arity_thresh=20):
    """
    Function used to summarize properties of a data frame (alternative to get_initial_analysis or describe_categorical
    :param df: input data frame
    :param columns_to_check: list of columns to do the summary for
    :param show_progress: outputs progress to the screen
    :param arity_thresh: max arity size where values are displayed to accepted values column
    :return:
    """
    if columns_to_check != 'all':
        df = df[columns_to_check]

    nrow = len(df)
    summary_df = pd.DataFrame(columns=['feature', 'datatype', 'nmissing', 'arity', 'accepted values'])
    len_df = len(summary_df)
    for col in df.columns:
        nmiss = nrow - df[col].value_counts().sum()
        narity = len(df[col].unique())
        if show_progress:
            # print(col, df[col].dtype,nmiss, "\t", narity,":\t", df[col].ix[8320])
            # else:
            print(col, df[col].dtype, nmiss, "\t", narity)
        accept_val = None
        if narity < arity_thresh:
            accept_val = df[col].unique()
        else:
            accept_val = 'Too many to show'
        summary_df.loc[len_df] = [col, df[col].dtype, nmiss, narity, accept_val]
        len_df += 1
    # assing fraction of missing
    summary_df['x_missing'] = summary_df['nmissing'] / float(nrow)

    return summary_df


class FillMissingValue(BaseEstimator, TransformerMixin):
    """
        A class that can be inserted into a pipeline

        This will replace missing values a given list of columns with specified fill value method;

        Parameters
        ----------
        X: Pandas dataframe

        returns a pandas dataframe
    """

    def __init__(self, fill_value=0, columns_to_fix='all', rows_to_scan='all', keep_dummies=False, fill_inf=True):

        self.fill_value = fill_value
        self.columns_to_fix = columns_to_fix
        self.rows_to_scan = rows_to_scan
        self.keep_dummies = keep_dummies
        self.fill_inf = fill_inf
        self.values = {}

    def fit(self, X, y=None):

        X_temp = X.copy()
        if self.columns_to_fix == 'all':
            X_temp = X_temp.replace(np.inf, np.nan)
            self.columns_to_fix_in = get_numeric(X_temp)
        elif self.columns_to_fix == 'auto':
            X_temp = X_temp.replace(np.inf, np.nan)
            self.columns_to_fix_in = get_columns_with_nulls(X_temp, 'all', 'all')
        else:
            self.columns_to_fix_in = get_list_of_columns_to_check(self.columns_to_fix, X.columns)
            self.columns_to_fix_in = [col for col in self.columns_to_fix_in if col in get_numeric(X_temp)]
            X_temp[self.columns_to_fix_in] = X_temp[self.columns_to_fix_in].replace(np.inf, np.nan)

        self.rows_to_scan_in = get_rows_to_scan(self.rows_to_scan, X.shape[0])

        # check if the fill_values are a list
        if type(self.fill_value) == list:
            self.values = dict(zip(self.columns_to_fix_in, self.fill_value))
        elif type(self.fill_value) == dict:
            self.values = self.fill_value
        else:
            for c in self.columns_to_fix_in:
                self.values[c] = self.fill_value

        return self

    def transform(self, X, y=None):
        X_temp = X.copy()

        if self.fill_inf:
            X_temp[self.columns_to_fix_in] = X_temp[self.columns_to_fix_in].replace(np.inf, np.nan)
        if self.keep_dummies:
            temp = pd.DataFrame(index=X_temp.index)
            for col in self.columns_to_fix_in:
                temp[(col + '_d')] = X_temp[col].isnull().astype("int")
            X_temp = pd.concat([X_temp, temp], axis=1)
        original_cols = list(X_temp.columns)
        X_temp = X_temp.fillna(self.values)
        X_temp = X_temp[original_cols]

        return X_temp


def parse_dollars_to_float(x):
    """
    function to convert dollar amounts (recorded with a leading $ sign and intervening commas) into a float.
    :param x: input string
    :return: y --> float
    """
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    try:
        if x.startswith('('):
            # print(x)
            y = -1.0 * locale.atof(x[2:-1])
        # print(x,y)
        elif x.startswith('$'):
            y = locale.atof(x[1:])
    except AttributeError:
        y = np.nan
    return y


class dollarsToFloat(BaseEstimator, TransformerMixin):
    # class to convert a dollar amount to a float
    def __init__(self, columns_to_fix=[]):
        """
        A class that can be inserted into a pipeline

        This will convert the list of columns that are input to this class

        Parameters
        ----------
        X: Pandas dataframe


        columns_to_fix: a list of columns to convert in the input DF.

        returns a pandas dataframe

        TODO: make this a generic ConvertByFunction() class that takes a defined function as an input option

        """
        self.columns_to_fix = columns_to_fix

    def fit(self, X, y=None):
        self.columns_to_fix = self.columns_to_fix
        return self

    def transform(self, X, y=None):
        X[self.columns_to_fix] = X[self.columns_to_fix].apply(lambda x: parse_dollars_to_float(x))

        return X
