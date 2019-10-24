# Data Acquisition, Analysis, & Transformation

# Resources
- https://help.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository
- http://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
- https://seaborn.pydata.org/examples/index.html
- https://www.w3schools.com/python/python_regex.asp

"""With some Python basics we will starting to combine existing packages to acquire, analysis, and transform data."""

# Data Acquisition

import pandas as pd
import wikipedia as wp


# https://qxf2.com/blog/web-scraping-using-python/
page_title = "Opinion polling for the 2019 Canadian federal election"

#Get the html source
html = wp.page(page_title).html().encode("UTF-8")

# Extract tables and convert the html tables into pd.DataFrames()
df = pd.read_html(html)[0]


# Inspect the data
df.head()

# We notice that there seems to be a double header
df.columns

# What is the type of columns
type(df.columns)

# Let's use a loop to extract and edit each element of this Mu
columnn_names = []
for c in df.columns:
    tmp = c[0].lower()
    columnn_names.append(tmp.replace(" ", "_"))

columnn_names

# Let's use regular expressions and list comprehension this time
import re
regex = "[a-z]+"
columnn_names = ["_".join(re.findall(regex, i)) for i in columnn_names]

# Let's edit the columns of our dataset
df.columns = columnn_names
df.head()

# Let's further rename those columns
names_dict = {
    "polling_firm": "source",
    "last_dateof_polling": "date",
    "samplesize": "sample_size",
    "marginof_error": "error",
    "cons": "cpc",
    "liberal": "lpc",
    "green": "gpc",
    "polling_method": "method",
}

type(names_dict)

# Pass the new dictionary as an argument to the .rename method
df.rename(columns=names_dict, inplace=True)
df.head()

# Let's check the data types
df.dtypes

# The date field needs to be converted
df[['date']] = pd.to_datetime(df.date)
df.head()

# We should also only keep the numeric values for the margins of error
regex = "(\d+\.*\d*)"
df.error = df.error.str.extract(regex)

# Let's look again at our dataset
df.head()

# What if we look at a random subsample
df.sample(5)

# Let's clean the sample
regex = r"\(.*\)"
df.sample_size = df.sample_size.str.replace(regex, "")
df.sample_size = df.sample_size.str.replace(" |,", "")

# How does the data look now?
df.sample(5)

# What about the data types?
df.info()

# Which of these variables are still objects?
df.select_dtypes(include='object')

# Let's use a dictionary to recode the data types
convert_dict = {
    'error': float,
    'sample_size': int,
    'lead': float
}

df = df.astype(convert_dict)

# Let's look once again at our data
df.sample(10)

# What are the remaining objects?
df.select_dtypes(include='object')

# Keep only necessary variables
to_keep = [
    'source',
    'date',
    'lpc',
    'cpc',
    'ndp',
    'bq',
    'gpc',
    'ppc',
    'method'
]

df = df[to_keep]


# Save the dataframe to a file
file_name = "national_polls_2019.csv"
df.to_csv(file_name, index=False)
print(df)

df.dtypes

# Read the data back-in from the recorded csv file.

# More info on read_csv
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
df = pd.read_csv("national_polls_2019.csv", parse_dates=['date'])
df.dtypes


## ---- Time Series Operations

# Let's transform this into a time-series
df.set_index('date', inplace=True)

# Time-series data should be stored in descending order
df = df.sort_values(by=['date', 'source'])

# How does the data look now?
df.head()

# What about the tail?
df.tail()

# A time indexed data frame provides a lot of control over the data
df.loc[df.index > '2019-10-15']

# We can look at a single party
df.lpc.loc['2019-10-20']

# We can focus on a subset of columns
parties = ["lpc", "cpc", "ndp", "bq", "gpc", "ppc"]
df.loc[:, parties]

# We can aggregate/resample the data
df[parties].resample('D', how='mean')

# We can also use pandas to plot
df[parties].resample('D', how='mean').plot()

# We can look at the distributions for each party
df[parties].plot(kind='kde')

# Or do a simple box-plot
df[parties].boxplot()

# Let's look at missing values
df.isnull().mean()

# We can remove missing values
df.dropna()

# We just lost half of our dataset...
# Maybe we should fill the missing values
tmp_df = df.fillna(method='ffill', limit=3).copy()
tmp_df.isnull().mean()

df = tmp_df

# Let's investigate which polling firms have released the most polls
df.source.value_counts()

# Let's remove the firms that released less than 5 polls
tmp_mask = df.source.value_counts() >= 5
mask = tmp_mask.index[tmp_mask]

df = df[df.source.isin(mask)]

# Once again we could decide to visualize directly the result
df.source.value_counts().plot(kind='barh')

# Let's try to do grouped operations and see how did each of these firms portrayed the liberal party

df.groupby('source').lpc.describe().sort_values(by='mean')

# We can also look at the means for all the parties
df.groupby('source')[parties].mean().sort_values('lpc')

# We can also apply custom functions by groups
z_score = lambda x: (x-x.mean()) / x.std()
df.reset_index().groupby('source')[parties].apply(z_score).head()

# Most algorithms need you to reshape the date to a long format
long_df = pd.melt(
    df.reset_index(),
    id_vars=['date', 'source'],
    value_vars=parties,
    var_name='party',
    value_name='share',
)

long_df.head()

# Seaborn, a statistical data visualization library uses long-format
import seaborn as sns
sns.set(style="whitegrid", palette="muted")

sns.swarmplot(
    x="party",
    y="share",
    hue="source",
    data=long_df,
)

# Let's say we want to add the sample size back?
new_df = long_df.merge(
    df[['method', 'source']].reset_index(),
    on=['date', 'source']
)

new_df.head()

# We can also expand the dataframe back to a wide format
new_df = new_df.pivot_table(
    index=['date', 'source', 'method'],
    columns='party',
    values='share',
)

new_df.head()

# Data Mdeling & Sharing
## sci-kit learn
# Let's start modeling and try to forecast the election.
# We will rely on data from the 2015 election to train a model.

title_train = "Opinion polling for the 2015 Canadian federal election"
html = wp.page(title_train).html().encode("UTF-8")
df_train = pd.read_html(html)[0]

# Cleaning the training set.
import re

# A function to fix the column names
def fix_names(input_df, names_dict):
    """Renames the columns in the input dataframe."""
    regex = "[a-z]+"

    columnn_names = []

    tmp_df = input_df.copy()

    for c in tmp_df.columns:
        tmp = c.lower()
        columnn_names.append(tmp.replace(" ", "_"))

    tmp_names = ["_".join(re.findall(regex, i)) for i in columnn_names]
    tmp_df.columns = tmp_names

    return tmp_df.rename(columns=names_dict)

# Let's edit them...
df_train = fix_names(df_train, names_dict)
df_train.columns

# Let's keep relevant variables only
df_train = df_train[to_keep]

# Remember lists also have useful methods
to_keep.remove('ppc')

# What does the training set look like ?
df_train = df_train[to_keep]
df_train.head()

# Let's store and remove the election results
results_2015 = df_train.iloc[1]
df_train = df_train.drop(1).dropna()

# Let's deal with missing values
df_train.dropna(inplace=True)

# What about the data types?
df_train.select_dtypes(include='object')

# Let's fix that date variable
df_train['date'] = pd.to_datetime(df_train.date)
df_train.sample(3)

# As we mentioned, most algorithms require the data to be in long-format
df_train = pd.melt(
    df_train.reset_index(),
    id_vars=['date', 'source', 'method'],
    value_vars=parties.remove('ppc'),
    var_name='party',
    value_name='share',
)

df_train.head()

# Let's do some more exploration and see if polls actually improve as we get closer to the election day?

# We need to merge the outcome of the election back
targets = (
    results_2015
    .transpose()
    .iloc[2:-1]
    .reset_index()
)

targets.columns = ['party', 'outcome']
targets['outcome'] = targets.outcome.astype('float')

df_train = df_train.merge(targets)
df_train.head()

# Does time have an impact on the error of pollsters?
df_train['error'] = abs(df_train.share - df_train.outcome)
df_train.set_index('date', inplace=True)
df_train.error.resample('D').mean().plot()

# What about the data collection method?
df_train.method.value_counts()

# Let's use some regex to do an initial cleaning
regex = r"\(.*\)|/| |rolling"
df_train['method'] = df_train.method.str.replace(regex, "")
df_train['method'].value_counts()

# Let's groups these even further
df_train['method'] = df_train.method.str.lower().str[:3]
df_train['method'].value_counts()

# Let's use seaborn this time as we now have a long-dataset and see see if there is an abservable difference between the data collection methods
sns.violinplot(x="method", y="error",
               split=True, inner="quart",
               data=df_train)
g.despine(left=True)

# Now that we have some intuition about 2015
# Let's start modeling

# We need to prepare our test set and verify it has the same form as the train set.
df_test = new_df.stack()
df_test.name = 'share'
df_test = df_test.reset_index().set_index('date')

data_2019 = {
    "party": ["lpc", "cpc", "bq", "ndp", "gpc"],
    "outcome": [33.1,34.4, 7.7, 15.9, 6.5],
}

df_test = df_test.reset_index().merge(pd.DataFrame(data_2019)).set_index('date')
df_test['error'] = abs(df_test.share - df_test.outcome)
all(df_test.columns == df_train.columns)

# Let's create a function to clean the method string!
def str_magic(input_series):
    regex = r"\(.*\)|/| |rolling"
    tmp = input_series.copy()
    tmp = df_test['method'].copy()
    tmp = tmp.str.replace(regex, "")
    return tmp.str.lower().str[:3]

df_test['method'] = str_magic(df_test['method'])

# Now that we have our train and test sets let's train our models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle

# We need to prepare our features

election_day_2015 = "2015-10-19"
election_day_2019 = "2019-10-21"

def add_days(df, election_day):
    test = pd.to_datetime(election_day) - df.reset_index()['date']
    test.index = df.index
    df['days'] = test.dt.days
    return df

def add_ohe_methods

df_train = add_days(df_train, election_day_2015)
df_test = add_days(df_test, election_day_2019)

# One-Hot Encoding
# Let's remove the group with most counts
df_train.method.value_counts().plot(kind='barh')

# Let's drop the most common value
train_dummies = pd.get_dummies(df_train['method'])
train_dummies.pop('tel')
df_train = pd.concat([df_train, train_dummies], axis=1)

test_dummies = pd.get_dummies(df_test['method'])
test_dummies.pop('tel')
df_test = pd.concat([df_test, test_dummies], axis=1)

y_var = 'outcome'
X_vars = ['share', 'days', 'ivr', 'onl']

predictions = []

models = [
    LinearRegression(),
    RandomForestRegressor(),
]

# Fit, predict, and save your models
for i in range(2):
    models[i].fit(df_train[X_vars], df_train[y_var])
    predictions.append(models[i].predict(df_test[X_vars]))
    pickle.dump(models[i], open(f"model_{i}.pkl", 'wb'))

input_date = '2019-09-20'
df_test.loc[input_date, [y_var]].assign(model_0=models[0].predict(df_test.loc[input_date,X_vars]))
df_test.loc[input_date, [y_var]].assign(model_0=models[1].predict(df_test.loc[input_date,X_vars]))

# Let's print the actuals vs predicted

# Load your models
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

preds_df = (
    df_test[[y_var]]
    .assign(
        model_1=predictions[0], model_2=predictions[1]
    )
)

preds_df = pd.melt(
    preds_df,
    id_vars='outcome',
    var_name='model',
    value_name='predicted',
)

sns.scatterplot(x='outcome', y="predicted",
                hue="model",
                data=preds_df)


## poetry
