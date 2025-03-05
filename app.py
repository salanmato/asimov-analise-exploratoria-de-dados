# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

df = pd.read_csv('survey_results_public.csv', index_col='ResponseId')

# 01-03

df.info()

df.isna()

# %%
df.isna().sum()

# %%
df.isna().sum() / len(df)

def format_percent(value):
    return f'{100 * value:.2f}%'


# %%
(df.isna().sum() / len(df)).apply(format_percent)

# %%
(df.isna().sum() / len(df)).sort_values(ascending=False).apply(format_percent)

percent_data = (df.isna().sum() / len(df)).sort_values(ascending=False).reset_index()
percent_data


# %%
percent_data.columns = ['Column', 'Percent']
# %%
percent_data
# %%
fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(x='Column', y='Percent', data=percent_data, ax=ax, orient='v',gap=0.5, palette='viridis')
fig.suptitle('Missing values percentage')
plt.xticks(rotation=90)
# %%

# 01-04

df.select_dtypes('number')
# %%

df.describe()
# %%
df[['WorkExp', 'ConvertedCompYearly']].select_dtypes('number').describe().round(2)
# %%

# 01-05

df.select_dtypes('object')
# %%
df['RemoteWork'].unique()

# %%
df['RemoteWork'].astype('string')

# %%
df['RemoteWork'].value_counts(normalize=True, dropna=False).apply(format_percent)
# %%

df[['Age', 'RemoteWork']].value_counts().sort_index()
# %%


# 02-01

sns.set()
sns.set_context('poster')
# %%

df['WorkExp'].describe().round(2)

fig, ax = plt.subplots(figsize=(14, 6))

sns.boxplot(x='WorkExp', data=df, ax=ax)
fig.suptitle('Work Experience')



# %%
x_noise = np.random.uniform(-0.1, 0.1, len(df))

x_scatterplot = df['WorkExp'] + x_noise

y_scatterplot = np.random.uniform(-0.1, 0.1, len(df))

fig, ax = plt.subplots(figsize=(14, 6))
sns.scatterplot(x=x_scatterplot, y=y_scatterplot, alpha=0.8, s=20, data=df, ax=ax)
fig.suptitle('Work Experience')

# %%
# 02-02
# 
sns.set()
sns.set_context('poster')

fig, ax = plt.subplots(figsize=(12, 6))

sns.histplot(data=df, ax=ax, x='WorkExp', binwidth=1.0, kde=True)
fig.suptitle('Work Experience')
plt.show()

# # %%

# %%
df['Age'].unique()
# %%
df['Age'].value_counts(normalize=True, dropna=False).apply(format_percent)

# %%
mapa_idades = {
    'Under 18 years old': '[00 - 18]',
    '18-24 years old': '[18 - 24]',
    '25-34 years old': '[25 - 34]',
    '35-44 years old': '[35 - 44]',
    '45-54 years old': '[45 - 54]',
    '55-64 years old': '[55 - 64]',
    '65 years or older': '[65 - 99]',
    'Prefer not to say': '[?? - ??]',
    np.nan: '[?? - ??]',
}

# %%
df['AgeRange'] = df['Age'].map(mapa_idades)

# %%
df['AgeRange'].value_counts(dropna=False)
# %%
df['Age'].value_counts(dropna=False)

# %%
df[['Age', 'AgeRange']].value_counts().sort_index()

# %%
df_filtered = df.loc[df['AgeRange'] != '[?? - ??]']
# %%
df_filtered['AgeRange'].value_counts(dropna=False)
# %%
fig, ax = plt.subplots(figsize=(12, 6))

idades = sorted(df_filtered['AgeRange'].unique())
sns.histplot(data=df_filtered, ax=ax, x='WorkExp', hue='AgeRange', binwidth=1.0, kde=True, multiple='stack', hue_order=idades)
fig.suptitle('Work Experience by Age Range')
# %%
fig, ax = plt.subplots(figsize=(12, 6))

idades = sorted(df_filtered['AgeRange'].unique())
sns.histplot(data=df_filtered, 
             ax=ax, x='WorkExp', 
             hue='AgeRange', 
             binwidth=1.0, 
             kde=True, 
             multiple='stack', 
             hue_order=idades,
             stat='density',
             common_norm=False)
fig.suptitle('Work Experience by Age Range')
# %%

df['RemoteWork'].value_counts(normalize=True, dropna=False).apply(format_percent)
# %%

# %%
fig, ax = plt.subplots(figsize=(12, 6))

sns.histplot(data=df_filtered, 
             ax=ax, x='WorkExp', 
             hue='RemoteWork', 
             binwidth=1.0, 
             kde=True, 
             multiple='stack', 
             stat='density',
             common_norm=False)
fig.suptitle('Work Experience by Age Range')
# %%
media = df['WorkExp'].mean()
desvio = df['WorkExp'].std()
v_min = df['WorkExp'].min()
v_max = df['WorkExp'].max()

xs = np.linspace(v_min, v_max, 10_000)
ys = norm.pdf(xs, loc=media, scale=desvio)

fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(data=df, ax=ax, x='WorkExp', binwidth=1.0, kde=True, stat='density')
ax.plot(xs, ys, color='red', lw=2)
# %%
