# %%
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
from pandas.api.types import CategoricalDtype
from plotnine import ggplot, aes, geom_line, labs, theme, facet_wrap, scale_color_manual
from ydata_profiling import ProfileReport

# %%
df = pd.read_csv('data/ground_station.csv')
df.describe()

# %%
df['Timestamp'] = pd.to_datetime(df['Created_at'])
df = df[df['Timestamp'] >= pd.to_datetime('2024-05-30')]
df.set_index('Timestamp', inplace=True)
df = df.drop(columns=['Created_at'], axis=1)
df = df.drop(columns=['Longitude'], axis=1)
df = df.drop(columns=['Latitude'], axis=1)
df = df.resample('30min').mean()
df = df.interpolate(method='time')
# %%
df.describe()

# %%
decomps = {}
residuals = pd.DataFrame(index=df.index)
for col in df.columns:
    if col == 'Timestamp':
        continue
    stl_res = STL(df[col], period=24, robust=True).fit()
    decomps[col] = stl_res
    residuals[col] = stl_res.resid

# %%
iso = IsolationForest(contamination='auto', random_state=42)
iso.fit(residuals)

# %%
flags = iso.predict(residuals)
is_multivariate_anomaly = pd.Series(flags == -1, index=residuals.index)
df['is_multivariate_anomaly'] = is_multivariate_anomaly
df.head()

# %%
print('Foram encontrados {} outliers'.format(is_multivariate_anomaly.sum()))

# %%
df_clean = df.copy()
for col, stl_res in decomps.items():
    cleaned_series = df[col].copy()
    cleaned_series[is_multivariate_anomaly] = (stl_res.trend + stl_res.seasonal)[is_multivariate_anomaly]
    df_clean[col] = cleaned_series
df_clean.drop(columns=['is_multivariate_anomaly'], inplace=True)
df.drop(columns=['is_multivariate_anomaly'], inplace=True)
df_clean.head()

# %%
df_clean['Temperature'] = df_clean['Temperature'].round(3)
df_clean['Precipitation'] = df_clean['Precipitation'].round(3)
df_clean['Humidity'] = df_clean['Humidity'].round(3)
df_clean['Wind_Speed_kmh'] = df_clean['Wind_Speed_kmh'].round(3)
df_clean['Wind_Direction'] = df_clean['Wind_Direction'].round(3)
df_clean['Soil_Moisture'] = df_clean['Soil_Moisture'].round(3)
df_clean['Soil_Temperature'] = df_clean['Soil_Temperature'].round(3)
df_clean.head()
# %%
print('--- Original ---')
print(df.describe())
print('--- Limpo ---')
print(df_clean.describe())

# %%
orig_long = (
    df
    .reset_index()
    .melt(id_vars='Timestamp',
          var_name='variable',
          value_name='value')
    .assign(type='original')
)

clean_long = (
    df_clean
    .reset_index()
    .melt(id_vars='Timestamp',
          var_name='variable',
          value_name='value')
    .assign(type='cleaned')
)

combined = pd.concat([orig_long, clean_long], ignore_index=True)
type_order = CategoricalDtype(['original', 'cleaned'], ordered=True)
combined['type'] = combined['type'].astype(type_order)

plot = (
    ggplot(combined, aes('Timestamp', 'value', color='type'))
    + geom_line()
    + facet_wrap('~variable', scales='free_y', ncol=2)
    + scale_color_manual(values=['firebrick', 'navy'])
    + labs(
        title='Comparação: valores originais vs pós-tratamento',
        x='Timestamp',
        y='Valor',
        color='Série'
    )
    + theme(figure_size=(10, 6))
)

plot.show()
# %%
df = df_clean


# %%
#Transformar Wind_Direction (porque ângulo 0° e 360° são "iguais")
df['Wind_Dir_Sin'] = np.sin(np.deg2rad(df['Wind_Direction']))
df['Wind_Dir_Cos'] = np.cos(np.deg2rad(df['Wind_Direction']))
df['Precipitation_log'] = np.log1p(df['Precipitation'])

df.drop(columns=['Wind_Direction'], inplace=True)
df.drop(columns=['Precipitation'], inplace=True)
# %%
profile = ProfileReport(df, title='Profile Report')
profile.to_file('profile_report.html')

# %%
df.reset_index(inplace=True)
# 1) Assegure que o índice ou coluna seja datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 2) Unix time normalizado (feature contínua)
df['ts_unix'] = df['Timestamp'].astype(int) // 10**9
df['ts_norm'] = (df['ts_unix'] - df['ts_unix'].mean()) / df['ts_unix'].std()

# 3) Componentes cíclicos do dia, do ano e da semana
# hora do dia
h = df['Timestamp'].dt.hour + df['Timestamp'].dt.minute/60
df['hour_sin'] = np.sin(2 * np.pi * h / 24)
df['hour_cos'] = np.cos(2 * np.pi * h / 24)

# dia do ano
doy = df['Timestamp'].dt.dayofyear + h/24
df['doy_sin'] = np.sin(2 * np.pi * doy / 365)
df['doy_cos'] = np.cos(2 * np.pi * doy / 365)

# dia da semana
dow = df['Timestamp'].dt.dayofweek
df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

# 4) Δt entre observações (seu índice for ordenado)
df = df.sort_values('Timestamp')
df['delta_t'] = df['ts_unix'].diff().fillna(0)
df['delta_t_norm'] = (df['delta_t'] - df['delta_t'].mean()) / df['delta_t'].std()
# %%
df.to_csv('data/cleaned_data.csv', index=False)
