
#Libraries
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

#dataset
data = fetch_ucirepo(id=555)
df = pd.concat([data.data.features, data.data.targets], axis=1)

#numeric columns
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
df['square_feet'] = pd.to_numeric(df['square_feet'], errors='coerce')

# keep only needed cols and drop NA
df = df[['price', 'bedrooms', 'bathrooms', 'square_feet',
         'category', 'cityname', 'body', 'amenities']].dropna()
df = df[df['price'] < 10000]
df['neighborhood'] = df['cityname']

#Simulate platforms with heterogeneous neighborhood mixes

platforms = ['Craigslist', 'Zillow', 'Apartments.com']

#one random platform mix per neighborhood
neighs = df['neighborhood'].unique()
mixes = np.random.dirichlet([1, 1, 1], size=len(neighs))
mix_df = pd.DataFrame(mixes, columns=['p_craigslist', 'p_zillow', 'p_apts'])
mix_df['neighborhood'] = neighs

#attach mix to listings
df = df.merge(mix_df, on='neighborhood', how='left')

def sample_platform(row):
    return np.random.choice(
        platforms,
        p=[row['p_craigslist'], row['p_zillow'], row['p_apts']]
    )

df['platform'] = df.apply(sample_platform, axis=1)


#Platform-level dispersion

platform_dispersion = (
    df.groupby(['platform', 'neighborhood'])['price']
      .std()
      .reset_index()
)
platform_dispersion.rename(columns={'price': 'platform_neigh_std'}, inplace=True)
df = df.merge(platform_dispersion, on=['platform', 'neighborhood'], how='left')

#Multihoming + exposure
df['landlord_id'] = np.random.randint(1000, 2000, size=len(df))
df['multihomed'] = df['landlord_id'] % 7 == 0  # ~14%

platform_weights = {'Craigslist': 1.0, 'Zillow': 1.5, 'Apartments.com': 1.3}
df['exposure_score'] = df['platform'].map(platform_weights) * (1 + df['multihomed'] * 0.8)

df['views'] = (
    df['exposure_score'] * (2000 / df['price'])
    + np.random.normal(0, 40, size=len(df))
)

#Proper platform share + HHI

# counts per neighborhood/platform
counts = (
    df.groupby(['neighborhood', 'platform'])['price']
      .count()
      .reset_index(name='platform_count')
)

#total listings per neighborhood
total = (
    df.groupby('neighborhood')['price']
      .count()
      .reset_index(name='total_count')
)

# merge and compute share
counts = counts.merge(total, on='neighborhood')
counts['platform_share_in_neighborhood'] = (
    counts['platform_count'] / counts['total_count']
)

# HHI per neighborhood
hhi = (
    counts
    .groupby('neighborhood')['platform_share_in_neighborhood']
    .apply(lambda s: (s**2).sum())
    .reset_index(name='platform_concentration_hhi')
)

# merge back to listing level
df = df.merge(
    counts[['neighborhood', 'platform', 'platform_count',
            'total_count', 'platform_share_in_neighborhood']],
    on=['neighborhood', 'platform']
)
df = df.merge(hhi, on='neighborhood')

#Time-on-market with U-shape in HHI

noise = np.random.normal(0, 2, size=len(df))
df['time_on_market'] = (
    28
    + 10 * (df['platform_concentration_hhi'] - 0.5)**2  # convex
    - 3 * df['exposure_score']
    + (df['price'] / 500)
    + noise
).clip(1, 60)

df['rented'] = (df['time_on_market'] < 20).astype(int)

# 8. Feature engineering + dummies
df['description_length'] = df['body'].fillna('').str.len()
df['amenities_count'] = df['amenities'].fillna('').apply(lambda x: len(str(x).split(',')))

df = pd.get_dummies(df, columns=['category', 'cityname', 'platform'], drop_first=True)

features = list(dict.fromkeys(
    [
        'bedrooms', 'bathrooms', 'square_feet',
        'description_length', 'amenities_count',
        'platform_neigh_std',
        'multihomed', 'exposure_score',
        'platform_share_in_neighborhood',
        'platform_concentration_hhi'
    ]
    + [c for c in df.columns
       if c.startswith('category_') or c.startswith('cityname_') or c.startswith('platform_')]
))

#Train/test split
X = df[features]
y_price = df['price']
y_time = df['time_on_market']
y_views = df['views']
y_rented = df['rented']

X_train, X_val, y_price_train, y_price_val = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)
_, _, y_time_train, y_time_val = train_test_split(
    X, y_time, test_size=0.2, random_state=42
)
_, _, y_views_train, y_views_val = train_test_split(
    X, y_views, test_size=0.2, random_state=42
)
_, _, y_rented_train, y_rented_val = train_test_split(
    X, y_rented, test_size=0.2, random_state=42
)

# Train LightGBM models
def train_lgbm(X_train, y_train, task='regression'):
    params = {
        'objective': 'regression' if task == 'regression' else 'binary',
        'metric': 'rmse' if task == 'regression' else 'auc',
        'learning_rate': 0.05,
        'verbosity': -1
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    return lgb.train(params, dtrain, num_boost_round=100)

model_price = train_lgbm(X_train, y_price_train)
model_time = train_lgbm(X_train, y_time_train)
model_views = train_lgbm(X_train, y_views_train)
model_rented = train_lgbm(X_train, y_rented_train, task='binary')

# 11. Evaluation
pred_price = model_price.predict(X_val)
pred_time = model_time.predict(X_val)
pred_views = model_views.predict(X_val)
pred_rented = model_rented.predict(X_val)

print(f"\nPrice RMSE: ${rmse(y_price_val, pred_price):.2f}")
print(f"Time-on-market RMSE: {rmse(y_time_val, pred_time):.2f} days")
print(f"Views RMSE: {rmse(y_views_val, pred_views):.2f} views")
print(f"Rented AUC: {roc_auc_score(y_rented_val, pred_rented):.3f}")

# 12. SHAP (optional — will pop plots)
explainer_price = shap.TreeExplainer(model_price)
shap_values_price = explainer_price.shap_values(X_val)
print("\nSHAP summary for rent price:")
shap.summary_plot(shap_values_price, X_val)

explainer_rented = shap.TreeExplainer(model_rented)
shap_values_rented = explainer_rented.shap_values(X_val)
print("\nSHAP summary for rental success:")
shap.summary_plot(shap_values_rented, X_val)

# 13. Partial dependence on HHI
h_min = X['platform_concentration_hhi'].min()
h_max = X['platform_concentration_hhi'].max()
h_grid = np.linspace(h_min, h_max, 80)

avg_preds = []
for h in h_grid:
    X_tmp = X.copy()
    X_tmp['platform_concentration_hhi'] = h
    preds = model_time.predict(X_tmp)
    avg_preds.append(preds.mean())

from scipy.ndimage import gaussian_filter1d
smooth_preds = gaussian_filter1d(avg_preds, sigma=1.2)

plt.figure(figsize=(6, 4))
plt.plot(h_grid, smooth_preds, color='darkred', lw=3, label='LightGBM PD (smoothed)')
plt.plot(h_grid, avg_preds, color='darkred', ls='--', alpha=0.25, label='raw PD')
plt.xlabel("Platform Concentration (HHI)")
plt.ylabel("Predicted Time on Market")
plt.title("Predicted Relationship: Concentration vs Time-on-Market")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
