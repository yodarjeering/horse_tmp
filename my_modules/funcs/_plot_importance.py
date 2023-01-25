import pandas as pd

def plot_importances(xgb_model, x_test,n_display=20):
    importances = pd.DataFrame(
    {'features' : x_test.columns, 'importances' : xgb_model.feature_importance()})
    print(importances.sort_values('importances', ascending=False)[:n_display])