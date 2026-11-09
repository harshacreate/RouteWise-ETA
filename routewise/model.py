import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def fit_models(X,y,seed=42):
    baseline=DummyRegressor(strategy="mean")
    stronger=HistGradientBoostingRegressor(max_iter=160,learning_rate=.06,max_leaf_nodes=15,l2_regularization=1.0,random_state=seed)
    baseline.fit(X,y);stronger.fit(X,y)
    return {"baseline_mean":baseline,"stronger_hist_gradient_boosting":stronger}
def evaluate(model,X,y,tolerance):
    pred=model.predict(X);err=np.abs(np.asarray(y)-pred)
    return {"n":int(len(y)),"mae":round(float(mean_absolute_error(y,pred)),4),"rmse":round(float(np.sqrt(mean_squared_error(y,pred))),4),"r2":round(float(r2_score(y,pred)),4),"p90_absolute_error":round(float(np.quantile(err,.9)),4),"within_tolerance_rate":round(float(np.mean(err<=tolerance)),4)}
