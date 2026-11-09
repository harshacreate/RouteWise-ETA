"""Synthetic regression data for RouteWise-ETA — Delivery-Time Forecasting Prototype."""
from __future__ import annotations
import numpy as np
FEATURES=['distance', 'stops', 'traffic', 'rain', 'hour', 'complexity', 'urban_share']
def make_dataset(n=2400,seed=42):
    if n<160:raise ValueError("n must be at least 160")
    rng=np.random.default_rng(seed)
    routes=max(30,n//30);route=np.arange(n)%routes;route_bias=rng.normal(0,5,routes)[route];distance=np.clip(rng.gamma(2.5,4,n),.5,60);stops=rng.poisson(2,n).astype(float);traffic=np.clip(rng.beta(2,2,n),0,1);rain=np.clip(rng.gamma(1,1.2,n),0,8);hour=rng.integers(0,24,n).astype(float);complexity=np.clip(rng.normal(.5,.2,n),0,1);urban_share=rng.uniform(0,1,n);y=8+2.4*distance+3.5*stops+18*traffic+2.2*rain+6*(hour>=16)+9*complexity+4*urban_share+route_bias+rng.normal(0,5,n);groups=route
    X=np.column_stack([distance, stops, traffic, rain, hour, complexity, urban_share]).astype(float)
    if X.shape!=(n,len(FEATURES)) or not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError("invalid generated regression data")
    return X,np.asarray(y,dtype=float),np.asarray(groups)
