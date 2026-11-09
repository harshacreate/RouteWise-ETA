import argparse,json
from .data import make_dataset
from .split import split_indices
from .model import fit_models,evaluate
PROJECT='RouteWise-ETA — Delivery-Time Forecasting Prototype';TOLERANCE=15
def run(seed=42,samples=2400):
    X,y,g=make_dataset(samples,seed);tr,te=split_indices(y,g,seed);models=fit_models(X[tr],y[tr],seed)
    return {"project":PROJECT,"data":"synthetic demonstration data","seed":int(seed),"split":"entity-group holdout","n_train":int(len(tr)),"n_test":int(len(te)),"unit":'minutes',"tolerance":TOLERANCE,"baseline":evaluate(models["baseline_mean"],X[te],y[te],TOLERANCE),"stronger_model":evaluate(models["stronger_hist_gradient_boosting"],X[te],y[te],TOLERANCE)}
def main(argv=None):
    p=argparse.ArgumentParser(description="Train/evaluate a synthetic regression prototype.");p.add_argument("--seed",type=int,default=42);p.add_argument("--samples",type=int,default=2400);p.add_argument("--json",action="store_true");a=p.parse_args(argv);print(json.dumps(run(a.seed,a.samples),indent=2))
if __name__=="__main__":main()
