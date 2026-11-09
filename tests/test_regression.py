import json,os,subprocess,sys
import numpy as np
from routewise.data import FEATURES,make_dataset
from routewise.split import split_indices
from routewise.train import run
def test_generator_and_group_split():
    X,y,g=make_dataset(480,11);X2,y2,g2=make_dataset(480,11)
    assert X.shape==(480,len(FEATURES)) and np.isfinite(X).all() and np.isfinite(y).all()
    assert np.array_equal(X,X2) and np.array_equal(y,y2)
    tr,te=split_indices(y,g,5);assert not set(g[tr]).intersection(g[te])
def test_metrics_and_cli():
    report=run(9,480)
    for key in ("baseline","stronger_model"):
        assert report[key]["mae"]>=0 and report[key]["p90_absolute_error"]>=0
    env=dict(os.environ);env["PYTHONPATH"]=os.pathsep.join([os.getcwd(),env.get("PYTHONPATH","")])
    p=subprocess.run([sys.executable,"-m","routewise.train","--samples","320","--json"],check=True,capture_output=True,text=True,env=env)
    assert json.loads(p.stdout)["project"]=='RouteWise-ETA — Delivery-Time Forecasting Prototype'
