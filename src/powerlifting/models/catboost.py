"""CatBoost v3 : contexte, précision de mesure et variantes de codage catégoriel."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from .catboost_features import allometric_cat_features, allometric_scale

def make_features(frame, config):
    x,cats=allometric_cat_features(frame)
    if config.get('context',True):
        x['EstimatedBirthYear']=x.MeetYear-x.Age
        w=x.BodyweightKg
        x['WeightInteger']=np.isclose(w,np.round(w),atol=1e-6).astype(float)
        x['WeightHalf']=np.isclose(w*2,np.round(w*2),atol=1e-6).astype(float)
        x['WeightTenth']=np.isclose(w*10,np.round(w*10),atol=1e-6).astype(float)
        x['WeightFraction']=w-np.floor(w)
        x['WeightPoundRounding']=np.abs(w/0.45359237-np.round(w/0.45359237))
        for a,b,name in [('Country','MeetCountry','TravelCountry'),('State','MeetState','TravelState')]:
            known=x[a].ne('Unknown') & x[b].ne('Unknown')
            x[name]=x[a].ne(x[b]).astype(float).where(known)
        pairs=[('Sex','Equipment'),('Sex','Federation'),('Sex','WeightClassKg'),
               ('Federation','MeetYear'),('Federation','Equipment')]
        for pair in pairs:
            name='__'.join(pair)
            x[name]=x[pair[0]].astype(str)+'|'+x[pair[1]].astype(str)
            cats.append(name)
        if config.get('competition_context',True):   # variable fédération|pays|région|année
            name='CompetitionContext'
            x[name]=x.Federation+'|'+x.MeetCountry+'|'+x.MeetState+'|'+x.MeetYear.astype(str)
            cats.append(name)
    return x,cats

def fit_candidate(frame, target, config, validation=None):
    x,cats=make_features(frame,config)
    exponent=config.get('exponent',0.)
    scale=allometric_scale(frame,exponent)
    pool=Pool(x,np.asarray(target,dtype=float)/scale,cat_features=cats,weight=scale**2)
    params=dict(config['params'])
    params['iterations']=int(config.get('best_iteration',params.get('iterations',4000)))
    model=CatBoostRegressor(**params)
    kwargs={'verbose':False}
    if validation is not None:
        valid,vy=validation
        vx,_=make_features(valid,config)
        vs=allometric_scale(valid,exponent)
        kwargs.update(eval_set=Pool(vx,np.asarray(vy)/vs,cat_features=cats,weight=vs**2),
                      early_stopping_rounds=config.get('patience',250),use_best_model=True)
    model.fit(pool,**kwargs)
    fixed=json.loads(json.dumps(config))
    fixed['best_iteration']=model.tree_count_
    return {'model':model,'config':fixed}

def predict_candidate(fitted,frame):
    x,_=make_features(frame,fitted['config'])
    p=fitted['model'].predict(x)*allometric_scale(frame,fitted['config'].get('exponent',0.))
    return np.maximum(p,0.)

def save_candidate(fitted,directory):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    fitted['model'].save_model(str(d/'model.cbm'))
    (d/'config.json').write_text(json.dumps(fitted['config'],indent=2))

def load_candidate(directory):
    d=Path(directory);m=CatBoostRegressor();m.load_model(str(d/'model.cbm'))
    return {'model':m,'config':json.loads((d/'config.json').read_text())}
