"""Single-stage flight revenue scenario model. Prices are public-fare proxies."""
import numpy as np
import pandas as pd
KEY = ['出發日期','航空公司','航班號碼']
ID = '原資料Excel列號'
PRED = 'XGBoost預測票價(TWD)'

def evaluate(p, p0, q0, s, e, m=1.0):
    p,p0,q0,s = map(lambda x:np.asarray(x,dtype=float),(p,p0,q0,s))
    q = q0*m*(p/p0)**e
    sold = np.minimum(q,s)
    return q,sold,p*sold

def optimal(p0,q0,s,e,band=.2):
    p0,q0,s = np.broadcast_arrays(np.asarray(p0,float),np.asarray(q0,float),np.asarray(s,float))
    # For negative constant elasticity the only possible interior optimum is capacity intersection.
    kink=p0*np.power(np.maximum(q0/s,1e-15),-1/e)
    cand=np.stack([p0,p0*(1-band),p0*(1+band),np.clip(kink,p0*(1-band),p0*(1+band))],axis=-1)
    rev=evaluate(cand,p0[...,None],q0[...,None],s[...,None],e)[2]
    i=np.argmax(rev,axis=-1)
    return np.take_along_axis(cand,i[...,None],axis=-1)[...,0]

def attach_predictions(base,pred):
    required=[ID,'DTD','票價(TWD)',PRED]
    missing=set(required)-set(pred.columns)
    if missing: raise ValueError('預測檔缺少欄位：'+str(sorted(missing)))
    p=pred[required].copy()
    for c in required: p[c]=pd.to_numeric(p[c],errors='raise')
    if p.empty or p.isna().any().any() or p[ID].duplicated().any(): raise ValueError('預測檔有空值、無資料或重複列號。')
    if not np.isfinite(p.to_numpy(dtype=float)).all() or (p[PRED]<=0).any(): raise ValueError('預測價格須為有限正數。')
    if not p[ID].isin(base[ID]).all(): raise ValueError('存在無法對應的原始列號。')
    z=base.merge(p,on=ID,suffixes=('','_check'),validate='one_to_one')
    if not np.allclose(z['DTD'],z['DTD_check'],rtol=0,atol=0) or not np.allclose(z['歷史報價'],z['票價(TWD)'],rtol=0,atol=.01): raise ValueError('DTD 或歷史報價核對失敗；請確認資料排列及列號。')
    return z.drop(columns=['DTD_check','票價(TWD)'])

def flight_sample(data,dtd,price_col,demand_col):
    d=data.loc[data['DTD']==dtd].copy()
    # Multiple quotes for a flight/day are aggregated once, not treated as separate flights.
    g=d.groupby(KEY,as_index=False).agg(P0=(price_col,'median'),Q0=(demand_col,'first'),S=('座位數','first'),時段=('起飛時段','first'),報價筆數=(ID,'size'))
    g=g[np.isfinite(g[['P0','Q0','S']]).all(axis=1)&(g.P0>0)&(g.Q0>=0)&(g.S>0)].copy()
    return g

def strategies(f,e,band):
    out=f.copy(); p=out.P0.to_numpy(); q=out.Q0.to_numpy(); s=out.S.to_numpy(); lf=q/s
    out['A']=p; out['B']=p*np.clip(np.where(lf<.8,.95,np.where(lf>.9,1.05,1)),1-band,1+band)
    out['C']=optimal(p,q,s,e,band)
    return out

def compare(f,e,m=1):
    rows=[]
    for name in ['A','B','C']:
        q,sold,r=evaluate(f[name],f.P0,f.Q0,f.S,e,m)
        rows.append({'策略':name,'票價收入估計':float(r.sum()),'載客率(%)':float(sold.sum()/f.S.sum()*100),'剩餘座位':float(f.S.sum()-sold.sum()),'平均售票價':float(r.sum()/sold.sum()) if sold.sum()>0 else np.nan})
    d=pd.DataFrame(rows);d['相對A收入變化(%)']=(d['票價收入估計']/d['票價收入估計'].iloc[0]-1)*100 if d['票價收入估計'].iloc[0]>0 else np.nan
    return d
