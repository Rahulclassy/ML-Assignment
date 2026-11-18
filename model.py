import pandas as pd,numpy as np,json,warnings;warnings.filterwarnings('ignore')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.svm import SVR,SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.metrics import *
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
df=pd.read_csv('besttrack_2021_2024.csv',parse_dates=['t'])
d=df.set_index(['sid','t'])
H=pd.Timedelta(hours=6)
recs=[]
for (sid,t),r in d.iterrows():
    nx=d.index.isin([(sid,t+H)]).any()
    if not nx: continue
    f=d.loc[(sid,t+H)]
    pv=d.loc[(sid,t-H)] if d.index.isin([(sid,t-H)]).any() else None
    t0=d.loc[sid].index.min()
    recs.append(dict(sid=sid,t=t,year=r.year,lat=r.lat,lon=r.lon,ecp=r.ecp,msw=r.msw,dp=r.dp,
      dmsw=r.msw-pv.msw if pv is not None else np.nan,decp=r.ecp-pv.ecp if pv is not None else np.nan,
      msin=np.sin(2*np.pi*t.month/12),mcos=np.cos(2*np.pi*t.month/12),age=(t-t0).total_seconds()/3600,
      y=f.msw,grade=r.grade))
S=pd.DataFrame(recs).sort_values('t').reset_index(drop=True)
S['inc']=(S.y-S.msw>=5).astype(int)
feats=['lat','lon','ecp','msw','dp','dmsw','decp','msin','mcos','age']
tr=S[S.year<=2023];te=S[S.year==2024]
Xtr,Xte=tr[feats],te[feats];ytr,yte=tr.y,te.y
print(len(S),len(tr),len(te),S.inc.mean(),tr.inc.mean(),te.inc.mean())
cv=TimeSeriesSplit(5)
def pipe(m,poly=False,scale=True):
    st=[('i',SimpleImputer(strategy='median'))]
    if poly: st.append(('p',PolynomialFeatures(2,include_bias=False)))
    if scale: st.append(('s',StandardScaler()))
    return Pipeline(st+[('m',m)])
regs={'Linear reg. (OLS)':(pipe(LinearRegression()),{}),
 'Ridge reg.':(pipe(Ridge()),{'m__alpha':[0.1,1,10,100]}),
 'Lasso reg.':(pipe(Lasso(max_iter=5000)),{'m__alpha':[0.01,0.1,1]}),
 'Polynomial (deg 2) + Ridge':(pipe(Ridge(),poly=True),{'m__alpha':[10,100,1000]}),
 'Decision tree':(pipe(DecisionTreeRegressor(random_state=42)),{'m__max_depth':[3,5,8],'m__min_samples_leaf':[5,10,20]}),
 'k-NN regressor':(pipe(KNeighborsRegressor()),{'m__n_neighbors':[5,10,20],'m__weights':['uniform','distance']}),
 'SVR (RBF)':(pipe(SVR()),{'m__C':[10,100],'m__epsilon':[1,5]}),
 'MLP (32-16)':(pipe(MLPRegressor(hidden_layer_sizes=(32,16),max_iter=3000,early_stopping=False,random_state=42)),{'m__alpha':[1e-4,1e-2,1]})}
res={};R=[]
pers=te.msw.values
def rm(y,p):return dict(MAE=mean_absolute_error(y,p),RMSE=mean_squared_error(y,p)**.5,R2=r2_score(y,p))
res['pers']=rm(yte,pers);res['pers']['cv']=None
preds={};best={}
for n,(p,g) in regs.items():
    gs=GridSearchCV(p,g,cv=cv,scoring='neg_root_mean_squared_error') if g else None
    if gs: gs.fit(Xtr,ytr);m=gs.best_estimator_;cvr=-gs.best_score_;bp=gs.best_params_
    else:
        m=p.fit(Xtr,ytr);cvr=np.mean([mean_squared_error(ytr.iloc[b],p.fit(Xtr.iloc[a],ytr.iloc[a]).predict(Xtr.iloc[b]))**.5 for a,b in cv.split(Xtr)]);bp={};m=p.fit(Xtr,ytr)
    pr=m.predict(Xte);preds[n]=pr;r=rm(yte,pr);r['cv']=cvr;res[n]=r;best[n]={k.replace('m__',''):v for k,v in bp.items()}
    print(n,r,bp)
# persistence cv rmse on train
res['pers']['cv']=mean_squared_error(ytr,tr.msw)**.5
bn=min([k for k in res if k!='pers'],key=lambda k:res[k]['RMSE'])
# persistence + trend baseline
tre=(te.msw+te.dmsw.fillna(0)).values;res['trend']=rm(yte,tre)
# classification
clfs={'Logistic regression':(pipe(LogisticRegression(max_iter=2000)),{'m__C':[0.01,0.1,1,10],'m__class_weight':[None,'balanced']}),
 'Gauss. Naive Bayes':(pipe(GaussianNB()),{}),
 'Decision tree':(pipe(DecisionTreeClassifier(random_state=42)),{'m__max_depth':[3,4,6],'m__class_weight':[None,'balanced']}),
 'k-NN':(pipe(KNeighborsClassifier()),{'m__n_neighbors':[5,15,25]}),
 'SVM (RBF)':(pipe(SVC(probability=True,random_state=42)),{'m__C':[0.1,1,10],'m__class_weight':[None,'balanced']}),
 'MLP (32-16)':(pipe(MLPClassifier(hidden_layer_sizes=(32,16),max_iter=3000,random_state=42)),{'m__alpha':[1e-4,1e-2,1]})}
yc,yct=tr.inc,te.inc
# persistence-trend baseline: intensifying if last 6h change >=5
bp_=(te.dmsw.fillna(0)>=5).astype(int)
C={'Trend baseline':dict(Acc=accuracy_score(yct,bp_),P=precision_score(yct,bp_,zero_division=0),R=recall_score(yct,bp_),F1=f1_score(yct,bp_),AUC=None)}
rocs={};cm=None
for n,(p,g) in clfs.items():
    if g:
        gs=GridSearchCV(p,g,cv=cv,scoring='f1');gs.fit(Xtr,yc);m=gs.best_estimator_;bp=gs.best_params_
    else: m=p.fit(Xtr,yc);bp={}
    pr=m.predict(Xte);pp=m.predict_proba(Xte)[:,1]
    C[n]=dict(Acc=accuracy_score(yct,pr),P=precision_score(yct,pr,zero_division=0),R=recall_score(yct,pr),F1=f1_score(yct,pr),AUC=roc_auc_score(yct,pp))
    rocs[n]=roc_curve(yct,pp)[:2]
    if n=='Logistic regression': cm=confusion_matrix(yct,pr).tolist();coef=dict(zip(feats,m.named_steps['m'].coef_[0]))
    print(n,C[n],bp)
    best['c_'+n]={k.replace('m__',''):v for k,v in bp.items()}
# by grade error of best
te2=te.copy();te2['err']=np.abs(preds[bn]-yte.values);te2['perr']=np.abs(pers-yte.values)
bg=te2.groupby('grade').agg(n=('err','size'),mae=('err','mean'),pm=('perr','mean')).reindex(['D','DD','CS','SCS','VSCS','ESCS']).dropna().reset_index()
# figures
cols={'D':'#9ecae1','DD':'#4292c6','CS':'#fdae6b','SCS':'#f16913','VSCS':'#d7301f','ESCS':'#7f0000'}
fig,ax=plt.subplots(figsize=(3.4,2.9))
for sid,g in df.groupby('sid'):
    ax.plot(g.lon,g.lat,color='#999',lw=.5,zorder=1)
for gr,c in cols.items():
    g=df[df.grade==gr];ax.scatter(g.lon,g.lat,s=5,color=c,label=gr,zorder=2)
ax.set_xlabel('Longitude (°E)');ax.set_ylabel('Latitude (°N)');ax.legend(fontsize=6,ncol=3,frameon=False,loc='upper left');ax.set_xlim(50,102);ax.set_ylim(4,32)
plt.tight_layout();plt.savefig('f1.png',dpi=220);plt.close()
fig,ax=plt.subplots(figsize=(3.4,2.2))
vc=df.grade.value_counts().reindex(list(cols));ax.bar(vc.index,vc.values,color=list(cols.values()))
for i,v in enumerate(vc.values):ax.text(i,v+4,str(v),ha='center',fontsize=7)
ax.set_ylabel('Best-track records');plt.tight_layout();plt.savefig('f2.png',dpi=220);plt.close()
fig,ax=plt.subplots(figsize=(3.4,3.0))
cc=S[feats+['y']].corr();im=ax.imshow(cc,cmap='RdBu_r',vmin=-1,vmax=1)
lab=['lat','lon','ECP','MSW','ΔP','ΔMSW6','ΔECP6','sin m','cos m','age','MSW+6h'];ax.set_xticks(range(11));ax.set_xticklabels(lab,rotation=70,fontsize=6);ax.set_yticks(range(11));ax.set_yticklabels(lab,fontsize=6)
for i in range(11):
    for j in range(11):ax.text(j,i,f'{cc.iloc[i,j]:.1f}'.replace('0.','.'),ha='center',va='center',fontsize=4.5)
plt.colorbar(im,fraction=.046);plt.tight_layout();plt.savefig('f3.png',dpi=220);plt.close()
fig,ax=plt.subplots(figsize=(3.4,3.0))
ax.scatter(yte,preds[bn],s=8,alpha=.6,color='#2c5aa0');ax.plot([15,115],[15,115],color='#e6730e')
ax.set_xlabel('Observed MSW at t+6 h (kt)');ax.set_ylabel('Predicted MSW (kt)');ax.set_title(f"{bn}: RMSE={res[bn]['RMSE']:.2f}, R²={res[bn]['R2']:.3f}",fontsize=7)
plt.tight_layout();plt.savefig('f4.png',dpi=220);plt.close()
fig,ax=plt.subplots(figsize=(3.4,3.0))
for n,(f,t_) in rocs.items():ax.plot(f,t_,label=f"{n} ({C[n]['AUC']:.3f})",lw=1)
ax.plot([0,1],[0,1],'--',color='grey',lw=.6);ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate');ax.legend(fontsize=5.5,frameon=False,loc='lower right')
plt.tight_layout();plt.savefig('f5.png',dpi=220);plt.close()
out=dict(res=res,C=C,best=best,bn=bn,cm=cm,coef=coef,bg=bg.values.tolist(),n=len(S),ntr=len(tr),nte=len(te),pinc=[S.inc.mean(),tr.inc.mean(),te.inc.mean()],
 nraw=len(df),nsys=int(df.sid.nunique()),sysyr=df.groupby('year').sid.nunique().to_dict(),recyr=df.groupby('year').size().to_dict(),
 gc=df.grade.value_counts().to_dict(),corr=cc['y'].to_dict(),ntrsys=int(tr.sid.nunique()),ntesys=int(te.sid.nunique()),
 maxmsw=int(df.msw.max()),minecp=float(df.ecp.min()))
json.dump(out,open('out.json','w'),default=float,indent=1)
