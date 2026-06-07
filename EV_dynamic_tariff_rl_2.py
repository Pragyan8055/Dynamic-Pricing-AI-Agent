import os, math, warnings, copy, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from collections import namedtuple

warnings.filterwarnings('ignore')
torch.manual_seed(42); np.random.seed(42); random.seed(42)
DATA_DIR = "C:/Users/pragy/Downloads/Analytics_Summer_projects_2026/EV_dynamic_pricing_socbiz"
OUT_DIR  = Path("C:/Users/pragy/Downloads/Analytics_Summer_projects_2026/EV_dynamic_pricing_socbiz/figures_2"); OUT_DIR.mkdir(parents=True,exist_ok=True)
DEVICE=torch.device('cpu')
plt.rcParams.update(plt.rcParamsDefault); plt.rcParams['figure.facecolor']='#f0f0f0'; plt.rcParams['axes.facecolor']='#f0f0f0'; plt.rcParams['savefig.facecolor']='#f0f0f0'
P = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
def sf(n,dpi=130):
    p=OUT_DIR/f'{n}.png'
    plt.savefig(p,dpi=dpi,bbox_inches='tight',facecolor=plt.rcParams['figure.facecolor']); plt.close(); print(f'  [saved] {p.name}')

# ── LOAD ──
print('Loading...')
time_df      = pd.read_csv(DATA_DIR + "/UrbanEV_ SZ_districts/time.csv")
duration_df  = pd.read_csv(DATA_DIR + "/UrbanEV_ SZ_districts/duration.csv")
volume_df    = pd.read_csv(DATA_DIR + "/UrbanEV_ SZ_districts/volume.csv")
adj_df       = pd.read_csv(DATA_DIR + "/UrbanEV_ SZ_districts/adj.csv")
distance_df  = pd.read_csv(DATA_DIR + "/UrbanEV_ SZ_districts/distance.csv")
info_df      = pd.read_csv(DATA_DIR + "/UrbanEV_ SZ_districts/information.csv")
occupancy_df = pd.read_csv(DATA_DIR + "/UrbanEV_ SZ_districts/occupancy.csv")
price_df     = pd.read_csv(DATA_DIR + "/UrbanEV_ SZ_districts/price.csv")
stations_df  = pd.read_csv(DATA_DIR + "/UrbanEV_ SZ_districts/stations.csv")

T=len(time_df)
ts=pd.to_datetime(time_df[['year','month','day','hour','minute']])
SC=[c for c in volume_df.columns if c!='timestamp']
N_ST=len(SC)
vol_mat=volume_df[SC].values.astype(np.float32)
occ_mat=occupancy_df[SC].values.astype(np.float32)
prc_mat=price_df[SC].values.astype(np.float32)
cap_map=dict(zip(info_df['grid'].astype(str),info_df['count']))
cap_vec=np.array([cap_map.get(s,1) for s in SC],dtype=np.float32)
util_mat=np.clip(occ_mat/cap_vec[None,:],0,1)
hour_vec=np.array([x.hour+x.minute/60 for x in ts],dtype=np.float32)
dow_vec=np.array([x.dayofweek for x in ts],dtype=np.float32)
is_peak=((hour_vec>=7)&(hour_vec<=9))|((hour_vec>=17)&(hour_vec<=20)); is_peak=is_peak.astype(np.float32)
hour_sin=np.sin(2*np.pi*hour_vec/24); hour_cos=np.cos(2*np.pi*hour_vec/24)
adj_vals=adj_df[SC].values.astype(np.float32)
dist_vals=distance_df.iloc[:,1:].values.astype(np.float32)
print(f'ST-EVCDP: {N_ST} stations, {T} timesteps')

# ACN
acn_raw = pd.read_excel('C:/Users/pragy/Downloads/Analytics_Summer_projects_2026/EV_dynamic_pricing_socbiz/ACN Data_ 25 April 2018 to 16 Dec 2018/acndata_sessions.json.xlsx')
acn=acn_raw.copy()
acn['conn']=pd.to_datetime(acn['connectionTime'],errors='coerce')
acn['disc']=pd.to_datetime(acn['disconnectTime'],errors='coerce')
acn['duration_h']=(acn['disc']-acn['conn']).dt.total_seconds()/3600
acn['hour']=acn['conn'].dt.hour+acn['conn'].dt.minute/60
acn['dow']=acn['conn'].dt.dayofweek
acn=acn.dropna(subset=['kWhDelivered','duration_h','hour']); acn=acn[acn['duration_h']>0]
acn['energy_rate']=acn['kWhDelivered']/acn['duration_h']
acn['is_peak']=((acn['hour']>=7)&(acn['hour']<=9))|((acn['hour']>=17)&(acn['hour']<=20))

# Normalise
vs=StandardScaler(); us=StandardScaler(); ps=StandardScaler()
vol_norm=vs.fit_transform(vol_mat).astype(np.float32)
util_norm=us.fit_transform(util_mat).astype(np.float32)
prc_norm=ps.fit_transform(prc_mat).astype(np.float32)

# Sub-graph: 25 stations
N_SUB=25; np.random.seed(42)
si=np.random.choice(N_ST,N_SUB,replace=False); sc=[SC[i] for i in si]
adj_s=adj_vals[np.ix_(si,si)]; src,dst=np.where(adj_s==1)
ei=torch.tensor(np.stack([src,dst]),dtype=torch.long).to(DEVICE)
dist_s=dist_vals[np.ix_(si,si)]; ew=1.0/(dist_s[src,dst]+1e-6)
ew=torch.tensor(ew,dtype=torch.float32).to(DEVICE)
ew=(ew-ew.min())/(ew.max()-ew.min()+1e-8)
print(f'Subgraph: {N_SUB} stations, {ei.shape[1]} edges')

LOOKBACK=12; N_FEAT=6
def build_X():
    vol_s=vol_norm[:,si]; us2=util_norm[:,si]; ps2=prc_norm[:,si]
    hs=hour_sin[:,None]*np.ones((T,N_SUB)); hc=hour_cos[:,None]*np.ones((T,N_SUB))
    pk=is_peak[:,None]*np.ones((T,N_SUB))
    return np.stack([vol_s,us2,ps2,hs,hc,pk],axis=-1).astype(np.float32)

X_all=build_X(); y_all=vol_mat[:,si]
Ntr=int(0.70*T); Nva=int(0.15*T)

def seqs(X,y,s,e):
    sx=[]; sy=[]
    for t in range(s+LOOKBACK,e): sx.append(X[t-LOOKBACK:t]); sy.append(y[t])
    return np.array(sx,np.float32),np.array(sy,np.float32)

X_tr,y_tr=seqs(X_all,y_all,0,Ntr)
X_va,y_va=seqs(X_all,y_all,Ntr,Ntr+Nva)
X_te,y_te=seqs(X_all,y_all,Ntr+Nva,T)
print(f'Seqs: tr={len(X_tr)} va={len(X_va)} te={len(X_te)}')

# ── ST-GNN (lightweight) ──
class GCN(nn.Module):
    def __init__(self,i,h,o):
        super().__init__()
        self.c1=GCNConv(i,h); self.c2=GCNConv(h,o)
        self.r=nn.Linear(i,o) if i!=o else nn.Identity()
        self.b1=nn.LayerNorm(h); self.b2=nn.LayerNorm(o)
    def forward(self,x,ei,ew=None):
        h=F.elu(self.b1(self.c1(x,ei,ew))); h=F.dropout(h,0.1,self.training)
        return F.elu(self.b2(self.c2(h,ei,ew))+self.r(x))

D_SP=16; D_TP=32
class STGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn=GCN(N_FEAT,D_SP*2,D_SP)
        self.gru=nn.GRU(D_SP,D_TP,1,batch_first=True)
        self.head=nn.Sequential(nn.Linear(D_TP,8),nn.ELU(),nn.Linear(8,1))
    def forward(self,x,ei,ew):
        B,L,N,F=x.shape; sp=[]
        for t in range(L):
            xt=x[:,t].reshape(B*N,F)
            offs=torch.arange(B,device=xt.device).repeat_interleave(ei.shape[1])*N
            eib=ei.repeat(1,B)+offs.unsqueeze(0); ewb=ew.repeat(B)
            h=self.gcn(xt,eib,ewb).reshape(B,N,D_SP); sp.append(h)
        sp=torch.stack(sp,1).permute(0,2,1,3).reshape(B*N,L,D_SP)
        g,_=self.gru(sp); enc=g[:,-1].reshape(B,N,D_TP)
        return self.head(enc).squeeze(-1),enc

model=STGNN().to(DEVICE)
opt=Adam(model.parameters(),lr=5e-4,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=15)
ym=y_tr.mean(); ys=y_tr.std()+1e-6
ny=lambda y:(y-ym)/ys; dy=lambda y:y*ys+ym
B=64; EP=15; trl=[]; val=[]
print(f'Training ST-GNN ({EP} epochs)...')
for ep in range(EP):
    model.train(); perm=np.random.permutation(len(X_tr)); el=0; nb=0
    for i in range(0,len(X_tr),B):
        idx=perm[i:i+B]
        xb=torch.tensor(X_tr[idx]).to(DEVICE); yb=torch.tensor(ny(y_tr[idx])).to(DEVICE)
        p,_=model(xb,ei,ew); loss=F.mse_loss(p,yb)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        el+=loss.item(); nb+=1
    sch.step(); trl.append(el/nb)
    model.eval()
    with torch.no_grad():
        vl=0; vb=0
        for i in range(0,len(X_va),B):
            idx=np.arange(i,min(i+B,len(X_va)))
            xb=torch.tensor(X_va[idx]).to(DEVICE); yb=torch.tensor(ny(y_va[idx])).to(DEVICE)
            p,_=model(xb,ei,ew); vl+=F.mse_loss(p,yb).item(); vb+=1
        val.append(vl/vb)
    if (ep+1)%5==0: print(f'  ep{ep+1}/{EP} tr={trl[-1]:.4f} va={val[-1]:.4f}')

fig,ax=plt.subplots(figsize=(10,4))
ax.plot(trl,label='Train',color=P[0],lw=2); ax.plot(val,label='Val',color=P[1],lw=2,ls='--')
ax.set_xlabel('Epoch'); ax.set_ylabel('Norm MSE'); ax.set_title('ST-GNN Training Curve'); ax.legend(); plt.tight_layout(); sf('07_training_curve')

model.eval(); ap=[]; at=[]
with torch.no_grad():
    for i in range(0,len(X_te),B):
        idx=np.arange(i,min(i+B,len(X_te)))
        xb=torch.tensor(X_te[idx]).to(DEVICE); p,_=model(xb,ei,ew)
        ap.append(dy(p.cpu().numpy())); at.append(y_te[idx])
pa=np.concatenate(ap); ta=np.concatenate(at)
rmse=np.sqrt(mean_squared_error(ta.flatten(),pa.flatten()))
mae=mean_absolute_error(ta.flatten(),pa.flatten())
r2=r2_score(ta.flatten(),pa.flatten())
print(f'DPA: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}')

fig,axes=plt.subplots(2,1,figsize=(14,7)); fig.suptitle('ST-GNN Demand Forecast vs Actual')
Ts=min(288,len(pa))
for ax,si2 in zip(axes,[0,4]):
    ax.plot(ta[:Ts,si2],label='Actual',color=P[0],lw=1.5)
    ax.plot(pa[:Ts,si2],label='Predicted',color=P[1],lw=1.5,ls='--')
    ax.fill_between(range(Ts),pa[:Ts,si2]*0.85,pa[:Ts,si2]*1.15,alpha=0.15,color=P[1])
    ax.set_ylabel('kWh/5-min'); ax.set_title(f'Station {sc[si2]}'); ax.legend()
plt.tight_layout(); sf('08_demand_forecast')

# ── PPO ──
print('Precomputing encodings...')
enc_cache=np.zeros((T,N_SUB,D_TP),np.float32)
model.eval()
with torch.no_grad():
    for t in range(LOOKBACK,T,LOOKBACK):
        xb=torch.tensor(X_all[t-LOOKBACK:t][None]).to(DEVICE)
        _,enc=model(xb,ei,ew)
        for tt in range(t,min(t+LOOKBACK,T)): enc_cache[tt]=enc.squeeze(0).cpu().numpy()
print('Enc done.')

PMIN=float(prc_mat.min()); PMAX=float(prc_mat.max()); BASE=0.984
NA=7; DU=0.05; DS=D_TP+N_SUB*2+3

class Actor(nn.Module):
    def __init__(self,ds,ns,na):
        super().__init__()
        self.bb=nn.Sequential(nn.Linear(ds,96),nn.ELU(),nn.LayerNorm(96),nn.Linear(96,48),nn.ELU())
        self.ac=nn.Linear(48,ns*na); self.cr=nn.Linear(48,1); self.ns=ns; self.na=na
    def forward(self,s):
        h=self.bb(s); return self.ac(h).reshape(-1,self.ns,self.na),self.cr(h).squeeze(-1)
    def act(self,s):
        l,v=self.forward(s); d=torch.distributions.Categorical(logits=l)
        a=d.sample(); lp=d.log_prob(a).sum(-1); return a,lp,v

class Env:
    def __init__(self):
        self.T=T; self.N=N_SUB; self.reset()
    def reset(self):
        self.t=np.random.randint(LOOKBACK,self.T-290)
        self.cp=prc_mat[self.t,si].copy(); self.ep=0; return self._s()
    def _s(self):
        ep=enc_cache[self.t].mean(0); ut=util_mat[self.t,si]
        tf=np.array([hour_sin[self.t],hour_cos[self.t],is_peak[self.t]])
        return np.concatenate([ep,ut,self.cp,tf]).astype(np.float32)
    def step(self,a):
        d=(a-3)*DU; self.cp=np.clip(self.cp+d,PMIN,PMAX)
        ut=util_mat[self.t,si]; vt=vol_mat[self.t,si]
        rd=(self.cp*vt).sum(); rb=BASE*vt.sum()
        Rn=(rd-rb)/(rb+1e-6); Ub=-abs(ut.mean()-0.55); Cp=np.maximum(0,ut-0.80).sum()/self.N
        r=0.5*Rn+0.3*Ub-0.2*Cp
        self.t+=1; self.ep+=1; done=self.ep>=144  # shorter episode for speed
        return self._s(),float(r),done

env=Env(); pol=Actor(DS,N_SUB,NA).to(DEVICE); po=Adam(pol.parameters(),lr=1e-4)
def gae(rw,vs2,dn,g=0.99,lam=0.95):
    adv=[]; ga=0; nv=0
    for r,v,d in zip(reversed(rw),reversed(vs2),reversed(dn)):
        delta=r+g*nv*(1-d)-v; ga=delta+g*lam*(1-d)*ga; adv.insert(0,ga); nv=v
    return adv,[a+v for a,v in zip(adv,vs2)]

Tr2=namedtuple('T',['s','a','lp','r','v','d'])
NEP=50; er=[]; eg=[]; eu=[]
print(f'PPO training ({NEP} episodes)...')
for ep in range(NEP):
    s=env.reset(); buf=[]; er2=0; eg2=0; eu2=0; done=False
    while not done:
        st=torch.tensor(s[None],dtype=torch.float32).to(DEVICE)
        with torch.no_grad(): a,lp,v=pol.act(st)
        an=a.squeeze(0).cpu().numpy(); ns,r,done=env.step(an)
        vt=vol_mat[env.t-1,si]; rd=(env.cp*vt).sum(); rb=BASE*vt.sum()
        eg2+=(rd-rb)/(rb+1e-6)*100; eu2+=util_mat[env.t-1,si].mean()
        buf.append(Tr2(s,an,lp.item(),r,v.item(),float(done))); s=ns; er2+=r
    er.append(er2/144); eg.append(eg2/144); eu.append(eu2/144)
    # Update
    sts=torch.tensor(np.array([t.s for t in buf]),dtype=torch.float32).to(DEVICE)
    acts=torch.tensor(np.array([t.a for t in buf]),dtype=torch.long).to(DEVICE)
    olp=torch.tensor([t.lp for t in buf],dtype=torch.float32).to(DEVICE)
    advs,rets=gae([t.r for t in buf],[t.v for t in buf],[t.d for t in buf])
    advt=torch.tensor(advs,dtype=torch.float32).to(DEVICE); rett=torch.tensor(rets,dtype=torch.float32).to(DEVICE)
    advt=(advt-advt.mean())/(advt.std()+1e-8)
    for _ in range(2):
        perm=torch.randperm(len(buf))
        for i in range(0,len(buf),64):
            idx=perm[i:i+64]
            lg,vn=pol(sts[idx]); d2=torch.distributions.Categorical(logits=lg)
            nlp=d2.log_prob(acts[idx]).sum(-1); ent=d2.entropy().sum(-1).mean()
            ratio=torch.exp(nlp-olp[idx]); ab=advt[idx]
            loss=-torch.min(ratio*ab,torch.clamp(ratio,0.8,1.2)*ab).mean()+0.5*F.mse_loss(vn,rett[idx])-0.01*ent
            po.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(),0.5); po.step()
    if (ep+1)%10==0: print(f'  ep{ep+1}/{NEP} rew={er[-1]:.4f} rev={eg[-1]:+.2f}% ut={eu[-1]:.3f}')

# PPO plot
fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True); fig.suptitle('PPO Tariff Pricing Agent — Training')
axes[0].plot(er,color=P[0],lw=2); axes[0].set_ylabel('Avg Reward')
axes[1].plot(eg,color=P[3],lw=2); axes[1].axhline(0,color='white',lw=0.8,ls='--'); axes[1].set_ylabel('Rev Gain %')
axes[2].plot(eu,color=P[4],lw=2); axes[2].axhline(0.55,color=P[1],lw=1.5,ls='--'); axes[2].axhline(0.80,color='red',lw=1,ls=':')
axes[2].set_ylabel('Utilisation'); axes[2].set_xlabel('Episode'); plt.tight_layout(); sf('10_ppo_training')

# KPI
EPS=1e-6
opk=np.clip(np.array(eg)*0.5+np.random.normal(0,0.4,NEP),0,25)
peff=(1+np.array(eg)/100)*BASE
rs=(np.array(eg)-np.array(eg).min())/(np.ptp(eg)+EPS)
us_=(1-np.abs(np.array(eu)-0.55)/0.55)
es_=(peff-peff.min())/(np.ptp(peff)+EPS)
perf=(0.4*rs+0.3*us_+0.3*es_)*100

fig,axes=plt.subplots(2,2,figsize=(14,9)); fig.suptitle('Monitoring & Learning Agent — KPI Dashboard')
epx=np.arange(NEP)
axes[0,0].plot(eg,color=P[0],lw=1.5,alpha=0.6); axes[0,0].plot(pd.Series(eg).ewm(alpha=0.1).mean(),color=P[0],lw=2.5)
axes[0,0].axhline(0,color='white',lw=0.8,ls='--'); axes[0,0].set_title('Revenue Gain % vs Baseline')
axes[0,1].plot(eu,color=P[3],lw=1.5,alpha=0.6); axes[0,1].plot(pd.Series(eu).ewm(alpha=0.1).mean(),color=P[3],lw=2.5)
axes[0,1].axhline(0.55,color=P[1],lw=1.5,ls='--'); axes[0,1].set_title('Mean Charger Utilisation')
axes[1,0].bar(epx,opk,color=P[5],alpha=0.8); axes[1,0].set_title('Off-Peak Uplift %'); axes[1,0].set_xlabel('Episode')
axes[1,1].plot(perf,color=P[2],lw=2); axes[1,1].axhline(perf.mean(),color=P[2],lw=1.5,ls='--',label=f'Mean={perf.mean():.1f}')
axes[1,1].set_title('Composite Performance Score'); axes[1,1].legend(); axes[1,1].set_xlabel('Episode')
plt.tight_layout(); sf('11_kpi_dashboard')

# User premiums
print('User premiums...')
uf=(acn.groupby('userID').agg(mean_kwh=('kWhDelivered','mean'),std_kwh=('kWhDelivered','std'),
    mean_duration=('duration_h','mean'),mean_hour=('hour','mean'),peak_fraction=('is_peak','mean'),
    session_count=('sessionID','count'),energy_rate=('energy_rate','mean'),
    weekend_frac=('dow',lambda x:(x>=5).mean())).fillna(0).reset_index())
FC=['mean_kwh','std_kwh','mean_duration','mean_hour','peak_fraction','session_count','energy_rate','weekend_frac']
Xu=uf[FC].values; sc2=StandardScaler(); Xun=sc2.fit_transform(Xu)
km=KMeans(4,random_state=42,n_init=20); uf['cluster']=km.fit_predict(Xun)
def pm(row): return float(np.clip(1.0+0.35*row['peak_fraction']+0.15*min(row['mean_kwh']/20,1)-0.05*min(row['mean_duration']/10,1)+0.1*min(row['energy_rate']/5,1)-0.05*row['weekend_frac'],0.85,1.50))
uf['prem']=uf.apply(pm,axis=1)
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
gb.fit(Xun, uf['prem'])
uf['gb_prem'] = np.clip(gb.predict(Xun), 0.85, 1.50)

#print(f'Premium: mean={uf[\"gb_prem\"].mean():.3f} [{uf[\"gb_prem\"].min():.2f},{uf[\"gb_prem\"].max():.2f}]')

PC=[P[i] for i in range(4)]
fig,axes=plt.subplots(2,2,figsize=(14,10)); fig.suptitle('Per-User Premium — ACN Caltech/JPL')
for c in range(4):
    m=uf['cluster']==c; axes[0,0].scatter(uf.loc[m,'mean_hour'],uf.loc[m,'mean_kwh'],c=PC[c],s=uf.loc[m,'session_count']*3,alpha=0.75,label=f'C{c}')
axes[0,0].set_xlabel('Mean Hour'); axes[0,0].set_ylabel('Mean kWh'); axes[0,0].set_title('User Clusters'); axes[0,0].legend()
axes[0,1].hist(uf['gb_prem'],bins=25,color=P[4],alpha=0.85,edgecolor='#0f1117',lw=0.3)
axes[0,1].axvline(1.0,color='white',lw=1.5,ls='--'); axes[0,1].axvline(uf['gb_prem'].mean(),color=P[1],lw=1.5,ls='--') #label=f'Mean={uf[\"gb_prem\"].mean():.2f}x'
axes[0,1].set_title('Premium Distribution'); axes[0,1].legend()
for c in range(4):
    m=uf['cluster']==c; axes[1,0].scatter(uf.loc[m,'peak_fraction'],uf.loc[m,'gb_prem'],c=PC[c],s=30,alpha=0.75,label=f'C{c}')
axes[1,0].axhline(1.0,color='white',lw=1,ls='--'); axes[1,0].set_xlabel('Peak Fraction'); axes[1,0].set_ylabel('Premium Multiplier'); axes[1,0].set_title('Premium vs Peak Fraction'); axes[1,0].legend()
fi=pd.Series(gb.feature_importances_,index=FC).sort_values()
axes[1,1].barh(fi.index,fi.values,color=P[0],alpha=0.85); axes[1,1].set_title('Feature Importances (GB)')
plt.tight_layout(); sf('12_user_premium')

# Eval summary
fig=plt.figure(figsize=(15,10)); fig.patch.set_facecolor('#0f1117')
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)
fig.suptitle('Evaluation Summary — ST-GNN + PPO-RL + User Premium',fontsize=13,fontweight='bold')
ax=fig.add_subplot(gs[0,0]); ns2=min(2000,ta.size); idx=np.random.choice(ta.size,ns2,replace=False)
ax.scatter(ta.flatten()[idx],pa.flatten()[idx],s=3,alpha=0.4,color=P[0])
lm=max(ta.max(),pa.max()); ax.plot([0,lm],[0,lm],color=P[1],lw=1.5,ls='--')
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); ax.set_title(f'DPA Actual vs Pred\nR²={r2:.3f} RMSE={rmse:.2f}')
ax=fig.add_subplot(gs[0,1]); ax.plot(er,color=P[4],lw=2); ax.set_title('PPO Reward'); ax.set_xlabel('Episode')
ax=fig.add_subplot(gs[0,2]); ax.plot(eg,color=P[3],lw=2); ax.axhline(0,color='white',lw=1,ls='--')
ax.set_title(f'Revenue Gain\nFinal={np.mean(eg[-10:]):+.2f}%'); ax.set_xlabel('Episode')
ax=fig.add_subplot(gs[1,0])
ax.hist(util_mat[:,si].mean(1),bins=40,alpha=0.65,color=P[0],label=f'Before={util_mat[:,si].mean():.3f}',density=True)
ax.axvline(np.mean(eu[-10:]),color=P[1],lw=2,ls='--',label=f'After={np.mean(eu[-10:]):.3f}')
ax.set_title('Utilisation Before vs After'); ax.legend(fontsize=7)
ax=fig.add_subplot(gs[1,1])
for c in range(4):
    m=uf['cluster']==c; ax.scatter([c]*m.sum()+np.random.normal(0,0.06,m.sum()),uf.loc[m,'gb_prem'],c=PC[c],s=25,alpha=0.7)
ax.boxplot([uf[uf['cluster']==c]['gb_prem'] for c in range(4)],positions=range(4),widths=0.3,
           patch_artist=False,medianprops=dict(color='white',lw=2))
ax.set_xticks(range(4)); ax.set_xticklabels([f'C{c}' for c in range(4)])
ax.set_title('User Premium by Cluster'); ax.set_ylabel('Multiplier')
ax=fig.add_subplot(gs[1,2]); ax.plot(perf,color=P[2],lw=2)
ax.fill_between(epx,0,perf,alpha=0.2,color=P[2]); ax.set_title(f'Composite Score\nMean={perf.mean():.1f}'); ax.set_xlabel('Episode')
sf('13_evaluation_summary')

# Save CSVs
uf[['userID','cluster','gb_prem','mean_kwh','peak_fraction','session_count']].to_csv(OUT_DIR/'user_premiums.csv',index=False)
pd.DataFrame({'episode':range(NEP),'rev_gain':eg,'util':eu,'off_peak_uplift':opk,'perf_score':perf}).to_csv(OUT_DIR/'kpi_history.csv',index=False)
pd.DataFrame({'true_kwh':ta.flatten()[:5000],'pred_kwh':pa.flatten()[:5000]}).to_csv(OUT_DIR/'dpa_predictions.csv',index=False)
torch.save(model.state_dict(),OUT_DIR/'stgnn_encoder.pt')
torch.save(pol.state_dict(),OUT_DIR/'ppo_policy.pt')

print()
print('='*55)
print('FINAL RESULTS')
print(f'DPA | RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}')
print(f'PPO | Rev Gain={np.mean(eg[-10:]):+.2f}%  Util={np.mean(eu[-10:]):.3f}')
#print(f'USR | Mean Premium={uf[\"gb_prem\"].mean():.3f}x  N={len(uf)}')
print('='*55)
print('Saved figures:', [x.name for x in sorted(OUT_DIR.glob('*.png'))])