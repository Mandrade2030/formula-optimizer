import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from math import sqrt

st.set_page_config(layout='wide',page_title='Formula Optimizer V3')
st.title('Formula Optimizer V3 - Bayesian DOE + Diversity')

if 'trials' not in st.session_state:
    st.session_state.trials=[]

nvar=st.sidebar.number_input('Numero variabili',2,15,4)
ndoe=st.sidebar.number_input('Numero prove iniziali',4,30,6)

vars=[]
for i in range(nvar):
    c1,c2,c3,c4=st.columns(4)
    vars.append({
        'name':c1.text_input(f'n{i}',f'V{i+1}'),
        'min':c2.number_input(f'min{i}',value=0.0,key=f'min{i}'),
        'max':c3.number_input(f'max{i}',value=10.0,key=f'max{i}'),
        'step':c4.number_input(f'step{i}',value=0.5,min_value=0.001,key=f'step{i}')})

def lhs(n,p):
    cols=[]
    for _ in range(p):
        x=[(i+random.random())/n for i in range(n)]
        random.shuffle(x)
        cols.append(x)
    return list(zip(*cols))

if st.button('Genera DOE iniziale'):
    rows=[]
    for idx,row in enumerate(lhs(ndoe,nvar),1):
        d={'ID':idx}
        for j,v in enumerate(vars):
            val=v['min']+row[j]*(v['max']-v['min'])
            val=round(round(val/v['step'])*v['step'],4)
            d[v['name']]=val
        d['Score']=np.nan
        rows.append(d)
    st.session_state.trials=rows

if st.session_state.trials:
    df=pd.DataFrame(st.session_state.trials)
    edited=st.data_editor(df,use_container_width=True)
    st.session_state.trials=edited.to_dict('records')

    scored=edited.dropna(subset=['Score'])

    if len(scored)>=5:
        names=[v['name'] for v in vars]
        X=scored[names].values.astype(float)
        y=scored['Score'].values.astype(float)

        gp=GaussianProcessRegressor(
            kernel=ConstantKernel(1.0)*RBF(length_scale=np.ones(len(names))),
            normalize_y=True,
            alpha=1e-5,
            random_state=1)
        gp.fit(X,y)

        st.subheader('Ranking variabili')
        sens=[]
        center=np.mean(X,axis=0)
        for i,n in enumerate(names):
            p1=center.copy(); p2=center.copy()
            p1[i]+=0.5; p2[i]-=0.5
            sens.append([n,abs(float(gp.predict([p1])[0]-gp.predict([p2])[0]))])
        st.dataframe(pd.DataFrame(sens,columns=['Variabile','Indice']).sort_values('Indice',ascending=False),use_container_width=True)

        if st.button('Suggerisci 3 prove V3'):
            existing=[list(r) for r in X.tolist()]
            candidates=[]

            for _ in range(6000):
                row=[]
                for v in vars:
                    x=random.uniform(v['min'],v['max'])
                    x=round(round(x/v['step'])*v['step'],4)
                    row.append(x)

                mu,std=gp.predict([row],return_std=True)
                mu=float(mu[0]); std=float(std[0])
                best=max(y)
                ei=(mu-best)+2.0*std

                min_dist=999999
                for e in existing:
                    d=sqrt(sum((a-b)**2 for a,b in zip(row,e)))
                    min_dist=min(min_dist,d)

                diversity=min_dist
                score=ei + 0.15*diversity
                candidates.append((score,ei,mu,std,diversity,row))

            candidates=sorted(candidates,key=lambda x:x[0],reverse=True)

            selected=[]
            for c in candidates:
                ok=True
                for s in selected:
                    d=sqrt(sum((a-b)**2 for a,b in zip(c[5],s[5])))
                    if d<2.0:
                        ok=False
                        break
                if ok:
                    selected.append(c)
                if len(selected)==3:
                    break

            out=[]
            for i,s in enumerate(selected,1):
                r={'Prova':i,'EI':round(s[1],2),'Score Previsto':round(s[2],2),'Incertezza':round(s[3],2),'Diversita':round(s[4],2)}
                for j,n in enumerate(names):
                    r[n]=s[5][j]
                out.append(r)

            st.subheader('Nuove prove consigliate')
            st.dataframe(pd.DataFrame(out),use_container_width=True)
