import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

st.set_page_config(layout='wide', page_title='Formula Optimizer V2')
st.title('Formula Optimizer V2 - Bayesian DOE')

if 'trials' not in st.session_state:
    st.session_state.trials=[]

nvar=st.sidebar.number_input('Numero variabili',2,15,4)
ndoe=st.sidebar.number_input('Numero prove iniziali',4,30,6)

vars=[]
st.subheader('Variabili')
for i in range(nvar):
    c1,c2,c3,c4,c5=st.columns(5)
    vars.append({
        'name':c1.text_input(f'N{i}',f'V{i+1}'),
        'min':c2.number_input(f'Min{i}',value=0.0),
        'max':c3.number_input(f'Max{i}',value=10.0),
        'step':c4.number_input(f'Step{i}',value=0.5,min_value=0.001),
        'base':c5.number_input(f'Base{i}',value=5.0)
    })

def lhs(n,p):
    cols=[]
    for _ in range(p):
        a=[(i+random.random())/n for i in range(n)]
        random.shuffle(a)
        cols.append(a)
    return list(zip(*cols))

if st.button('Genera DOE iniziale'):
    rows=[]
    for idx,s in enumerate(lhs(ndoe,nvar),1):
        d={'ID':idx}
        for j,v in enumerate(vars):
            x=v['min']+s[j]*(v['max']-v['min'])
            x=round(round(x/v['step'])*v['step'],4)
            d[v['name']]=x
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
        X=scored[names].values
        y=scored['Score'].astype(float).values

        kernel=ConstantKernel(1.0)*(RBF(length_scale=np.ones(X.shape[1])))
        gp=GaussianProcessRegressor(kernel=kernel,alpha=1e-6,normalize_y=True,random_state=1)
        gp.fit(X,y)

        st.subheader('Sensibilità variabili (proxy)')
        base=np.mean(X,axis=0)
        sens=[]
        for i,n in enumerate(names):
            test=base.copy()
            p1=test.copy(); p2=test.copy()
            p1[i]=min(p1[i]+1,np.max(X[:,i]))
            p2[i]=max(p2[i]-1,np.min(X[:,i]))
            imp=abs(gp.predict([p1])[0]-gp.predict([p2])[0])
            sens.append([n,imp])
        sdf=pd.DataFrame(sens,columns=['Variabile','Indice'])
        st.dataframe(sdf.sort_values('Indice',ascending=False),use_container_width=True)

        if st.button('Suggerisci 3 nuove prove (Bayesian)'):
            best=np.max(y)
            cand=[]
            existing=set(tuple(r) for r in X.tolist())

            for _ in range(3000):
                row=[]
                for v in vars:
                    x=random.uniform(v['min'],v['max'])
                    x=round(round(x/v['step'])*v['step'],4)
                    row.append(x)

                if tuple(row) in existing:
                    continue

                mu,std=gp.predict([row],return_std=True)
                mu=float(mu[0]); std=float(std[0])

                ei=(mu-best)+1.5*std
                cand.append((ei,mu,std,row))

            cand=sorted(cand,key=lambda x:x[0],reverse=True)[:3]
            out=[]
            for k,c in enumerate(cand,1):
                d={'Prova':k,'Expected Improvement':round(c[0],2),'Score Previsto':round(c[1],2),'Incertezza':round(c[2],2)}
                for i,n in enumerate(names):
                    d[n]=c[3][i]
                out.append(d)
            st.subheader('Nuove prove consigliate')
            st.dataframe(pd.DataFrame(out),use_container_width=True)
