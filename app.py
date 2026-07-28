import streamlit as st
import pandas as pd
import numpy as np
import random, json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from math import sqrt

st.set_page_config(layout='wide', page_title='Formula Optimizer Lab Edition')
st.title('Formula Optimizer - Laboratory Edition')

if 'trials' not in st.session_state:
    st.session_state.trials=[]

nvar = st.sidebar.number_input('Numero variabili',2,20,6)
ndoe = st.sidebar.number_input('Numero prove iniziali',3,30,8)
nsuggest = st.sidebar.number_input('Nuove prove suggerite',1,10,3)

st.subheader('Definizione variabili')
vars=[]
for i in range(nvar):
    c1,c2,c3,c4,c5,c6=st.columns(6)
    vars.append({
        'name': c1.text_input(f'n{i}',f'RM{i+1}'),
        'base': c2.number_input(f'base{i}',value=10.0,key=f'b{i}'),
        'min': c3.number_input(f'min{i}',value=5.0,key=f'mn{i}'),
        'max': c4.number_input(f'max{i}',value=20.0,key=f'mx{i}'),
        'step': c5.number_input(f'step{i}',value=0.5,min_value=0.001,key=f's{i}'),
        'locked': c6.checkbox('Lock',key=f'l{i}')
    })

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
        rec={'ID':idx,'Iterazione':0}
        for j,v in enumerate(vars):
            span=(v['max']-v['min'])*0.35
            center=v['base']
            val=center-span + row[j]*(2*span)
            val=max(v['min'],min(v['max'],val))
            val=round(round(val/v['step'])*v['step'],4)
            rec[v['name']]=val
        rec['Score']=np.nan
        rows.append(rec)
    st.session_state.trials=rows

if st.session_state.trials:
    df=pd.DataFrame(st.session_state.trials)
    edited=st.data_editor(df,use_container_width=True)
    st.session_state.trials=edited.to_dict('records')

    scored=edited.dropna(subset=['Score'])

    if len(scored)>0:
        best=scored.sort_values('Score',ascending=False).iloc[0]
        st.subheader('Miglior formulazione corrente')
        st.write(f"Score migliore: {best['Score']}")

    if len(scored)>=5:
        names=[v['name'] for v in vars]
        X=scored[names].values.astype(float)
        y=scored['Score'].astype(float).values

        gp=GaussianProcessRegressor(
            kernel=ConstantKernel(1.0)*RBF(length_scale=np.ones(len(names))),
            normalize_y=True,
            alpha=1e-5,
            random_state=1)
        gp.fit(X,y)

        st.subheader('Importanza variabili (proxy)')
        center=np.mean(X,axis=0)
        imp=[]
        for i,n in enumerate(names):
            p1=center.copy(); p2=center.copy()
            p1[i]+=0.5; p2[i]-=0.5
            delta=abs(float(gp.predict([p1])[0]-gp.predict([p2])[0]))
            imp.append([n,delta])
        st.dataframe(pd.DataFrame(imp,columns=['Variabile','Indice']).sort_values('Indice',ascending=False),use_container_width=True)

        st.line_chart(scored[['Score']])

        if st.button('Genera nuove prove'):
            existing=[list(r) for r in X.tolist()]
            cand=[]
            besty=max(y)
            current_iter=int(max(edited['Iterazione']))+1

            for _ in range(8000):
                row=[]
                for idx,v in enumerate(vars):
                    if v['locked']:
                        row.append(float(best[v['name']]))
                    else:
                        x=random.uniform(v['min'],v['max'])
                        x=round(round(x/v['step'])*v['step'],4)
                        row.append(x)

                mu,std=gp.predict([row],return_std=True)
                mu=float(mu[0]); std=float(std[0])
                ei=(mu-besty)+2.0*std

                dist=min(sqrt(sum((a-b)**2 for a,b in zip(row,e))) for e in existing)
                score=ei+0.15*dist
                cand.append((score,ei,mu,std,dist,row))

            cand=sorted(cand,key=lambda z:z[0],reverse=True)
            selected=[]
            for c in cand:
                ok=True
                for s in selected:
                    d=sqrt(sum((a-b)**2 for a,b in zip(c[5],s[5])))
                    if d<2:
                        ok=False
                if ok:
                    selected.append(c)
                if len(selected)>=nsuggest:
                    break

            nextid=int(max(edited['ID']))+1
            for s in selected:
                rec={'ID':nextid,'Iterazione':current_iter}
                for i,n in enumerate(names):
                    rec[n]=s[5][i]
                rec['Score']=np.nan
                nextid+=1
                st.session_state.trials.append(rec)
            st.rerun()
