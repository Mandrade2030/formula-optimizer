# Formula Optimizer Mixture Laboratory V1
# Streamlit prototype
import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

st.set_page_config(layout="wide")
st.title("Formula Optimizer - Mixture Laboratory V1")

if "trials" not in st.session_state:
    st.session_state.trials=[]

nvar=st.sidebar.number_input("Numero variabili",2,15,5)
ndoe=st.sidebar.number_input("DOE iniziali",3,30,8)
nsuggest=st.sidebar.number_input("Prove suggerite",1,10,3)

vars=[]
st.subheader("Variabili (somma obbligatoria = 100%)")
for i in range(nvar):
    c1,c2,c3,c4,c5,c6=st.columns(6)
    vars.append({
        'name':c1.text_input(f'n{i}',f'RM{i+1}'),
        'base':c2.number_input(f'base{i}',value=20.0,key=f'b{i}'),
        'min':c3.number_input(f'min{i}',value=0.0,key=f'mn{i}'),
        'max':c4.number_input(f'max{i}',value=100.0,key=f'mx{i}'),
        'step':c5.number_input(f'step{i}',value=0.5,key=f's{i}'),
        'lock':c6.checkbox('Lock',key=f'l{i}')
    })

def norm100(v):
    s=sum(v)
    return [round(x*100/s,2) for x in v]

if st.button('Genera DOE Mixture'):
    rows=[]
    for k in range(ndoe):
        vals=[]
        for v in vars:
            span=(v['max']-v['min'])*0.10
            vals.append(max(v['min'],min(v['max'],v['base']+random.uniform(-span,span))))
        vals=norm100(vals)
        r={'ID':k+1,'Iterazione':0}
        for i,v in enumerate(vars):
            r[v['name']]=vals[i]
        r['Totale']=round(sum(vals),2)
        r['Score']=np.nan
        rows.append(r)
    st.session_state.trials=rows

if st.session_state.trials:
    df=pd.DataFrame(st.session_state.trials)
    df=st.data_editor(df,use_container_width=True)
    st.session_state.trials=df.to_dict('records')

    scored=df.dropna(subset=['Score'])
    if len(scored)>0:
        best=scored.sort_values('Score',ascending=False).iloc[0]
        st.success(f'Miglior score: {best["Score"]}')

    if len(scored)>=5:
        names=[v['name'] for v in vars]
        X=scored[names].values
        y=scored['Score'].astype(float).values

        gp=GaussianProcessRegressor(kernel=ConstantKernel(1.0)*RBF(),normalize_y=True)
        gp.fit(X,y)

        if st.button('Genera nuove prove'):
            besty=max(y)
            cand=[]
            for _ in range(5000):
                vals=[]
                for v in vars:
                    if v['lock']:
                        vals.append(float(best[v['name']]))
                    else:
                        vals.append(random.uniform(v['min'],v['max']))
                vals=norm100(vals)
                mu,std=gp.predict([vals],return_std=True)
                ei=float(mu[0]-besty+1.5*std[0])
                cand.append((ei,vals))
            cand=sorted(cand,key=lambda x:x[0],reverse=True)[:nsuggest]
            out=[]
            for i,c in enumerate(cand,1):
                r={'Prova':i,'EI':round(c[0],2)}
                for j,n in enumerate(names):
                    r[n]=round(c[1][j],2)
                r['Totale']=round(sum(c[1]),2)
                out.append(r)
            st.dataframe(pd.DataFrame(out),use_container_width=True)
