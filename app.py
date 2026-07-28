import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random

st.set_page_config(page_title="Formula Optimizer DOE", layout="wide")
st.title("Formula Optimizer DOE - Web Prototype")

if 'trials' not in st.session_state:
    st.session_state.trials=[]

nvar=st.sidebar.number_input('Numero variabili',2,15,6)
ndoe=st.sidebar.number_input('Numero prove iniziali',4,30,8)

st.subheader('Definizione variabili')
vars=[]
cols=st.columns(5)
headers=['Nome','Base','Min','Max','Passo']
for c,h in zip(cols,headers):
    c.write('**'+h+'**')

for i in range(nvar):
    c1,c2,c3,c4,c5=st.columns(5)
    vars.append({
        'name':c1.text_input(f'n{i}',f'V{i+1}',key=f'n{i}'),
        'base':c2.number_input(f'b{i}',value=10.0,key=f'b{i}'),
        'min':c3.number_input(f'mn{i}',value=5.0,key=f'mn{i}'),
        'max':c4.number_input(f'mx{i}',value=20.0,key=f'mx{i}'),
        'step':c5.number_input(f's{i}',value=0.5,key=f's{i}')})

def lhs(n,p):
    cols=[]
    for _ in range(p):
        x=[(i+random.random())/n for i in range(n)]
        random.shuffle(x)
        cols.append(x)
    return list(zip(*cols))

if st.button('Genera DOE'):
    trials=[]
    for idx,row in enumerate(lhs(ndoe,len(vars))):
        vals=[]
        for j,v in enumerate(vars):
            x=v['min']+row[j]*(v['max']-v['min'])
            x=round(round(x/v['step'])*v['step'],4)
            vals.append(x)
        trials.append({'ID':idx+1,**{vars[k]['name']:vals[k] for k in range(len(vars))},'Score':None})
    st.session_state.trials=trials

if st.session_state.trials:
    st.subheader('Prove')
    df=pd.DataFrame(st.session_state.trials)
    edited=st.data_editor(df,use_container_width=True)
    st.session_state.trials=edited.to_dict('records')

    scored=edited.dropna(subset=['Score'])

    if len(scored)>=5:
        X=scored[[v['name'] for v in vars]].values
        y=scored['Score'].values
        rf=RandomForestRegressor(n_estimators=300,random_state=1)
        rf.fit(X,y)

        st.subheader('Importanza variabili')
        imp=pd.DataFrame({'Variabile':[v['name'] for v in vars],'Importanza %':rf.feature_importances_*100})
        st.dataframe(imp.sort_values('Importanza %',ascending=False),use_container_width=True)

        if st.button('Suggerisci 3 nuove prove'):
            candidates=[]
            for _ in range(1000):
                row=[]
                for v in vars:
                    x=random.uniform(v['min'],v['max'])
                    x=round(round(x/v['step'])*v['step'],4)
                    row.append(x)
                pred=rf.predict([row])[0]
                candidates.append((pred,row))
            candidates=sorted(candidates,key=lambda z:z[0],reverse=True)[:3]

            st.subheader('Nuove prove suggerite')
            out=[]
            for i,(pred,row) in enumerate(candidates,1):
                d={'Prova':i,'Score Previsto':round(float(pred),2)}
                for k,v in enumerate(vars):
                    d[v['name']]=row[k]
                out.append(d)
            st.dataframe(pd.DataFrame(out),use_container_width=True)
