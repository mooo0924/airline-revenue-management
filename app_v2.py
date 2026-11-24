from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from revenue import *
st.set_page_config(page_title='航空收益管理決策系統',layout='wide', initial_sidebar_state='expanded')
st.title('TPE → NRT｜航空收益管理決策系統')
st.caption('機器學習輔助的單階段定價情境模擬｜非實際營收、非訂位庫存預測')
@st.cache_data
def load():
    return pd.read_csv(Path(__file__).parent/'data/航班基礎資料.csv')
base=load(); st.sidebar.header('資料來源與設定')
up=st.sidebar.file_uploader('匯入最新版 XGBoost 預測票價 CSV',type=['csv'])
bundled=Path(__file__).parent/'data'/'XGBoost逐筆預測票價.csv'
if up is not None or bundled.is_file():
    try: data=attach_predictions(base,pd.read_csv(up if up is not None else bundled)); price_col=PRED
    except Exception as exc: st.error(str(exc)); st.stop()
    st.success(f'已對接 {len(data):,} / {len(base):,} 筆預測報價。未匯入者不納入比較。')
else:
    data=base;price_col='歷史報價';st.warning('目前使用歷史公開報價作為暫時基準，不是 XGBoost 預測票價。匯入預測檔後才是完整模型整合。')
dtds=sorted(data.DTD.astype(int).unique().tolist());dtd=st.sidebar.selectbox('整體策略比較與敏感度分析：固定 DTD（天）',dtds,index=dtds.index(30) if 30 in dtds else 0)
demand_col=st.sidebar.radio('需求模型',['XGB需求','RF需求'])
e=st.sidebar.slider('定價用價格彈性',-2.5,-.5,-1.46,.01)
band=st.sidebar.slider('允許調價幅度 (%)',5,40,20,5)/100
f=flight_sample(data,dtd,price_col,demand_col)
if f.empty: st.error('無有效航班');st.stop()
st.caption(f'整體策略比較與敏感度分析：固定 DTD={dtd}，共 {len(f)} 個不重複航班。個別航班可獨立選擇 DTD。')
t1,t2,t3=st.tabs(['個別航班最佳化','整體策略比較','敏感度分析'])
fixed=strategies(f,e,band)
with t1:
    st.caption('先選航班，再選該航班實際有資料的 DTD；不受側邊欄整體比較 DTD 限制。')
    a=st.selectbox('航空公司',sorted(data['航空公司'].unique()))
    g=data[data['航空公司']==a];dt=st.selectbox('出發日期',sorted(g['出發日期'].unique()));g=g[g['出發日期']==dt]
    no=st.selectbox('航班號碼',sorted(g['航班號碼'].unique()));g=g[g['航班號碼']==no]
    available=sorted(g['DTD'].astype(int).unique().tolist())
    chosen=st.selectbox('個別航班距起飛天數 DTD（天）',available,index=min(range(len(available)),key=lambda i:abs(available[i]-30)))
    single=flight_sample(g,chosen,price_col,demand_col)
    if single.empty: st.error('該航班及 DTD 無有效價格、需求或座位數。');st.stop()
    r=single.iloc[0]
    p0,q0,s=float(r.P0),float(r.Q0),float(r.S)
    st.write(f'基準票價 NT${p0:,.0f}　｜　基準需求 {q0:.1f} 人　｜　座位 {s:.0f} 席　｜　{r.時段}')
    if q0>s: st.info('基準估計需求超過容量：僅作為情境需求指標，不代表已售座位或已證實的未滿足需求。')
    ratio=st.slider('模擬票價（基準百分比）',int(round((1-band)*100)),int(round((1+band)*100)),100,1)
    m=st.slider('需求情境調整 (%)',80,120,100,5)/100
    p=p0*ratio/100; q,sold,rev=evaluate(p,p0,q0,s,e,m);best=float(optimal(p0,q0*m,s,e,band));_,bs,br=evaluate(best,p0,q0,s,e,m);_,_,base_rev=evaluate(p0,p0,q0,s,e,m)
    c=st.columns(4);c[0].metric('模擬票價',f'NT${p:,.0f}');c[1].metric('估計需求',f'{q:.1f} 人');c[2].metric('容量限制後載客率',f'{sold/s:.1%}');c[3].metric('票價收入估計',f'NT${rev:,.0f}')
    x=np.unique(np.r_[np.linspace(p0*(1-band),p0*(1+band),301),best,p]);qq,ss,rr=evaluate(x,p0,q0,s,e,m)
    fig=go.Figure();fig.add_scatter(x=x,y=rr,mode='lines',name='票價收入');fig.add_scatter(x=[p],y=[float(rev)],mode='markers',marker_size=12,name='目前定價');fig.add_scatter(x=[best],y=[float(br)],mode='markers',marker_size=12,name='情境最佳定價');fig.update_layout(xaxis_title='票價 TWD',yaxis_title='票價收入估計 TWD');st.plotly_chart(fig,use_container_width=True)
    fig=go.Figure();fig.add_scatter(x=x,y=qq,name='情境需求');fig.add_scatter(x=x,y=ss,name='容量限制後售票量');fig.add_hline(y=s,line_dash='dash',annotation_text='座位容量');fig.update_layout(xaxis_title='票價 TWD',yaxis_title='人數');st.plotly_chart(fig,use_container_width=True)
    c=st.columns(3);c[0].metric('情境最佳票價',f'NT${best:,.0f}');c[1].metric('最高情境收入',f'NT${br:,.0f}');c[2].metric('相對基準收入增幅',f'{(br/base_rev-1):.2%}' if base_rev>0 else '不適用')
    st.caption('單班最佳票價隨情境重新計算，是已知情境下的理論結果。穩健性請見敏感度頁。')
with t2:
    st.write('A：維持基準報價；B：預估載客率低於80%降價5%、高於90%漲價5%；C：在調價範圍內最佳化。')
    comp=compare(fixed,e);st.dataframe(comp.round(2),hide_index=True,use_container_width=True)
    fig=go.Figure(go.Bar(x=comp['策略'],y=comp['票價收入估計']));fig.update_layout(yaxis_title='票價收入估計 TWD');st.plotly_chart(fig,use_container_width=True)
    st.download_button('下載策略比較 CSV',comp.to_csv(index=False).encode('utf-8-sig'),'策略比較.csv','text/csv')
    st.download_button('下載逐班決策 CSV',fixed.to_csv(index=False).encode('utf-8-sig'),'逐班定價.csv','text/csv')
    st.caption('同一 DTD 下每個出發航班只計算一次。最佳化策略在設計情境中優於其他策略屬數學結果，不能證明真實營收提升。')
with t3:
    st.write('固定上頁 A/B/C 的定價決策，在其他需求及價格彈性情境下測試；不在每個情境重新求最佳價格。')
    rows=[]
    for m0 in [.9,.95,1,1.05,1.1]:
        for ee in [-.5,-1,-1.46,-2,-2.5]:
            cc=compare(fixed,ee,m0);cc['需求倍率']=m0;cc['實際情境彈性']=ee;rows.append(cc)
    stress=pd.concat(rows,ignore_index=True);mat=stress[stress['策略']=='C'].pivot(index='需求倍率',columns='實際情境彈性',values='相對A收入變化(%)')
    fig=go.Figure(go.Heatmap(z=mat.values,x=[str(x) for x in mat.columns],y=[str(x) for x in mat.index],colorscale='RdBu',zmid=0,text=np.round(mat.values,2),texttemplate='%{text:.2f}%'));fig.update_layout(xaxis_title='價格彈性情境',yaxis_title='需求倍率',title='固定最佳化策略 C 相較 A 的收入變化 (%)');st.plotly_chart(fig,use_container_width=True)
    st.dataframe(stress.round(2),hide_index=True,use_container_width=True);st.download_button('下載敏感度分析 CSV',stress.to_csv(index=False).encode('utf-8-sig'),'敏感度分析.csv','text/csv')
    st.caption('情境倍率與其他彈性值為壓力測試假設，不是統計信賴區間；-1.46 為市場層級文獻參考，非實測航班彈性。')
st.divider();st.caption('研究限制：價格為公開報價或其預測值；假設整班基準需求對應該基準票價，無法以成交量驗證。未建構訂位曲線、剩餘庫存、艙等、成本或競爭反應；本系統為單階段情境，不可將不同 DTD 的收入相加。')
