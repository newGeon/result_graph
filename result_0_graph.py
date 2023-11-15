import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from tqdm import tqdm


## AUPR & AUC 관련 그래프 생성 함수
def fn_make_graph(json_data, title):

    x = [7, 6, 5, 4, 3, 2, 1, 0]
    x_rev = x[::-1]

    # Line 1
    y1 = []
    y1_upper = []
    y1_lower = []
    
    # Line 2
    y2 = []
    y2_upper = []
    y2_lower = []

    # Line 3
    y3 = []
    y3_upper = []
    y3_lower = []
    
    # Line 4
    y4 = []
    y4_upper = []
    y4_lower = []
    list_kinds = ['Origin', 'GRU-D', 'O-SNR100', 'G-SNR100']

    for i in range(8):
        y1.append(json_data['Origin'][i]['horizon_{}'.format(i)][1])
        y1_upper.append(json_data['Origin'][i]['horizon_{}'.format(i)][3])
        y1_lower.append(json_data['Origin'][i]['horizon_{}'.format(i)][2])

        y2.append(json_data['GRU-D'][i]['horizon_{}'.format(i)][1])
        y2_upper.append(json_data['GRU-D'][i]['horizon_{}'.format(i)][3])
        y2_lower.append(json_data['GRU-D'][i]['horizon_{}'.format(i)][2])

        y3.append(json_data['O-SNR100'][i]['horizon_{}'.format(i)][1])
        y3_upper.append(json_data['O-SNR100'][i]['horizon_{}'.format(i)][3])
        y3_lower.append(json_data['O-SNR100'][i]['horizon_{}'.format(i)][2])

        y4.append(json_data['G-SNR100'][i]['horizon_{}'.format(i)][1])
        y4_upper.append(json_data['G-SNR100'][i]['horizon_{}'.format(i)][3])
        y4_lower.append(json_data['G-SNR100'][i]['horizon_{}'.format(i)][2])

    y1_lower = y1_lower[::-1]
    y2_lower = y2_lower[::-1]
    y3_lower = y3_lower[::-1]
    y4_lower = y4_lower[::-1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y1_upper+y1_lower,
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Origin',
    ))
    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y2_upper+y2_lower,
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line_color='rgba(255,255,255,0)',
        name='GRU-D',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y3_upper+y3_lower,
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Origin + SNR100',
    ))
    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y4_upper+y4_lower,
        fill='toself',
        fillcolor='rgba(240,240,0,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='GRU-D + SNR100',
    ))    
    fig.add_trace(go.Scatter(
        x=x, y=y1,
        line_color='rgb(0,100,80)',
        name='Origin',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y2,
        line_color='rgb(0,176,246)',
        name='GRU-D',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y3,
        line_color='rgb(231,107,243)',
        name='Origin + SNR100',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y4,
        line_color='rgb(255,255,0)',
        name='GRU-D + SNR100',
    ))

    fig.update_traces(mode='lines')
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(yaxis=dict({"title": title}))
    fig.update_layout(xaxis=dict({"tickvals":[0,1,2,3,4,5,6,7],
                                  "ticktext":["7","6","5","4","3","2","1","0"],
                                  "title": "Hours before Sepsis Onset"
    }))
    fig.show()
    print('--------------------------------------------------------')


if __name__ == '__main__':
    print('-- 실험 결과 그래프 생성 코드 ----------------------------------------------------------')
    df_data = pd.read_csv('./result/experiment_result.csv')

    list_kinds = ['Origin', 'GRU-D', 'O-SNR100', 'G-SNR100']
    list_va = ['va_prc', 'va_auc']

    result_dict = dict()
    
    for va in list_va:
        one_dict = dict()         
        for kind in list_kinds:        
            
            list_results = []

            ex_name = kind + '_' + va

            for horizon in range(8):  
                df_horizon = df_data[df_data['Horizon'] == horizon]
                df_vals = df_horizon[[ex_name]].dropna().values
                
                ex_median = np.median(df_vals)

                clean_vals = [x for x in df_vals if x >= ex_median]

                ex_mean = np.mean(clean_vals)
                ex_median = np.median(clean_vals)
                ex_min = np.min(clean_vals)
                ex_max = np.max(clean_vals)

                list_results.append({'horizon_{}'.format(horizon) : [ex_mean, ex_median, ex_min, ex_max]})

            one_dict[kind] = list_results
        result_dict[va] = one_dict

    fn_make_graph(result_dict['va_prc'], 'AUPR')
    fn_make_graph(result_dict['va_auc'], 'AUC')

    print('--------------------------------------------')
