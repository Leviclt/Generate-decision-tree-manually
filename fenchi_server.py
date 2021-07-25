import os
import random
import pandas as pd
import numpy as np

import graphviz
from jinja2 import Template
from sklearn import tree


COLOR_PALLETE = ["#006400", "#0d6c0d", "#1a741a", "#277c27", "#358435", "#408b40", "#4d934d", "#5a9b5a", "#68a368", "#75ab75", "#80b280", "#8dba8d", "#9bc29b", "#a8caa8",
                 "#b5d2b5", "#c2dac2", "#cce0cc", "#d9e8d9", "#e6f0e6", "#f3f8f3"]	

def path_join(*args):
    the_original_path = os.path.join(*args) 
    return the_original_path.replace('\\','/')           

def calculate_gini(pd):
    # Gini is an indicator used to measure node purity. The smaller the Gini value, the purer the sample
    return 1 - pd**2 - (1-pd)**2

def calculate_entropy(pd):
    return -(pd*np.log2(pd) + (1-pd)*np.log2(1-pd) )

def calculate_mse(y):
    average = y.mean()
    return np.mean((y - average)**2)

def init_zero_node_info(model_type, Y,  criterion):
    if model_type == 'PD':
        if criterion == 'gini':
            impurity = calculate_gini(Y.mean())
        elif criterion == 'entropy':
            impurity = calculate_entropy(Y.mean())
        else:
            raise("must be gini or entropy")

        init_dict = {
                    'node_id': 0,
                    'n': len(Y),
                    'impurity': round(impurity, 3),
                    'event': sum(Y),                                      # eg: bad sample number
                    'event_pct': f"({'{:.1%}'.format(sum(Y)/len(Y))})",   # eg: bad sample ratio
                    'pd': Y.mean(),                                       # eg: bad sample ratio
                    'pd3digit': round(Y.mean(), 3),                       # eg: bad sample ratio
                    'node_desc': '',          
		           }
    return init_dict
    

# The feature satisfies the sample index corresponding to the split value
def parse_condition_str(condition_str, df):     
    # eg: condition_str ='【v22 <= 70】【v21 <= 60】'
    condition_list = condition_str.replace('【', '').split('】')   
    var_list = []      # eg: v22
    sign_list = []     # eg: <=
    value_list = []    # eg: 70
    for element in condition_list:
        if len(element) > 0:
            element_list = element.split(' ')
            var_list.append(element_list[0])            
            sign_list.append(element_list[1])           
            value_list.append(float(element_list[2]))   
    indicator_list = []
    for idx in range(len(var_list)):
        if sign_list[idx] == '<=':
            indicator_sub = np.where(df[var_list[idx]]<=value_list[idx], 1, 0) # eg: <=70 fill 1 else 0
        else:
            indicator_sub = np.where(df[var_list[idx]]>value_list[idx], 1, 0)  # eg: >70 fill 1 else 0
        indicator_list.append(indicator_sub)
    indicator = np.where(sum(indicator_list)==len(var_list), 1, 0)  

    return indicator 	

def calculate_feature_importance(model_type, X, y, criterion):
    if model_type == 'PD':
        tmp_tree = tree.DecisionTreeClassifier(criterion=criterion, min_samples_leaf=0.05, max_depth=10) 
    else:
        tmp_tree = tree.DecisionTreeRegressor(criterion=criterion, min_samples_leaf=0.05, max_depth=10)
    tmp_tree = tmp_tree.fit(X,y)
    feature_importance = tmp_tree.feature_importances_
    feature_importance_df = pd.DataFrame({
                                        'feature': X.columns,
                                        'importance': feature_importance,
                                        'min':X.apply(min),
                                        'p25':X.apply(lambda x: np.quantile(x, 0.25)),
                                        'p50':X.apply(lambda x: np.quantile(x, 0.5)),										   
                                        'p75':X.apply(lambda x: np.quantile(x, 0.75)),	
                                        'max':X.apply(max)}).sort_values('importance', ascending=False)                                       
    return feature_importance_df

def calculate_manual_split(model_type, split_var, split_value, df, y_label, criterion, node_id, node_desc, max_node_id, sample_tot, event_tot):
    left_n = sum(df[split_var]<=split_value)
    right_n = sum(df[split_var]>split_value)

    left_node_id = max_node_id + 1 
    right_node_id = max_node_id + 2
    
    if model_type == 'PD':
        left_event = sum( (df[split_var]<=split_value)&(df[y_label]==1) )
        right_event = sum( (df[split_var]>split_value)&(df[y_label]==1) )
        left_pd = left_event/left_n
        right_pd = right_event/right_n
               
        if criterion == 'gini':
            left_impurity = calculate_gini(left_pd)
            right_impurity = calculate_gini(right_pd)
        elif criterion == 'entropy':
            left_impurity = calculate_entropy(left_pd)
            right_impurity = calculate_entropy(right_pd)
        else:
            raise('必须为gini或者entropy')

        left_child_info = {
            'node_id': left_node_id,
            'impurity': round(left_impurity, 3),
            'n': left_n,
            'pct': f"({ '{:.1%}'.format(left_n/sample_tot)})" if left_node_id<=2 else f"({'{:.1%}'.format(left_n/sample_tot)},{'{:.1%}'.format(left_n/(left_n+right_n))})",
            'event': left_event,
            'event_pct': f"({ '{:.1%}'.format(left_event/event_tot)})" if left_node_id<=2 else f"({'{:.1%}'.format(left_event/event_tot)},{'{:.1%}'.format(left_event/(left_event+right_event))})",
            'pd': left_pd,
            'pd3digit': round(left_pd, 3),
            'parent_node': node_id,
            'node_desc': node_desc + f'【{split_var} <= {split_value}】',
        }

        right_child_info = {
            'node_id': right_node_id,
            'impurity': round(right_impurity, 3),
            'n': right_n,
            'pct': f"({ '{:.1%}'.format(right_n/sample_tot)})" if right_node_id<=2 else f"({'{:.1%}'.format(right_n/sample_tot)},{'{:.1%}'.format(right_n/(left_n+right_n))})",
            'event': right_event,
            'event_pct': f"({ '{:.1%}'.format(right_event/event_tot)})" if right_node_id<=2 else f"({'{:.1%}'.format(right_event/event_tot)},{'{:.1%}'.format(right_event/(left_event+right_event))})",
            'pd': right_pd,
            'pd3digit': round(right_pd, 3),
            'parent_node': node_id,
            'node_desc': node_desc + f"【{split_var} > {split_value}】",
        }    

   
    return left_child_info, right_child_info

def manual_tree_plot(model_type, data_list, criterion, result_path, plot_name):
    
    if model_type == 'PD':
        weigted=''
        dot_template ="""digraph Tree {
        node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;
        graph [ranksep=equally, splines=polyline] ;
        edge [fontname=helvetica] ;
        {% for info in node_info_list %}
  
        {% if info['node_id'] == 0 %}
        {{info['node_id']}} [label=<node &#35; 0<br/>{{info['split_var']}} &le; {{info['split_value']}}<br/>{{weigted}}{{criterion}}={{info['impurity']}}<br/> samples={{info['n']}}(100%)<br/> events={{info['event']}}(100%)<br/>PD#,% = [{{info['event']}},{{info['pd3digit']}}]>, fillcolor="{{info['color_code']}}"] ;
        {% else %}

        {% if info.get('split_var', '') != '' %}
        {{info['node_id']}} [label=<node &#35;{{info['node_id']}}<br/>{{info['split_var']}} &le; {{info['split_value']}}<br/>{{weigted}}{{criterion}} = {{info['impurity']}}<br/> samples={{info['n']}} {{info['pct']}}<br/>events = {{info['event']}}{{info['event_pct']}}<br/> PD#,%=[{{info['event']}},{{info['pd3digit']}}]>, fillcolor="{{info['color_code']}}"] ;
        {% else %}
        {{info['node_id']}} [label=<node &#35; {{info['node_id']}}<br/> {{weigted}}{{criterion}} = {{info['impurity']}}<br/> samples={{info['n']}} {{info['pct']}}<br/> events = {{info['event']}} {{info['event_pct']}} <br/> PD#,%=[{{info['event']}}, {{info['pd3digit']}}]>, fillcolor="{{info['color_code']}}"] ;
        {% endif %}

        {% if info['node_id'] % 2 == 1 %}
        {{info['parent_node']}} -> {{info['node_id']}} [labeldistance=2.5, labelangle=45, headlabel=True] ;
        {% else %}
        {{info['parent_node']}} -> {{info['node_id']}} [labeldistance=2.5, labelangle=-45, headlabel=False] ;
        {% endif %}

        {% endif %}
        {% endfor %}
        {rank=same; {{node_leaves}}} ; }"""


    node_leaves = '; '.join([str(i['node_id']) for i in data_list if 'split_var' not in i])
    df = pd.DataFrame(data_list)
    if model_type == 'PD':
        col_name = 'pd'
    else:
        col_name = 'average'
    bds = sorted(list( np.percentile(df[col_name], list(range(0, 100, 5))[1:]) + [random.uniform(0, 0.0000001) for i in range(19) ] )   + [-np.inf, np.inf])
 
    binned = pd.cut(df[col_name], bds)
    df['positions'] = binned.cat.codes
    df['color_code'] = df['positions'].apply(lambda x: COLOR_PALLETE[x])
    df = df.fillna('')
    df['parent_node'] = df['parent_node'].apply(lambda x: int(x) if x!='' else x)
    data_list = df.to_dict('records')
    manual_dot_data = Template(dot_template).render(node_info_list=data_list, criterion=criterion, node_leaves=node_leaves, weigted=weigted)
    graph = graphviz.Source(manual_dot_data)
    graph.render(path_join(result_path, plot_name), cleanup=True, format='png', view=True)

    return graph






