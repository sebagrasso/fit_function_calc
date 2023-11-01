from dash import html, dash_table
import pandas as pd
import numpy as np

def parse_contents(filename):
    if filename is not None:
        try:
            if 'csv' in filename:
                df = pd.read_csv(filename)
            elif 'xls' in filename:
                df = pd.read_excel(filename)
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])

        return [df.to_json(date_format='iso', orient='split'), 
                html.Div([
                    html.H5(filename,
                        style={
                            'width': '30%',
                            'margin': '10px',
                            'font-size':20, "font-family":"calibri"
                        }),
        
                    html.Hr(),
        
                    dash_table.DataTable(
                        df.to_dict('records'),
                        [{'name': i, 'id': i} for i in df.columns],
                        style_cell={'textAlign': 'center'},
                        page_size=8
                    )], 
                    style={
                            # 'width': '30%',
                            'margin': '10px',
                            'font-size':15, "font-family":"calibri"
                        }
                )]

def fit_equation(model):
    poly_features = model.named_steps['poly']
    linear_regression = model.named_steps['linear']
    coefficients = linear_regression.coef_
    intercept = linear_regression.intercept_
    
    feature_names_ev = ['xtest**{}'.format(i) for i in range(1, poly_features.degree+1)]
    feature_names_pr = ['x^{}'.format(i) for i in range(1, poly_features.degree+1)]
    equation_eval, equation_print = "", ""
    equation_eval += "{:.5e}".format(intercept)
    equation_print += "{:.5e}".format(intercept)
    for i, coef in enumerate(coefficients):
        if i == 0 and coef == 0:
            pass
        elif i == 0:
            equation_eval += "{:.5e}".format(coef)
            equation_print += "{:.5e}".format(coef)
        else:
            equation_eval += " {} {:.5e} * {}".format('+' if coef >= 0 else '-', abs(coef), feature_names_ev[i-1])
            equation_print += " {} {:.5e} * {}".format('+' if coef >= 0 else '-', abs(coef), feature_names_pr[i-1])
    
    return equation_eval, equation_print

def x_y_from_selected(selectedData):
    selected_indices = selectedData['points']
    X=np.array([selected_indices[i]["x"] for i in range(len(selected_indices))])
    y=np.array([selected_indices[i]["y"] for i in range(len(selected_indices))])
    return X, y

def describe_df_func(df, selected):   
    describe_df = pd.DataFrame(df[selected].describe()).reset_index().round(2)
    describe_df = describe_df.rename(columns={"index":"statistics"})
    return describe_df