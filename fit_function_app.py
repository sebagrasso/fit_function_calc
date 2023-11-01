import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import base64
import io
import sys

import warnings
warnings.filterwarnings("ignore")

import applayout, appfuncs

app = dash.Dash(title="Dataset Fit", 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                )
app.server

app.layout = dbc.Container(
                    [
                        ## HEADER - INPUT DATA                        
                        dbc.Row([
                            dbc.Row(
                            [
                                dbc.Col(
                                    [applayout.HTML_TITLE],
                                    width="12"
                                )                                                             
                            ]
                            ),
                            dbc.Row(
                            [
                                dbc.Col([
                                    applayout.DCC_UPLOAD,
                                    html.Br(),
                                    html.Div(id='data-info', 
                                    style={'font-size':15,'margin': '2%'})],
                                    width = 3,
                                ),
                                dcc.Store(id='data-store', storage_type='memory'),
                                dbc.Col(
                                        [
                                            html.Br(),
                                            dcc.Loading(
                                                        id="loading-content",
                                                        type="circle",
                                                        children=[
                                                                    html.Div(id='output-data-upload',
                                                                    style={'font-size':9})
                                                                ]
                                                        )
                                        ],
                                    width = 9,
                                )
                            ]
                            ),
                            dbc.Row(
                                [html.Br(),
                                html.Hr()], style={"height":"1%"}
                            ),
                        ], style={"height":"20%"}),

                        ## DISTRIBUTION PLOTS HEADER
                        dbc.Row([
                                ## HEADERS
                                dbc.Row([
                                        dbc.Col([
                                                html.H2("Histogram", style={'textAlign': 'center',
                                                                                'font-size':24, "font-family":"calibri"}),
                                                html.Label("(Scatterplot's variable X)", style={'textAlign': 'center', 'width':'100%',
                                                                                                        'font-size':18, "font-family":"calibri"}),         
                                                ],
                                            width = 6,
                                        ),
                                        dbc.Col([
                                                html.H2("Boxplot", style={'textAlign': 'center',
                                                                            'font-size':24, "font-family":"calibri"}),
                                                html.Label("(Scatterplot's variable Y)", style={'textAlign': 'center', 'width':'100%',
                                                                                                        'font-size':18, "font-family":"calibri"}),                                         
                                                ], width = 6,
                                        ),]
                                    ),
                                ## DROPDOWNS
                                dbc.Row([
                                        dbc.Col(
                                            [],
                                            width = 2,
                                        ),
                                        dbc.Col(
                                            [applayout.DCC_DROPDOWN_X],
                                            width = 2,
                                        ),
                                        dbc.Col(
                                            [],
                                            width = 4,
                                        ),
                                        dbc.Col(
                                            [applayout.DCC_DROPDOWN_Y],
                                            width = 2,
                                        ),
                                        dbc.Col(
                                            [],
                                            width = 2,
                                        ),
                                        ]),
                        ], style={"height":"5%"}),
                        
                        ## DISTRIBUTION PLOTS
                        dbc.Row([
                                ## DISTRIBUTION PLOTS
                                dbc.Row([
                                        dbc.Col(
                                            [dcc.Graph(id='histogram_x')],
                                            width = 4,
                                        ),
                                        dbc.Col(
                                                [
                                                html.Div(id='describe-hist', style={'font-size':15,'padding-top': '30%'})
                                                ],
                                            width = 2,
                                        ),
                                        dbc.Col(
                                            [dcc.Graph(id='histogram_y')],
                                            width = 4,
                                        ),
                                        dbc.Col(
                                                [
                                                html.Div(id='describe-box', style={'font-size':15,'padding-top': '30%'})
                                                ],
                                            width = 2,
                                        ),
                                        ]),
                                dbc.Row(
                                        [html.Hr()], style={"height":"1%"}
                                        ),
                        ], style={"height":"20%"}),
                        
                        ## SCATTER IN HEADER
                        dbc.Row([
                            ## HEADERS
                            dbc.Row([                             
                                    dbc.Col(
                                        [html.H2("Input Data", style={
                                                             'textAlign': 'center',
                                                             'font-size':24, "font-family":"calibri"
                                                         })],
                                        width = 12,)
                                    ]),
                            ## COLOR
                            dbc.Row(
                            [
                                dbc.Col(
                                    [applayout.DCC_DROPDOWN_COLOR],
                                    width = 1,
                                ),
                                dbc.Col(
                                    [
                                    dcc.RangeSlider(
                                                    id="color-range-slider",
                                                    min=0,
                                                    max=1,
                                                    step=0.5,
                                                    value=[0, 1]
                                                )
                                    ], style={"padding-top":"1%"} , width = 2,
                                ),
                                dbc.Col(
                                    [applayout.DCC_DROPDOWN_COLOR],
                                    width = 1,
                                ),
                                dbc.Col(
                                    [
                                    dcc.RangeSlider(
                                                    id="color-range-slider",
                                                    min=0,
                                                    max=1,
                                                    step=0.5,
                                                    value=[0, 1]
                                                )
                                    ], style={"padding-top":"1%"} , width = 2,
                                ),
                                dbc.Col(
                                    [applayout.DCC_DROPDOWN_COLOR],
                                    width = 1,
                                ),
                                dbc.Col(
                                    [
                                    dcc.RangeSlider(
                                                    id="color-range-slider",
                                                    min=0,
                                                    max=1,
                                                    step=0.5,
                                                    value=[0, 1]
                                                )
                                    ], style={"padding-top":"1%"} , width =2,
                                ),
                                dbc.Col(
                                    [applayout.DCC_DROPDOWN_COLOR],
                                    width = 1,
                                ),
                                dbc.Col(
                                    [
                                    dcc.RangeSlider(
                                                    id="color-range-slider",
                                                    min=0,
                                                    max=1,
                                                    step=0.5,
                                                    value=[0, 1]
                                                )
                                    ], style={"padding-top":"1%"} , width =2,
                                ),                               
                            ]
                            ),
                        ], style={"height":"5%"}),
                        
                        ## SCATTER IN CHART
                        dbc.Row([
                            ## SCATTER IN
                            dbc.Row(
                            [
                                dbc.Col(
                                    [dcc.Graph(id='scatter_input')],
                                    width = 12,
                                )
                            ]
                            ),
                        ], style={"height":"20%"}),
                        
                        ### SCATTER OUT HEADER
                        dbc.Row([
                            ## HEADER
                            dbc.Row([
                                dbc.Col(
                                    [html.H5("Filtered Data", style={
                                                            'textAlign': 'center',
                                                            'font-size':24, "font-family":"calibri"
                                                        }), html.Br()],
                                    width = 12,)
                                    ]),
                            ### SLIDER POL DEGREE
                            dbc.Row(
                            [
                                dbc.Col(
                                    [],
                                    width = 1,
                                ),
                                dbc.Col(
                                    [html.Label("Polynomic degree: ", style={'font-size':20,'textAlign': 'right'})],
                                    width = 2,
                                ),
                                dbc.Col(
                                    [
                                     dcc.Slider(id="slider-poly-degree", min=0, max=6,
                                                marks={i: str(i) for i in range(21)},
                                                value=3,
                                                step=1
                                                )],
                                    width = 6,
                                ),
                                dbc.Col(
                                    [],
                                    width = 3,
                                ),                                
                            ]
                            ),
                        ], style={"height":"5%"}),
                        
                        ### SCATTER OUT CHART
                        dbc.Row([
                            ### SCATTER OUT
                            dbc.Row(
                            [
                                dbc.Col(
                                    [
                                    dcc.Loading(
                                        id="loading",
                                        type="circle",
                                        children=[
                                                    dcc.Graph(id='scatter_output')]
                                    )],
                                    width = 12,
                                )
                            ]
                            ),                  
                            dbc.Row(
                            [html.Hr()], style={"height":"1%"}
                            ),
                        ], style={"height":"20%"}),
                        
                        ## REPORTS
                        dbc.Row([
                            html.Div([
                                    dbc.Row([
                                            dbc.Col(
                                                            [
                                                            html.H2("Fit funtion equation: ", style={"color":"white", 'font-size':24, 
                                                                                                    "font-family":"calibri"}
                                                                    ),
                                                            html.Label(id="print_fit_function", style={"color":"white", 'font-size':18,
                                                                                                        "font-weight":"bold"}
                                                                        )
                                                            ], width=5
                                                            ),
                                            dbc.Col(
                                                            [
                                                            dbc.Row([
                                                                        html.H2("Function test: ", 
                                                                            style={"color":"white", 'font-size':24, "font-family":"calibri"}),
                                                                    ]),
                                                            dbc.Row([
                                                                    dbc.Col([
                                                                            html.Label("X =    ", style={"color":"white", "padding-right":"3%"}),
                                                                            dcc.Input(placeholder="value", id="xtest-function", type="number", 
                                                                            value=0, style={"width":"70%"})
                                                                            ],
                                                                        ),
                                                                    dbc.Col([
                                                                            html.Label("Y = ", style={"color":"white", }), 
                                                                            html.Label(id="ytest-function", style={"color":"white", 'font-size':18,
                                                                                                                    "font-weight":"bold",
                                                                                                                    "padding-left":"3%"}),
                                                                            ],
                                                                        )
                                                                    ])
                                                            ], width=3
                                                            ),
                                            dbc.Col([
                                                            dcc.Loading(
                                                                id="loading-button",
                                                                type="default",
                                                                children=[dbc.Button("Export Selected Points",
                                                                        id="export-button", 
                                                                        color="success", className="mr-1")],
                                                                ),
                                                            dcc.Download(id="download-data"),
                                                            html.Label(id="export-label", style={"textAlign":"left", "color":"white", 
                                                                                                'font-size':18, "font-weight":"lighter",
                                                                                                'padding-top':"3%"})
                                                            ], style={'padding-top':"1%"}, width=4),
                                            ]
                                            ),
                            ], style={"padding-left":"20px", "padding-top":"20px", "background-color":"#8DBCBC"}),
                        
                            ## FOOTER
                            html.Div([
                                        dbc.Row(
                                                dbc.Col(
                                                    [html.Br(), html.Hr(), html.Br()]
                                                    )
                                                )
                                    ], style = {"background-color":"#8DBCBC"}
                                    )
                        ], style={"height":"5%"}),
                    
                    ], style={'maxWidth': '90%', 
                              'height':'100%'},
                )

############################################ UPLOAD FILE 
@callback(Output('data-store', 'data'),
          Output('output-data-upload', 'children'),
          Input('upload-data', 'filename'),
          prevent_initial_call=True)
def update_output(list_of_names):
    children = [appfuncs.parse_contents(list_of_names)[1]]
    df_json = appfuncs.parse_contents(list_of_names)[0]
    return df_json, children

############################################ DATA INFO
@app.callback(
    Output('data-info', 'children'),
    Input('data-store', 'data'),
    prevent_initial_call=True)
def update_data_info(df_load):
    if df_load is not None:
        df = pd.read_json(df_load, orient='split')
        output = io.StringIO()
        sys.stdout = output
        df.info()
        sys.stdout = sys.__stdout__  # Reset sys.stdout to the default

        # Render the captured output as text within the div element
        return html.Pre(output.getvalue(), style={'white-space': 'pre-wrap'})   
    return ''

############################################ DESCRIBE HIST
@app.callback(
    Output('describe-hist', 'children'),    
    Input('data-store', 'data'),
    Input('x-dropdown', 'value'),    
    prevent_initial_call=True)
def update_data_info(df_load, selected_x):
    if df_load is not None and selected_x is not None:
        df = pd.read_json(df_load, orient='split')
        describe_df = appfuncs.describe_df_func(df, selected_x)
        return dash_table.DataTable(describe_df.to_dict('records'),
                                    style_cell={'textAlign': 'center'},
                                    page_size=8)
    return ''

############################################ DESCRIBE BOX
@app.callback(
    Output('describe-box', 'children'),    
    Input('data-store', 'data'),
    Input('y-dropdown', 'value'),    
    prevent_initial_call=True)
def update_data_info(df_load, selected_y):
    if df_load is not None and selected_y is not None:
        df = pd.read_json(df_load, orient='split')
        describe_df = appfuncs.describe_df_func(df, selected_y)
        return dash_table.DataTable(describe_df.to_dict('records'),
                                    style_cell={'textAlign': 'center'},
                                    page_size=8)
    return ''

############################################ X, Y DROPDOWN OPTIONS 
@app.callback(
    Output('x-dropdown', 'options'),
    Output('y-dropdown', 'options'),
    Input('data-store', 'data'),
    prevent_initial_call=True)
def update_dropdown_options(data):
    if data is not None:
        df = pd.read_json(data, orient='split')
        options = [{'label': col, 'value': col} for col in df.columns]
        
        return options, options
    
    return [], []

############################################ Z DROPDOWN OPTIONS 
@app.callback(
    Output('color-dropdown', 'options'),
    Input('data-store', 'data'),
    prevent_initial_call=True)
def update_dropdown_options(data):
    if data is not None:
        df = pd.read_json(data, orient='split')
        options = [{'label': col, 'value': col} for col in df.columns]
        options.append({'label': 'None', 'value': ''})
        
        return options
    
    return []

############################################ HISTOGRAM X 
@app.callback(
    Output('histogram_x', 'figure'),
    Input('x-dropdown', 'value'),
    Input('data-store', 'data'),)
def update_scatter_plot_x(selected_x, data):
    if selected_x and data is not None:
        df = pd.read_json(data, orient='split')
        fig = px.histogram(df, x=selected_x, nbins=30, opacity=0.4, color_discrete_sequence=['blue'])
        fig.update_layout(transition_duration=500)
        return fig
    else:
        return px.histogram([], nbins=30, opacity=0.4, color_discrete_sequence=['blue'])

############################################ BOXPLOT Y
@app.callback(
    Output('histogram_y', 'figure'),
    Input('y-dropdown', 'value'),
    Input('data-store', 'data'))
def update_scatter_plot_y(selected_y, data):
    if selected_y and data is not None:
        df = pd.read_json(data, orient='split')
        fig = px.box(df, x=selected_y, color_discrete_sequence=['green'])
        fig.update_layout(transition_duration=500)
        return fig
    else:
        return px.box([], color_discrete_sequence=['green'])

############################################ SCATTERPLOT INPUT
@app.callback(
    Output('scatter_input', 'figure'),
    Input('x-dropdown', 'value'),
    Input('y-dropdown', 'value'),
    Input('color-dropdown', 'value'),
    Input('data-store', 'data'),
    Input('color-range-slider', 'value'),
    prevent_initial_call=True)
def update_scatter_plot(selected_x, selected_y, selected_color, data, slider_value):
    if selected_x and selected_y and data is not None:
        df = pd.read_json(data, orient='split')
  
        if selected_color:
            df = df[(df[selected_color] >= slider_value[0]) & (df[selected_color] <= slider_value[1])]
            fig = px.scatter(df, x=selected_x, y=selected_y, color=df[selected_color].astype(str), height=600,
            size=[100]*df.shape[0], size_max=8, color_discrete_sequence=px.colors.qualitative.Light24)
        else:
            fig = px.scatter(df, x=selected_x, y=selected_y, height=600,
            size=[100]*df.shape[0], size_max=8)
        fig.update_layout(xaxis=dict(range=[df[selected_x].min(), df[selected_x].max()]),
                          yaxis=dict(range=[df[selected_y].min(), df[selected_y].max()]),
                          transition_duration=800)
        return fig 
    else:
        return px.scatter()

############################################ SCATTERPLOT OUTPUT
@app.callback(
    Output('scatter_output', 'figure'),
    Output('print_fit_function', 'children'),
    Output('ytest-function', 'children'),
    Input('scatter_input', 'selectedData'),
    Input('slider-poly-degree', 'value'),
    Input('xtest-function', 'value'),
    prevent_initial_call=True)
def display_selected_data(selectedData, degree, xtest):
    if selectedData is not None:
        X, y = appfuncs.x_y_from_selected(selectedData)        
        scatter = go.Scatter(x=X, y=y, mode="markers", marker=dict(size=8, opacity=0.5), name = "Filtered data")
       
        model = Pipeline([("poly", PolynomialFeatures(degree=degree)), ("linear", LinearRegression())])
        if (X.size > 2) and (y.size > 2):
            model.fit(X.reshape(-1,1), y)

            x_plot = np.arange(X.min(), X.max(), 0.1)
            y_pred = model.predict(x_plot.reshape(-1,1))

            poly = go.Scatter(x=x_plot.flatten(), y=y_pred, mode='lines', 
                                line=dict(color= "black", width=3), name='Fit function')

            layout = go.Layout(height=600, transition_duration=800)

            fig = go.Figure(data=[scatter, poly], layout=layout)
            equation_eval, equation_print = appfuncs.fit_equation(model)
            return fig, equation_print, np.round(eval(equation_eval),2)
        else:
            layout = go.Layout(height=600, transition_duration=800)
            fig = go.Figure(data=[scatter], layout=layout)
            return fig, "No function built", "-"
    else:
        return px.scatter(), "No function built", "-"

############################################ FILTER SLIDERS
@app.callback(
    Output('color-range-slider', 'min'),
    Output('color-range-slider', 'max'),
    Output('color-range-slider', 'step'),
    Output('color-range-slider', 'value'),
    Input('data-store', 'data'),
    Input('color-dropdown', 'value'),
    prevent_initial_call=True)
def update_sliders(df_json, selected_color):
    if df_json and selected_color:
        df = pd.read_json(df_json, orient='split')
        min_value = df[selected_color].min()
        max_value = df[selected_color].max()
        if (df[selected_color].dtype=="int64") or (df[selected_color].dtype=="int32"):
            step=1
        else:
            step=0.5
        value = [min_value, max_value]

        return min_value, max_value, step, value
    else:
        return 0, 1, 0.5, [0, 1]

############################################ EXPORT BUTTON
@app.callback(
    Output("download-data", "data"),
    Output('export-label', 'children'),
    Output('export-button', 'n_clicks'),
    Input('export-button', 'n_clicks'),
    Input('scatter_input', 'selectedData'),
    Input('x-dropdown', 'value'),
    Input('y-dropdown', 'value'),
    prevent_initial_call=True)
def export_to_csv(n_clicks, selectedData, selected_x, selected_y):
    if not n_clicks:
        raise PreventUpdate
    elif selectedData is not None:
        X, y = appfuncs.x_y_from_selected(selectedData)
        df = pd.DataFrame({selected_x: X, selected_y: y})
        filename = f"selected_points_{datetime.now().strftime('%d%b%Y_%H%Mhs')}.xlsx"
        # excel_file = df.to_excel(filename, index=False)
        return dcc.send_data_frame(df.to_excel, filename=filename), f"'{filename}' downloaded succesfully", None
    else:
        return dash.no_update, "No data selected", None

if __name__ == "__main__":
    app.run_server(debug=True, port=2323)
    # app.run_server(port=2323)



## BUY ME A COFFEE
## CLUSTERS
## OTHER FUNCTIONS. TREE, SVC, ETC
## COMMENTS
## GUIA TUTORIAL PASO A PASO ( DON'T SHOW AGAIN)
## LINKEDIN - GITHUB - PORFOLIO
## DROPNA BUTTONS POR COL Y ROW, ANY Y ALL, REFRESH STORE INFO AND TABLE. (RESET BUTTON?)
## TXT UPLOAD