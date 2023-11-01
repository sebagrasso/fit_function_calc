from dash import html, dcc

HTML_TITLE = html.H1("Fit data with function", 
                    style={"text-align":"center", 
                           "font-size":50, "font-family":"calibri",
                           "background-color":"#000038",
                           "color":"white",
                           "height":"80px",
                           "padding-top":"8px"}
                    )

DCC_UPLOAD = dcc.Upload(
                        id="upload-data",
                        children=html.Div([
                            "Drag and Drop or ",
                            html.A("Select Files", style={"font-weight":"bold"}),
                            html.Div("(csv or xls)", style={"font-size":18})
                        ]),
                        style={
                            # 'lineHeight': '60px',
                            'borderWidth': '1.6px',
                            'borderStyle': 'dashed',
                            'borderRadius': '12px',
                            'border-color':"grey",
                            'textAlign': 'center',
                            'font-size':20, "font-family":"calibri"
                        },
                        multiple=False)

DCC_DROPDOWN_X = dcc.Dropdown(
                                id='x-dropdown',
                                options=[],
                                value=None,
                                placeholder="X",
                                style={
                                   'textAlign': 'center',
                                   'margin': '5%',
                                   'font-size':15, 
                                   "font-family":"calibri"},
                                clearable=False,
                                persistence=False,
                            )

DCC_DROPDOWN_Y = dcc.Dropdown(
                                id='y-dropdown',
                                options=[],
                                value=None,
                                placeholder="Y",
                                style={
                                   'textAlign': 'center',
                                   'margin': '5%',
                                   'font-size':15, 
                                   "font-family":"calibri"},
                                clearable=False,
                                persistence=False,
                            )

DCC_DROPDOWN_COLOR = dcc.Dropdown(
                                id='color-dropdown',
                                options=[],
                                value=None,
                                placeholder="COLOR",
                                style={
                                   'textAlign': 'center',
                                   'margin': '5%',
                                   'font-size':15,
                                   "font-family":"calibri"},
                                clearable=False,
                                persistence=False,
                            )