import dash
import base64
import datetime

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
from dash import callback

import plotly.express as px
import plotly.graph_objects as go

from utilities.utils import *
from utilities.constants import *

import pandas as pd

# template data
df_sheets = pd.read_excel(os.path.join(data_dir,template_filename), sheet_name=None, engine= 'openpyxl')

# Model features data
model_features_data = pd.read_csv(os.path.join(data_dir,selected_features_filename))

# predict sales at combination level here
model_features_data = process_features_file(model_features_data)
# baseline sales predictions
df_sales_pred_scenario = pd.read_csv(os.path.join(data_dir,sales_pred_filename))

df_sales_pred_scenario['Manufacturer'] = df_sales_pred_scenario['Manufacturer'].astype('str')
df_sales_pred_scenario['Channel'] = df_sales_pred_scenario['Channel'].astype('str')
df_sales_pred_scenario['Date'] = pd.to_datetime(df_sales_pred_scenario['Date'])

min_date = df_sales_pred_scenario[df_sales_pred_scenario['Sales'].isna()]['Date'].min()

# getting values for Channel and Manufacturer filters
fnameDict = {}
full_lst = []
for cnl in df_sales_pred_scenario.Channel.unique():
    mfgs = df_sales_pred_scenario[df_sales_pred_scenario['Channel']==cnl]['Manufacturer'].unique().tolist()
    fnameDict[cnl] = mfgs
    full_lst = list(set(full_lst + mfgs))

fnameDict['TOTAL'] = full_lst
names = list(fnameDict.keys())
nestedOptions = fnameDict[names[0]]


def scen_parse_contents(contents, filename, date):
    """parse the contents that are uploaded and convert them into readable format

    Args:
        contents (str): encrypted string containing data
        filename (str): filename
        date (str): timestamp

    Returns:
        html div: with data pre processed, filters, sales predictions, 
        Sales and SOM graphs and also KPI trend graph yo verify edited columns
    """    
    global model_features_data, df_sales_pred_scenario
    
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

        elif 'xlsx' in filename:
            # print(filename)
            df = preprocessing_future_data(io.BytesIO(decoded))
            df = generate_comp_features_1(df)

            sales_pred = predict_sales(df, df_sales_pred_scenario, model_features=model_features_data)
    except Exception as e:
        print("-"*20)
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(f"Uploaded file name:{filename}"),

        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label('Channel', style= {'marginLeft': '0px', 'marginRight': '64px', 'font-family': 'Verdana', 'font-size': '14px',
                                                'font-weight': 'bold'}),
                    dcc.Dropdown(id='channel',
                        options=[{"label": i, "value": i} for i in names],
                                value=names[0], clearable = False, style={'color': 'black', 'font-family': 'Verdana', 'font-size': '14px'}),
                ], width=2),

                dbc.Col([
                    html.Label('Manufacturer', style= {'marginLeft': '0px', 'marginRight': '50px', 'font-family': 'Verdana', 'font-size': '14px',
                                                        'font-weight': 'bold'}),
                    dcc.Dropdown(id='manufacturer',
                        options=[{"label": i, "value": i} for i in nestedOptions],
                                value='PEPSICO', clearable= False, style={'color': 'black', 'font-family': 'Verdana', 'font-size': '14px'}),
                ], width=2),
            ]),
                dbc.Row([
                    dbc.Col([
                        dbc.ListGroup([
                            dbc.ListGroupItem(children= '', style = {'width': '90px', 'font-weight': 'bold', 'color': '#F9F9F9',
                                                                    'font-color': 'white', 'border': 'none',
                                                            'background-color': '#F9F9F9', 'font-family': 'Verdana'}),
                            dbc.ListGroupItem(children= 'Baseline SOM For Total 2023', style = {'font-weight': 'bold', 'border': 'none', 'text-align': 'center',
                                                                                'width': '230px', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '12px'}),
                            dbc.ListGroupItem(style = {'border': 'none', 'width': '20px','background-color': '#F9F9F9'}),
                            dbc.ListGroupItem(children= 'User Modified SOM For Total 2023', style = {'font-weight': 'bold', 'text-align': 'center', 'width': '230px',
                                                                                'border': 'none', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '12px'})
                        ], horizontal=True),
                    ], width = 6),
                    dbc.Col([
                        dbc.ListGroup([
                            dbc.ListGroupItem(children= '', style = {'width': '90px', 'font-weight': 'bold', 'color': '#F9F9F9',
                                                                    'font-color': 'white', 'border': 'none',
                                                            'background-color': '#F9F9F9', 'font-family': 'Verdana'}),
                            dbc.ListGroupItem(children= 'Baseline Sales For Total 2023 (10^3 Pesos)', style = {'font-weight': 'bold', 'border': 'none', 'text-align': 'center',
                                                                                'width': '230px', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '12px'}),
                            dbc.ListGroupItem(style = {'border': 'none', 'width': '20px', 'background-color': '#F9F9F9'}),
                            dbc.ListGroupItem(children= 'User Modified Sales For Total 2023 (10^3 Pesos)', style = {'font-weight': 'bold', 'text-align': 'center', 'width': '230px',
                                                                                'border': 'none', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '12px'})
                        ], horizontal=True),
                    ], width = 6)
                ], justify="center"),
                dbc.Row([
                    dbc.Col([
                        dbc.ListGroup([
                            dbc.ListGroupItem(children= '', style = {'width': '90px', 'font-weight': 'bold', 'color': '#F9F9F9',
                                                                    'font-color': 'white', 'border': 'none',
                                                            'background-color': '#F9F9F9', 'font-family': 'Verdana'}),
                            dbc.ListGroupItem(id= 'baseline-som-scenario', style = {'background-color': '#01529C', 'width': '230px',
                                                               'color': 'white',
                                                               'font-weight': 'bold',
                                                               'font-size': '12px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center'}),
                            dbc.ListGroupItem(style = {'border': 'none', 'width': '20px', 'background-color': '#F9F9F9'}),
                            dbc.ListGroupItem(id = 'modified-som-scenario', style = {'background-color': '#C9002B',
                                                               'color': 'white', 'width': '230px',
                                                               'font-weight': 'bold',
                                                               'font-size': '12px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center'})
                        ], horizontal=True),
                    ], width = 6),
                    dbc.Col([
                        dbc.ListGroup([
                            dbc.ListGroupItem(children= '', style = {'width': '90px', 'font-weight': 'bold', 'color': '#F9F9F9',
                                                                    'font-color': 'white', 'border': 'none',
                                                            'background-color': '#F9F9F9', 'font-family': 'Verdana'}),
                            dbc.ListGroupItem(id = 'baseline-sales-scenario', style = {'background-color': '#01529C', 'width': '230px',
                                                               'color': 'white',
                                                               'font-weight': 'bold',
                                                               'font-size': '12px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center'}),
                            dbc.ListGroupItem(style = {'border': 'none', 'width': '20px', 'background-color': '#F9F9F9'}),
                            dbc.ListGroupItem(id= 'modified-sales-scenario', style = {'background-color': '#C9002B', 'width': '230px',
                                                               'color': 'white',
                                                               'font-weight': 'bold',
                                                               'font-size': '12px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center'})
                        ], horizontal=True),
                    ], width = 6)
                ], justify="center"),
                html.Br(),

                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='simulate-div1')
                    ], width=6, style = {'border-right': '2px solid grey', 'margin-top': '5px', 'margin-bottom': '10px'}),
                    dbc.Col([
                        dcc.Graph(id='simulate-div2')
                    ], width=6)
                ], style = {'border-top': '2px solid grey', 'border-bottom': '2px solid grey', 'margin': '0px'}),

                html.Div(id= 'data-div'),
                dcc.Store(id='stored-data', data=sales_pred.to_dict('records')),
                dcc.Store(id='stored-data-2', data=df.to_dict('records'))
    ])])



# Dash App
# registering the name of this page --> Scenario Management
dash.register_page(__name__)

# layout styling for dash app
layout = dbc.Container([
    dbc.Row([
        dbc.ButtonGroup(
        [
            dbc.Button("Home", id= 'navigation-page',
                       style = {'width': '3in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white',
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       className="ml-auto",
                       href='/home'),
            dbc.Button("Baseline Prediction", id= 'first-page',
                       style = {'width': '3in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white',
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/Simulation',
                       className="ml-auto"),
            dbc.Button("Prediction Accuracy", className="ml-auto", id= 'second-page',
                       style = {'width': '3in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white',
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/ErrorMetrics'
                       ),
            dbc.Button("Scenario Management", className="ml-auto", id= 'third-page',
                       style = {'width': '3in', 'background-color': '#01529C', 'color': 'white',
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                    #    href='/scenario'
                       )
        ],
    )
    ], align = 'center', justify= 'center'),

        html.Br(),
        
        html.Div([
            html.Button("Download Template", id="btn_xlsx"),
            dcc.Download(id="download-dataframe-xlsx"),
            ]),

        dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select File'),
                    # html.Button(children='Select a File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'font-family': 'Verdana'
                },
                # Allow multiple files to be uploaded
                 multiple=True
    ),

    html.Div(id='output-datatable'),

], fluid = True, style = {'background-color': '#F9F9F9'})


# Template download function
@callback(
    Output("download-dataframe-xlsx", "data"),
    Input("btn_xlsx", "n_clicks"),
    prevent_initial_call=True,
)

def download_as_excel(n_clicks):
    """downloads the file as a template with baseline predictions, to be edited and
    uploaded again

    Args:
        n_clicks (int)

    Raises:
        PreventUpdate

    Returns:
        downloads a file onto your system
    """    
    global df_sheets
    if not n_clicks:
        raise PreventUpdate
    else:
        writer = pd.ExcelWriter('template_scenario.xlsx', engine="xlsxwriter")
        for key in df_sheets.keys():
            df_sheets[key].to_excel(writer, sheet_name=key,index=False)
        writer.save()
        

        return dcc.send_file('template_scenario.xlsx')


# parse contents function
@callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              )

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            scen_parse_contents(c, n, d) for c, n, d in zip(list_of_contents,
                                                       list_of_names,
                                                       list_of_dates)]
        return children


# Gets select kpi text, dropdown of kpis, verify values button,
# export button, and dash data table
@callback(
    Output(component_id='data-div', component_property='children'),
    State('stored-data','data'),
    Input(component_id='channel', component_property='value'),
    Input(component_id='manufacturer', component_property='value')
)

def get_table(data, cnl, mfg):
    # show data from here
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Channel']==cnl)&(df['Manufacturer']==mfg)]
    df = df.drop(columns= ['PPU', 'TDP'])

    cols = df.columns
    cols_ = list()
    cols_1 = list()
    for i,x in enumerate(cols):
        if not str(x).isnumeric():
            cols_.append(x)

    for i,x in enumerate(cols_):
        if 'Ratio' not in x:
            cols_1.append(x)

    fixed_cols = ['Manufacturer','Channel','Date','Month',
                  'Baseline Sales Prediction','User modified Sales Prediction',
                  'Baseline SOM Prediction', 'User modified SOM Prediction']

    cols = list(set(cols_1).difference(set(fixed_cols)))
    cols = sorted(cols, key=sort_fun)
    df = df.sort_values(by='Date')
    df = df[fixed_cols+cols]

    df['Date'] = df['Date'].dt.date

    df = round(df,2)
    df = df.dropna(axis = 1,how='all')

    if cnl == 'TOTAL':
        cols = []
        placeholder = 'Please select a channel'
    else:
        placeholder = 'Select a KPI'

    return   html.Div([
                html.P("Select a KPI"),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(id='xaxis-data',placeholder=placeholder,
                                    options=[{'label':x, 'value':x} for x in cols],
                                    style = {"display":'inline-block', 'color': 'black', 'font-family': 'Verdana', 
                                            'font-size': '14px',
                                            'width':300,
                            })
                    ], width=4.5),
                    dbc.Col([
                        html.Button(id="submit-button-scenario",
                                    style = {'display':'inline-block',
                                            'width': 200, 'margin-left': '10px','font-family': 'Verdana', 'font-size': '14px',
                                            'width': '120px', 'height': '36px'
                                            }
                                    )
                    ], width=2.5),
                    dbc.Col([
                        html.Button(id='export-button-scenario', children='Export Results', n_clicks=0, 
                                     style = {"display":'inline-block',
                                              'width':200,
                                              'margin-left':20, 'font-family': 'Verdana', 'font-size': '14px',
                                              'width': '120px', 'height': '36px', 
                                              }
                                    ),
                        dcc.Download(id="download-dataframe-scenario-csv")
                    ], width=2.5),
                ], style= {'margin-left': '3px'}),
                html.Div([html.Div(id='output-div'),
                dash_table.DataTable(id='filtered-data',
                                     data=df.to_dict('records'), 
                                     columns=[{'name': str(i), 'id': str(i)} for i in df.columns],
                                style_header={ 'border': '1px solid white', 'whiteSpace':'normal', 'color': 'white',
                                              'font-weight': 'bold', 'backgroundColor': '#01529C', 'font-family': 'Verdana',
                                              'font-size':'10px'},
                                style_cell={ 'border': '1px solid grey', 'minWidth': 100, 'maxWidth': 120,
                                            'background-color': '#F9F9F9', 'font-family': 'Verdana', 'font-size':'10px'},
                                style_table={'overflowX': 'auto', 'height': '300px', 'overflowY': 'auto'},
                                fixed_rows={'headers': True},
                                style_header_conditional=[{
                                    'if': {'column_editable': True},
                                    'backgroundColor': '#F17836',
                                    'color': 'white'
                                    }],
                                ),
                dbc.Row([
                        dbc.Col([
                            html.Br()
                        ], width=12)
                    ], style= {'border-top': '1px solid grey', 'margin': '0px'}),
                
                ])
                        # page_size=15
            ])


# Download user modified scenario results data
@callback(
    Output("download-dataframe-scenario-csv", "data"),
    Output("export-button-scenario", "n_clicks"),
    Input("export-button-scenario", "n_clicks"),
    Input('filtered-data', 'data'),
    # Input('table-editing-simple', 'columns'),
    prevent_initial_call=True,
)
def download_scenario_data(n_clicks, data):
    """download the user uploaded scenario results, 
    i.e., sales predictions based on user uploaded values can be downloaded here as csv file

    Args:
        n_clicks (int): no of times the download button is clicked
        data (dataframe): dataframe containing the sales predictions of the selected manufacturer and channel 

    Returns:
        a csv file download: dataframe containing the sales predictions of the selected manufacturer and channel 
    """    
    df = pd.DataFrame(data)
    if n_clicks > 0:
        # df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
        n_clicks = 0
        return dcc.send_data_frame(df.to_csv, "Scenario_results_data.csv"), n_clicks
    else:
        n_clicks = 0
        return dash.no_update, n_clicks


# Makes verify results graphs of KPI
@callback(Output('output-div', 'children'),
          Output('submit-button-scenario', 'children'),
              Input('submit-button-scenario','n_clicks'),
              State('stored-data-2','data'),
              State('xaxis-data','value'),
              Input(component_id='channel', component_property='value'),
              Input(component_id='manufacturer', component_property='value'),
)

def make_trend_graphs(n, data, kpi,cnl, mfg):
    """plot the trend graphs for user to verify the 
    edited values and baeline predicted values of TDP/PPU 
    across different manufacturers

    Args:
        n (int): no of clicks
        data (dataframe)
        kpi (str): selected from the filter in the scenario page
        cnl (str): selected from the filter in the scenario page
        mfg (str): selected from the filter in the scenario page

    Returns:
        graph: a trend graph with user modified values and baseline predicted values of TDP/PPU
    """    
    if n is None:
        return dash.no_update, 'Verify values'
    elif n%2 == 0:
        return html.Div(id='output-div'), 'Verify values'
    else:
        df_sales_pred_data = df_sales_pred_scenario[(df_sales_pred_scenario['Manufacturer']==mfg) &
                                    (df_sales_pred_scenario['Channel']==cnl) &
                                    (df_sales_pred_scenario['Date']>=min_date)][['Date',kpi]]

        df_sales_pred_data.rename(columns = {kpi:f"Baseline {kpi}"}, inplace=True)
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])

        df = df[(df['Channel']==cnl) &
            (df['Manufacturer']==mfg) &
            (df['Date']>=min_date)
            ]
        df['Date'] = pd.to_datetime(df['Date'])
        df['PPU'] = df[f"{mfg} PPU"]
        df['TDP'] = df[f"{mfg} TDP"]
        dff = df.merge(df_sales_pred_data, on= ['Date'])
        dff.rename(columns = {kpi:f"User Modified {kpi}"}, inplace=True)

        bar_fig_2 = px.line(dff, x=dff['Date'], y=[f"Baseline {kpi}",f"User Modified {kpi}"])
        bar_fig_2 = go.Figure()
        bar_fig_2.add_trace(go.Scatter(x= dff['Date'], y = dff[f"Baseline {kpi}"], name= f"Baseline {kpi}", line=dict(color='#015CB4')))
        bar_fig_2.add_trace(go.Scatter(x= dff['Date'], y = dff[f"User Modified {kpi}"],
                                name= f"User Modified {kpi}", marker= dict(color= '#C9002B')))
        bar_fig_2.update_layout(title= f"Baseline {kpi} vs User Modified {kpi}", yaxis_title = 'value', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                            legend= dict(orientation= 'h', title=None))

        return dcc.Graph(figure=bar_fig_2), 'Collapse'

# SOM graphs
@callback(
    Output('simulate-div1', 'figure'),
    State('stored-data','data'),
    Input(component_id='channel', component_property='value'),
    Input(component_id='manufacturer', component_property='value'),
)

def make_som_graphs(data, cnl, mfg):
    """creates SOM graphs in scenario page

    Args:
        data (dataframe): sales prediction data
        cnl (str): selected from the filter in the scenario page
        mfg (str): selected from the filter in the scenario page

    Returns:
        Graph: Trend graph of SOM with baseline prediction and used modified prediction is shown
    """    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])

    df = df[(df['Channel']==cnl) &
        (df['Manufacturer']==mfg) &
        (df['Date']>=min_date)
        ]
    df['Date'] = pd.to_datetime(df['Date'])
    df['PPU'] = df[f"{mfg} PPU"]
    df['TDP'] = df[f"{mfg} TDP"]

    fig_som = go.Figure()
    fig_som.add_trace(go.Scatter(x= df['Date'], y = df['Baseline SOM Prediction'],
                                   name= 'Baseline SOM Prediction',
                                   line=dict(color='#015CB4')))
    fig_som.add_trace(go.Scatter(x= df['Date'], y = df['User modified SOM Prediction'],
                                 name= 'User modified SOM Prediction',
                                 marker= dict(color= '#C9002B')))
    fig_som.update_layout(title= "SOM Simulation - Baseline Vs User Modified", yaxis_title = 'SOM', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                        legend= dict(orientation= 'h', title=None))

    return fig_som


# Sales graph
@callback(
    Output('simulate-div2', 'figure'),
    Output('baseline-som-scenario', 'children'),
    Output('modified-som-scenario', 'children'),
    Output('baseline-sales-scenario', 'children'),
    Output('modified-sales-scenario', 'children'),
    State('stored-data','data'),
    Input(component_id='channel', component_property='value'),
    Input(component_id='manufacturer', component_property='value'),
)

def make_sales_graphs(data, cnl, mfg):
    """creates Sales graphs in scenario page

    Args:
        data (dataframe): sales prediction dataframe
        cnl (str): selected from the filter in the scenario page
        mfg (str): selected from the filter in the scenario page

    Returns:
        Graph: Trend graph of Sales with baseline prediction and used modified prediction is shown
    """    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    # print(df.shape)
    # print(df.head(2))
    # text blocks
    df['year'] = df['Date'].dt.year
    df_year = df[df['year']==2023]
    #print('='*30)
    #print(df_year.head())
    df_year = df_year.groupby(['Manufacturer', 'Channel'])[['Baseline Sales Prediction', 'User modified Sales Prediction']].sum().reset_index()
    df_year['total_baseline_sales'] = df_year.groupby(['Channel'])['Baseline Sales Prediction'].transform('sum')
    df_year['total_modified_sales'] = df_year.groupby(['Channel'])['User modified Sales Prediction'].transform('sum')
    df_year['baseline_SOM'] = df_year['Baseline Sales Prediction']*100 / df_year['total_baseline_sales']
    df_year['modified_SOM'] = df_year['User modified Sales Prediction']*100 / df_year['total_modified_sales']
    df_year = df_year[(df_year['Manufacturer']==mfg)&(df_year['Channel']==cnl)]
    #print(df_year.head(2))
    baseline_som = np.round(df_year['baseline_SOM'].values[0], 2)
    modified_som = np.round(df_year['modified_SOM'].values[0], 2)
    baseline_sales = np.int(df_year['total_baseline_sales'].values[0])
    modified_sales = np.int(df_year['total_modified_sales'].values[0])

    baseline_som, modified_som = f'{baseline_som} %', f'{modified_som} %'
    baseline_sales, modified_sales = '{:,}'.format(baseline_sales), '{:,}'.format(modified_sales)

    df = df[(df['Channel']==cnl) &
        (df['Manufacturer']==mfg) &
        (df['Date']>=min_date)
        ]
    df['Date'] = pd.to_datetime(df['Date'])
    df['PPU'] = df[f"{mfg} PPU"]
    df['TDP'] = df[f"{mfg} TDP"]


    fig_sales = go.Figure()
    fig_sales.add_trace(go.Scatter(x= df['Date'], y = df['Baseline Sales Prediction'],
                                   name= 'Baseline Sales Prediction',
                                   line=dict(color='#015CB4')))
    fig_sales.add_trace(go.Scatter(x= df['Date'], y = df['User modified Sales Prediction'],
                                 name= 'User modified Sales Prediction',
                                 marker= dict(color= '#C9002B')))
    fig_sales.update_layout(title= "Sales Simulation - Baseline Vs User Modified",
                            yaxis_title = 'Sales (10^3 Pesos)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                            legend= dict(orientation= 'h', title=None))

    return fig_sales, baseline_som, modified_som, baseline_sales, modified_sales
