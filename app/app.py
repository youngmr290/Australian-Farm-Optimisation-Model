# import dash
# import dash_bootstrap_components as dbc
#
# app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])
#
# server = app.server
# app.config.suppress_callback_exceptions = True


import dash
import plotly
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='initial value', type='text')
    ]),
    html.Br(),
    html.Div(id='my-output'),

])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    import Exp
    return f'Output: {input_value}'


if __name__ == '__main__':
    app.run_server(debug=True)
