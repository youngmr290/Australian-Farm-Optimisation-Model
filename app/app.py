from dash import Dash, dcc, html, Input, Output, callback
from Pages import Page1_home, Page2_inputs, Page3_outputs


app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/page1':
        return Page1_home.layout
    elif pathname == '/page2':
        return Page2_inputs.layout
    elif pathname == '/page3':
        return Page3_outputs.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)