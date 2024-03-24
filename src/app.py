#!/usr/bin/env python
# coding: utf-8

# In[7]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
suppress_callback_exceptions=True

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Define CSS styles
styles = {
    'fontFamily': 'Arial, sans-serif',  # Change font type
    'color': '#333',  # Change font color
    'backgroundColor': '#f0f0f0'  # Change background color
}
button_style = {
    'backgroundColor': '#4CAF50',  # Green background color
    'border': '2px solid #808080', # Grey border
    'color': 'white',              # White text color
    'padding': '15px 32px',        # Padding
    'text-align': 'center',        # Center text
    'text-decoration': 'none',     # No text decoration
    'display': 'inline-block',     # Display as inline block
    'font-size': '16px',           # Font size
    'margin': '4px 2px',           # Margin
    'cursor': 'pointer',           # Cursor style
    'transition': 'background-color 0.3s, border-color 0.3s'  # Transition effect
}

# Define CSS styles for disabled button
disabled_button_style = {
    'backgroundColor': '#d3d3d3',  # Light gray background color
    'border': 'none',              # No border
    'color': '#a9a9a9',            # Light gray text color
    'padding': '15px 32px',        # Padding
    'text-align': 'center',        # Center text
    'text-decoration': 'none',     # No text decoration
    'display': 'inline-block',     # Display as inline block
    'font-size': '16px',           # Font size
    'margin': '4px 2px',           # Margin
    'cursor': 'not-allowed',       # Cursor style set to "not-allowed" to indicate button is disabled
    'pointer-events': 'none'       # Disable pointer events to prevent interaction
}

app.layout = html.Div([
    html.H2("Welcome to this IUPAC-recommended tool for the estimation of reactivity ratios"),
    html.Br(),
    html.P([
        "This tool allows you to upload a CSV file, set a number of parameters, and then calculates the best point estimate of reactivity ratios (r",
        html.Sub("1"),
        ", r",
        html.Sub("2"),
        ") and the corresponding 95% joint confidence interval (JCI)."]),
    html.P([
        "The CSV file should be formatted as follows:"
    ]),
    html.Ul([
        html.Li("The file should not have headers."),
        html.Li("Column 1: Initial fraction of monomer 1 in the reaction mixture (f10)."),
        html.Li("Column 2: Overall monomer conversion at data point (conv)."),
        html.Li(["Column 3: Copolymer composition, expressed as fraction of monomer 1 in the copolymer (F",
        html.Sub("1"),
        ")."]),
        html.Li("Column 4: Absolute error in copolymer composition (deltaF).")
    ]),

    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Your CSV File.')]),
        style={
            'width': '50%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    html.Br(),
    html.P("The weighing of errors in the calculation can be done in one of two different ways. The defauls way is to base the weighing on the errors given in your CSV file. As an alternative, a uniform weighing of the errors can be adopted. You can check the method of your choice with the radiobuttons below."),
    dcc.RadioItems(
        id='weighting-choice',
        options=[
            {'label': 'Weighting based on given errors in F', 'value': 1},
            {'label': 'Uniform weighting', 'value': 2}
        ],
        value=1  # Default value
    ),
    html.Br(),
    html.P([
        "In order to determine the r-values that best describe your copolymerization, you are required to provide a range of r-values for which the program calculates the residual sum of squares. The way this input needs to be provided is that you give an r",
        html.Sub("1"),
        " around the value that you expect. Next, you define an r1range, which is the width of the range of r",
        html.Sub("1"),
        " values. For example, if you choose r",
        html.Sub("1"),
        " = 0.5, and r1range = 0.6, the sum of squares will be calculated for 0.2 < r",
        html.Sub("1"),
        " < 0.8. You will do the same for r",
        html.Sub("2"),
        " and r2range."
        ]),
    html.P("After setting the r-value ranges, the submit button becomes available. After submitting the data, it may take a few minutes before the result of the calculation is shown."),
        html.Div([
        html.Div([
            dcc.Input(id='r1', type='number', placeholder='r1'),
            dcc.Input(id='r1range', type='number', placeholder='r1range'),
        ], style={'padding': 10}),
        html.Div([
            dcc.Input(id='r2', type='number', placeholder='r2'),
            dcc.Input(id='r2range', type='number', placeholder='r2range'),
        ], style={'padding': 10}),
        ]),
    html.Br(),
    html.Button('Submit', style=disabled_button_style, id='submit-button', n_clicks=0, disabled=True),
    html.Br(),
    html.Div(id='graph-container', style={'display': 'none'}, children=[dcc.Graph(id='output-data-plot')]),
    html.Div(id='message-output', style={'margin': '20px', 'padding': '10px', 'border': '1px solid #ddd'})
], style = styles)

@app.callback(
    Output('container-button-basic', 'children'),
    Input('weighting-choice', 'value')
)
def update_output(value):
    global errscheme
    errscheme = value
    return 'The error scheme is {}'.format(errscheme)    

@app.callback(
    [Output('submit-button', 'style'),
     Output('submit-button', 'disabled')],
    [Input('upload-data', 'contents'),
     Input('r1', 'value'),
     Input('r1range', 'value'),
     Input('r2', 'value'),
     Input('r2range', 'value'),
     Input('submit-button', 'n_clicks')],
    [State('submit-button', 'style')]
)
def update_button_state(contents, r1, r1range, r2, r2range, n_clicks, current_style):
    if contents and all([r1, r1range, r2, r2range]):
        return button_style, False  # Enable the button and apply the regular button style
    elif n_clicks and not current_style.get('backgroundColor') == '#45a049':
        return button_style, False  # Enable the button and apply the regular button style if it has been clicked
    else:
        return disabled_button_style, True  # Disable the button and apply the disabled button style
    
@app.callback(
    Output('Fpol1-output', 'children'),
    [Input('submit-button', 'n_clicks')],  # Corrected to the actual button ID
    [State('f10', 'value'),
     State('conv', 'value'),
     State('r1', 'value'),
     State('r2', 'value')]
)

def calculate_Fpol1(f10, conv, r1, r2, steps = 100):
    """
    Calculate Fpol1, the average copolymer composition over the indicated conversion interval.

    Parameters:
    f10 (float): Initial fraction of monomer 1.
    conv (float): Conversion up to which the average is to be calculated.
    r1, r2 (float): Reactivity ratios of monomer 1 and monomer 2.
    steps (int): Number of steps to use for numerical integration.

    Returns:
    float: Average copolymer composition Fpol1 over the conversion interval.
    """
    delta_conv = conv / steps  # Small increment in conversion for numerical integration
    F1_sum = 0  # Sum of f1 values over each step for integration
    f1 = f10
    f2 = 1 - f10
    f1u = f1
    f2u = f2
    
    for step in range(steps):
        # Calculate the midpoint conversion for this step
        midpoint_conv = (step + 0.5) * delta_conv
        # Calculate the fraction of monomer 1 at this conversion
        F1 = (r1 * f1 ** 2 + f1 * f2) / (r1 * f1 ** 2 + 2 * f1 * f2 + r2 * f2 ** 2)
        f1u = f1u - F1 * delta_conv
        f2u = f2u - (1 - F1) * delta_conv
        f1 = f1u / (f1u + f2u)
        f2 = 1 - f1
        # Sum the contributions
        F1_sum += F1
    
    # Average copolymer composition over the interval
    Fpol1 = F1_sum / steps
    
    return Fpol1

@app.callback(
    [Output('output-data-plot', 'figure'),  # Graph figure
     Output('message-output', 'children'),  # Text message
     Output('graph-container', 'style')],  # Style of the graph container to control visibility
    [Input('submit-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('r1', 'value'),
     State('r1range', 'value'),
     State('r2', 'value'),
     State('r2range', 'value')]
)
def update_graph(n_clicks, contents, r1, r1range, r2, r2range):
    if n_clicks < 1 or contents is None:
        return dash.no_update, dash.no_update, {'display': 'none'}
    errscheme = update_output
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
    
    # Extracting f10, conv, and gF
    f10, conv, gF, deltaF = df[0].values, df[1].values, df[2].values, df[3].values

    # Calculate error structure
    if errscheme ==1:
        w = 1 / deltaF **2
    else:
        w = 1
    rtheor = np.sum(w * deltaF **2)    
    
    # Prepare r1 and r2 grid
    r1_values = np.linspace(r1 - r1range/2, r1 + r1range/2, 100)  # Adjust 100 to change resolution
    r2_values = np.linspace(r2 - r2range/2, r2 + r2range/2, 100)  # Adjust 100 to change resolution
    R1, R2 = np.meshgrid(r1_values, r2_values)
    SSR = np.zeros(R1.shape)
    
    for i in range(len(r1_values)):
        for j in range(len(r2_values)):
            # Placeholder for Fpol1 calculation based on f10, conv, and the r1, r2 combination
            # Fpol1 = calculate_Fpol1(f10, conv, r1_values[i], r2_values[j])
            # For this example, let's assume Fpol1 = f10 * r1_values[i] / (r2_values[j] + 1) as a placeholder
            
            # Actual calculation should replace the line below
            Fpol1 = calculate_Fpol1(f10, conv, r1_values[i], r2_values[j])
            
            # Calculate SSR for this r1, r2 combination
            SSR[i, j] = np.sum(w * (Fpol1 - gF) ** 2)
    
    # Find the index of the minimum SSR value
    min_ssr_index = np.argmin(SSR)

    # Convert the flat index back to 2D indices
    min_i, min_j = np.unravel_index(min_ssr_index, SSR.shape)
    deltamin = SSR[min_i, min_j]
    l95 = deltamin + 5.99147 * rtheor / (len(f10) - 2)

    # Use these indices to find the corresponding R1 and R2 values
    min_r1 = r1_values[min_i]
    min_r2 = r2_values[min_j]

    formatted_min_r1 = format(min_r1, '.3f')
    formatted_min_r2 = format(min_r2, '.3f')

    # Plotting
  #  fig = go.Figure(data=[go.Surface(z=SSR, x=R1, y=R2)])
  #  fig.update_layout(title='SSR Surface Plot', autosize=True,
  #  scene=dict(xaxis_title='r1', yaxis_title='r2', zaxis_title='SSR'))

    fig = go.Figure(data= 
                    go.Contour(
                        z=SSR, 
                        x=r1_values, 
                        y=r2_values,
                        contours=dict(
                            start=l95, end=l95, size=0.1,  # Setting the start and end to min_val to get a single contour level
                            coloring='lines',  # Only show lines for contours
                            showlabels=False,  # Show labels on contours
                            ),
                        line=dict(  # Customize the appearance of the contour lines
                            color='black',  # Specific color for the contour line
                            width=2,  # Optional: Adjust the line width
                            ),
                        showscale=False  # This hides the color scale (color bar) on the right
                        )
                   )                    

    # Add a scatter plot to mark the minimum Z value
    fig.add_trace(go.Scatter(
        x=[min_r1], 
        y=[min_r2],
        mode='markers',
        marker=dict(color='red', size=10),
        showlegend=False,
        ))  

    # Update the layout for a 2D plot
    fig.update_layout(
        title='Point estimate plus 95% JCI',
        width=600,
        height=450,
        xaxis_title='r1',
        yaxis_title='r2',
        plot_bgcolor='white',  # Set background color of the plot
        )
    
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
        )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

        
    message = f"The r-values corresponding to the minimum SSR value are: r1 = {formatted_min_r1}, r2 = {formatted_min_r2}"   
    return fig, message, {'display': 'block'}


if __name__ == '__main__':
    app.run_server(debug=True)


# #### 

# In[ ]:





# In[ ]:





# In[ ]:




