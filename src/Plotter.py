import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go
import os

def plotter(N_coils: int, FourierCoefficients: list[int | float], Trajectories = jnp.zeros(0)):
    trace_curve = np.empty(N_coils, dtype=object)
    phi = np.linspace(0, 2*np.pi, 200)
    for i in range(len(FourierCoefficients)):
        trace_curve[i] = go.Scatter3d(x=FourierCoefficients[i][0]+FourierCoefficients[i][1]*jnp.cos(phi)+FourierCoefficients[i][2]*jnp.sin(phi),
                                      y=FourierCoefficients[i][3]+FourierCoefficients[i][4]*jnp.cos(phi)+FourierCoefficients[i][5]*jnp.sin(phi),
                                      z=FourierCoefficients[i][6]+FourierCoefficients[i][7]*jnp.cos(phi)+FourierCoefficients[i][8]*jnp.sin(phi),
                                      mode='lines', name=f'Curve{i+1}')

    # Create layout for the plot
    layout = go.Layout(scene=dict(aspectmode='cube'))

    # Create figure and add traces to it
    fig = go.Figure(data=list(trace_curve), layout=layout)

    # Add the trajectories to the plot
    for i in range(len(Trajectories)):
        fig.add_trace(go.Scatter3d(x=Trajectories[i][0], y=Trajectories[i][1], z=Trajectories[i][2], mode='markers', marker=dict(size=1, color='red')))
    
    fig.update_layout(showlegend=False, title=None)
    fig.update_layout(scene=dict(
        xaxis=dict(title='x'),
        yaxis=dict(title='y'),
        zaxis=dict(title='z')
    ))

    # Show the plot
    fig.show()

    # Write the plot to a pgn file
    if not os.path.exists("images"):
        os.mkdir("images")
    
    #fig.write_image("images/Trajectory.png")
