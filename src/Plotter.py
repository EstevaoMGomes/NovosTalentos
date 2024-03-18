import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
from src.CreateCoil import CreateCoil

def plot3D(dofs: jnp.ndarray, Trajectories: jnp.ndarray = jnp.zeros(0)):
    dofs = jnp.array(dofs)
    N_coils = len(dofs)
    trace_curve = np.empty(N_coils, dtype=object)
    for i in range(N_coils):
        order = int((len(dofs[i])/3-1)/2)
        coil = CreateCoil(dofs[i], 200, order)
        trace_curve[i] = go.Scatter3d(x=coil[:,0],y=coil[:,1],z=coil[:,2], mode='lines', name=f'Coil {i+1}', line=dict(color='rgb(179,179,179)', width=4))

    # Create layout for the plot
    layout = go.Layout(scene=dict(aspectmode='cube'))

    # Create figure and add traces to it
    fig = go.Figure(data=list(trace_curve), layout=layout)

    # Add the trajectories to the plot
    for i in range(len(Trajectories)):
        for j in range(len(Trajectories[i])):
            fig.add_trace(go.Scatter3d(x=[Trajectories[i][j][0]], y=[Trajectories[i][j][1]], z=[Trajectories[i][j][2]], mode='markers', marker=dict(size=2, color=f'rgb({255/len(Trajectories)*i},100,140)'), name= f"Trajectory {i+1}"))
    
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

def plot2D(shape: str, x_axis: tuple, y_axis: tuple, title: str | tuple, x_label: str | tuple, y_label: str | tuple, filename: str):
    if shape == "1x1":
        if type(title) != str:
            raise ValueError("title must be a string")
        if type(x_label) != str:
            raise ValueError("x label must be a string")
        if type(y_label) != str:
            raise ValueError("y label must be a string")
        plt.scatter(x_axis, y_axis)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig("images/"+filename+".png")
        plt.show()
    elif shape == "1x2":
        if len(x_axis) != 2:
            raise ValueError("x_axis must have 2 elements")
        if len(y_axis) != 2:
            raise ValueError("y_axis must have 2 elements")
        if type(title) != tuple:
            raise ValueError("title must be a tuple with strings")
        if len(title) != 2:
            raise ValueError("title must have 2 elements")
        if type(x_label) != tuple:
            raise ValueError("x_label must be a tuple with strings")
        if len(x_label) != 2:
            raise ValueError("x_label must have 2 elements")
        if type(y_label) != tuple:
            raise ValueError("y_label must be a tuple with strings")
        if len(y_label) != 2:
            raise ValueError("y_label must have 2 elements")
        plt.figure()
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (1, 0))
        ax1.scatter(x_axis[0], y_axis[0])
        ax2.scatter(x_axis[1], y_axis[1])
        ax1.set_title(title[0])
        ax1.set_xlabel(x_label[0])
        ax1.set_ylabel(y_label[0])
        ax2.set_title(title[1])
        ax2.set_xlabel(x_label[1])
        ax2.set_ylabel(y_label[1])
        plt.savefig("images/"+filename+".png")
        plt.show()
    elif shape == "1x3":
        if len(x_axis) != 3:
            raise ValueError("x_axis must have 3 elements")
        if len(y_axis) != 3:
            raise ValueError("y_axis must have 3 elements")
        if type(title) != tuple:
            raise ValueError("title must be a tuple with strings")
        if len(title) != 3:
            raise ValueError("title must have 3 elements")
        if type(x_label) != tuple:
            raise ValueError("x_label must be a tuple with strings")
        if len(x_label) != 3:
            raise ValueError("x_label must have 3 elements")
        if type(y_label) != tuple:
            raise ValueError("y_label must be a tuple with strings")
        if len(y_label) != 3:
            raise ValueError("y_label must have 3 elements")
        plt.figure()
        ax1 = plt.subplot2grid((3, 1), (0, 0))
        ax2 = plt.subplot2grid((3, 1), (1, 0))
        ax3 = plt.subplot2grid((3, 1), (2, 0))
        ax1.scatter(x_axis[0], y_axis[0])
        ax2.scatter(x_axis[1], y_axis[1])
        ax3.scatter(x_axis[2], y_axis[2])
        ax1.set_title(title[0])
        ax1.set_xlabel(x_label[0])
        ax1.set_ylabel(y_label[0])
        ax2.set_title(title[1])
        ax2.set_xlabel(x_label[1])
        ax2.set_ylabel(y_label[1])
        ax3.set_title(title[2])
        ax3.set_xlabel(x_label[2])
        ax3.set_ylabel(y_label[2])
        plt.savefig("images/"+filename+".png")
        plt.show()
    else:
        raise ValueError("Shape not supported") 
    plt.close()
    return plot2D