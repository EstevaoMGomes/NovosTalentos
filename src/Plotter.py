import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from jax import jit
from jax.lax import fori_loop
from src.CreateCoil import CreateCoil

def plot3D(dofs: jnp.ndarray, Trajectories: jnp.ndarray = jnp.zeros(0)):
    dofs = jnp.array(dofs)
    N_coils = len(dofs)
    trace_curve = np.empty(N_coils, dtype=object)
    for i in range(N_coils):
        coil = CreateCoil(dofs[i], 200)
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
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

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


def update(num, trajectories, lines):
    for line, walk in zip(lines, trajectories):
        line.set_data(walk[:num, :2].T)
        line.set_3d_properties(walk[:num, 2])
    return lines

def plot_animation3d(
    dofs, surface, trajectories, num_steps, distance=1, show=True, save_movie=False
):
    """
    Show a three-dimensional animation of a particle
    orbit together with a flux surface of the stellarator
    """
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    dofs = jnp.array(dofs)

    #################################################################
    """
    from src.MagneticField import B
    N_CurvePoints = 100
    FC_order = int((len(dofs[0])/3-1)/2)
    curves_points = jnp.empty((2, N_CurvePoints, 3))

    @jit
    def fori_createcoil(coil: jnp.int32, curves_points: jnp.ndarray) -> jnp.ndarray:
        return curves_points.at[coil].set(CreateCoil(dofs[coil], N_CurvePoints, FC_order))
    curves_points = fori_loop(0, 2, fori_createcoil, curves_points)

    x = np.arange(-2.5, 2.6, 1.25)
    y = np.arange(-1, 1.1, 0.5)
    z = np.arange(-1, 1.1, 0.5)

    u = np.zeros(5)
    v = np.zeros(5)
    w = np.zeros(5)

    for i in range(5):
        u[i], v[i], w[i] = B(jnp.array([x[i],y[i],z[i]]), curves_points, jnp.array([1e7, 1e7]))


    x, y, z = np.meshgrid(x, y, z)
    u, v, w = np.meshgrid(u, v, w)
    ax.quiver(x, y, z, x+u, y+v, z+w, length=0.1, normalize=False, linewidth = 2, color = "tomato")
    """
    #################################################################

    
    N_coils = len(dofs)
    for i in range(N_coils):
        order = int((len(dofs[i])/3-1)/2)
        coil = CreateCoil(dofs[i], 200)
        ax.plot(coil[:,0],coil[:,1],coil[:,2], color='gray')

    if surface[0] == "cylinder":
        us = np.linspace(0, 2 * np.pi, 128)
        xs = np.linspace(surface[1], surface[2], 2)

        us, xs = np.meshgrid(us, xs)

        ys = surface[3] * np.cos(us)
        zs = surface[3] * np.sin(us)
        ax.plot_surface(xs, ys, zs, cmap='plasma', alpha=0.2)
    
    if surface[0] == "torus":
        minor_radius = surface[1]
        major_radius = surface[2]

        phi = np.linspace(0, 2 * np.pi, 64)
        theta = np.linspace(0, 2 * np.pi, 64)

        phi, theta = np.meshgrid(phi, theta)

        xs = (minor_radius*np.cos(theta)+major_radius)*np.cos(phi)
        ys = (minor_radius*np.cos(theta)+major_radius)*np.sin(phi)
        zs = minor_radius*np.sin(theta)
        ax.plot_surface(xs, ys, zs, cmap='plasma', alpha=0.2)
    
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)  # pylint: disable=W0212
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)  # pylint: disable=W0212
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)  # pylint: disable=W0212
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off()
    ax.dist = distance
    #ax.set_zlim(-2,2)
    #ax.set_xlim(-2,2)
    #ax.set_ylim(-2,2)

    #plt.show()
    #plt.savefig("images/B_Field.png", transparent=True)
    #raise SystemExit
    
    lines = [ax.plot([], [], [])[0] for _ in trajectories]

    ani = animation.FuncAnimation(
        fig,
        update,
        num_steps,
        fargs=(trajectories, lines),
        interval=num_steps / 200,
        #interval=100,
    )

    if show:
        plt.show()

    if save_movie:
        ani.save(
            filename="images/ParticleOrbit.gif",
            fps=30,
            dpi=300,
            #codec="libx264",
            #bitrate=-1,
            savefig_kwargs={"transparent": True},
        )