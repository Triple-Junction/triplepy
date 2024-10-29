import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def plotField2D(field, title, dpi=100):
    [nx, ny] = field.shape
    plt.figure(figsize=(5, 5), dpi=dpi)
    plt.imshow(
        np.transpose(field), cmap="viridis", origin="lower", extent=[0, nx, 0, ny]
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()


def plotInteractiveField2D(field, title):
    [nx, ny] = field[0].shape

    # Create the initial plot
    fig, ax = plt.subplots()
    im = ax.imshow(np.transpose(field[0]), cmap='viridis', origin='lower', extent=[0, nx, 0, ny])

    # Add a slider for changing timeframes
    frame_count = field.shape[0]-1
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    time_slider = Slider(ax_slider, 'Frame', 0, frame_count, valinit=0, valstep=1)

    # Function to update the plot based on the slider value
    def update(val):
        t = int(time_slider.val)
        im.set_array(field[t].T)
        ax.set_title(title+f' at time: {t}')
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    time_slider.on_changed(update)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return time_slider  # Return the slider object


def plotTriplePoint(phia, phib, phic, dpi=200, grid=-1):
    plt.figure(figsize=(8, 8), dpi=dpi)
    if grid == -1:
        print("If you would like to specify the grid coordinates add grid=[X,Y] to function call.")
        [nx, ny] = phia.shape
        plt.imshow(np.transpose(phia+2*phib), cmap='viridis', origin='lower', extent=[0, nx, 0, ny])
        plt.contour(np.transpose(phia - phib), levels=[0.0], colors=['red'])
        plt.contour(np.transpose(phia - phic), levels=[0.0], colors=['blue'])
        plt.contour(np.transpose(phic - phib), levels=[0.0], colors=['white'])
    else:
        plt.imshow(np.transpose(phia+2*phib), cmap='viridis', origin='lower', extent=[grid[0][0, 0], grid[0][-1, -1], grid[1][0, 0], grid[1][-1, -1]])
        plt.contour(grid[0], grid[1], np.transpose(phia - phib), levels=[0.0], colors=['red'])
        plt.contour(grid[0], grid[1], np.transpose(phia - phic), levels=[0.0], colors=['blue'])
        plt.contour(grid[0], grid[1], np.transpose(phic - phib), levels=[0.0], colors=['white'])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Triple point visualization')
    plt.show()


def plotInteractiveTriplePoint(phia, phib, phic, dpi=200, m=1, f=1):
    global iso_ab
    [nx, ny] = phia[0].shape

    # Create the initial plot
    fig, ax = plt.subplots()
    im = ax.imshow(np.transpose(phia[0]+2*phib[0]), cmap='viridis', origin='lower', extent=[0, nx, 0, ny])
    ax.contour(np.transpose(phia[0] - phib[0]), levels=[0.0], colors=['red'])
    ax.contour(np.transpose(phia[0] - phic[0]), levels=[0.0], colors=['blue'])
    ax.contour(np.transpose(phic[0] - phib[0]), levels=[0.0], colors=['white'])

    # Add a slider for changing timeframes
    frame_count = phia.shape[0]-1
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    time_slider = Slider(ax_slider, 'Frame', 0, frame_count, valinit=0, valstep=1)

    # Function to update the plot based on the slider value
    def update(val):
        t = int(time_slider.val)
        im.set_array((phia[t]+2*phib[t]).T)
        for coll in ax.collections:
            coll.remove()

        ax.contour(np.transpose(phia[t] - phib[t]), levels=[0.0], colors=['red'])
        ax.contour(np.transpose(phia[t] - phic[t]), levels=[0.0], colors=['blue'])
        ax.contour(np.transpose(phic[t] - phib[t]), levels=[0.0], colors=['white'])
        ax.set_title(f'Triple point at frame: {t}')
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    time_slider.on_changed(update)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return time_slider  # Return the slider object
