import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Define the golden ratio and the complex function
phi = (1 + np.sqrt(5)) / 2  # Golden ratio

def golden_ratio_function(x):
    return np.exp(1j * x) + np.exp(1j * phi * x)

# Expand the range of x values while reducing the number of points to maintain duration
extended_range_x_values = np.linspace(-100, 100, 1000)  # Significantly more frames for smoother animation
extended_range_y_values = golden_ratio_function(extended_range_x_values)

def update_high_quality_animation(frame):
    x = extended_range_x_values[frame]
    y = extended_range_y_values[frame]
    point.set_data(y.real, y.imag)
    time_text.set_text(f'x = {x:.2f}')

    if frame > 0:
        y_prev = extended_range_y_values[frame - 1]
        ax.plot([y_prev.real, y.real], [y_prev.imag, y.imag], color=plt.cm.plasma(frame / len(extended_range_x_values)), lw=3)

    return point, time_text

# Creating the high-quality, extended animation
fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot(extended_range_y_values.real, extended_range_y_values.imag, lw=1, alpha=0.3)
point, = ax.plot([], [], 'ro', markersize=10)
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

ax.set_xlim((extended_range_y_values.real.min(), extended_range_y_values.real.max()))
ax.set_ylim((extended_range_y_values.imag.min(), extended_range_y_values.imag.max()))
ax.set_xlabel('Real Part')
ax.set_ylabel('Imaginary Part')
ax.set_title('High-Quality Extended Animation of e^(ix) + e^(i phi x)')

high_quality_ani = FuncAnimation(fig, update_high_quality_animation, frames=len(extended_range_x_values), blit=True)

# Save the high-quality animation as a video file
high_quality_ani_file = 'golden_ratio_function_animation.mp4'
high_quality_ani.save(high_quality_ani_file, writer='ffmpeg', fps=20)
