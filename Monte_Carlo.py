import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import cv2
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import matplotlib.patches as patches

class VPS:
    def __init__(self, image_size=(512, 512),
                 pixel_size=0.1e-6,  # meters per pixel (100 nm/pixel)
                 temperature=298,
                 viscosity=0.001,
                 particle_radius=1e-6,
                 fps=30,
                 particle_brightness=0.8,
                 background_noise=0.05,
                 blur_sigma=1.0):
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.T = temperature
        self.eta = viscosity
        self.r = particle_radius
        self.dt = 1.0 / fps
        self.fps = fps
        self.particle_brightness = particle_brightness
        self.background_noise = background_noise
        self.blur_sigma = blur_sigma

        self.k_B = 1.38064852e-23
        
        # calculating the diffusion coefficient.
        self.D = self.k_B * self.T / (6 * np.pi * self.eta * self.r)
        
        # taking the Step size in metersand  then converting it  into pixels.
        self.sigma_meters = np.sqrt(2 * self.D * self.dt)
        self.sigma_pixels = self.sigma_meters / self.pixel_size
        
        # particle radius in pixels. 
        self.particle_radius_pixels = self.r / self.pixel_size

        # should I print the values? 

    def initialize_particles(self,n_particles, distribution='random'):
        if distribution == "random":
             x = np.random.uniform(self.particle_radius_pixels,
                                   self.image_size[0] - self.particle_radius_pixels,
                                   n_particles
                                   )
             y = np.random.uniform(self.particle_radius_pixels, self.image_size[1] - self.particle_radius_pixels, n_particles)
        elif distribution == "center":
            center_x, center_y = self.image_size[0]/2, self.image_size[1]/2
            spread = min(self.image_size) * 0.1
            x = np.random.normal(center_x, spread, n_particles)
            y = np.random.normal(center_y, spread, n_particles)
        elif distribution == "grid" :
            grid_size = int(np.ceil(np.sqrt(n_particles)))
            spacing_x = self.image_size[0] / (grid_size + 1)
            spacing_y = self.image_size[1] / (grid_size + 1)
            
            x = []
            y = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(x) < n_particles:
                        x.append((i + 1) * spacing_x)
                        y.append((j + 1) * spacing_y)
            
            x = np.array(x)
            y = np.array(y)
        
        return np.column_stack([x, y])
    def update_particle_positions(self, positions):
        n_particles = positions.shape[0]

        # generating randome steps in pixels: 
        steps = np.random.normal(0, self.sigma_pixels, (n_particles, 2)) # generating random steps

        new_positions = positions + steps # adding the steps to initial positions. 
        # left or right boundaries. 
        out_left = new_positions[:, 0] < self.particle_radius_pixels
        out_right = new_positions[:, 0] > self.image_size[0] - self.particle_radius_pixels
        # keeping the particles in boundaries. 
        new_positions[out_left, 0] = 2 * self.particle_radius_pixels - new_positions[out_left, 0]
        new_positions[out_right, 0] = 2 * (self.image_size[0] - self.particle_radius_pixels) - new_positions[out_right, 0]
        
        out_top = new_positions[:, 1] < self.particle_radius_pixels
        out_bottom = new_positions[:, 1] > self.image_size[1] - self.particle_radius_pixels
        new_positions[out_top, 1] = 2 * self.particle_radius_pixels - new_positions[out_top, 1]
        new_positions[out_bottom, 1] = 2 * (self.image_size[1] - self.particle_radius_pixels) - new_positions[out_bottom, 1]
        
        return new_positions
    def render_frame(self, positions, frame_number = 0):
        # making a blank image : 
        image = np.zeros(self.image_size, dtype=np.float64)

        # just in case - background noise
        if self.background_noise>0:
            noise = np.random.normal(0, self.background_noise, self.image_size)
            image += noise

        x_coords, y_coords = np.mgrid[0:self.image_size[0], 0:self.image_size[1]]

        for i, (px,py) in enumerate(positions):
            distance = np.sqrt((x_coords - px)**2 + (y_coords - py)**2) 

            # smooth particles with gaussian profiles: 
            particle_mask = np.exp(-(distance**2) / (2 * (self.particle_radius_pixels/2)**2))
            image += self.particle_brightness * particle_mask

            #image blur
        if self.blur_sigma > 0:
            image = gaussian_filter(image, sigma=self.blur_sigma)

        image = np.clip(image, 0, 1)

        return image 

    def simulate_video(self, n_particles, n_frames, output_filename=None, 
                      show_trails=False, particle_distribution='random'):  
        
        positions = self.initialize_particles(n_particles, particle_distribution)

        frames = []
        all_positions = []
        
        for frame in range(n_frames):
            if frame % 50 == 0:
                print(f"Frame {frame}/{n_frames}")
            
            image = self.render_frame(positions, frame) # rendering the current frame
            frames.append(image)
            all_positions.append(positions.copy())

            if frame < n_frames - 1: # updating the next frame
                positions = self.update_particle_positions(positions)
            
        frames = np.array(frames)
        all_positions = np.array(all_positions)

        if output_filename:
            self.save_video(frames, output_filename)
        
        return frames, all_positions
    
    def save_video(self, frames, filename):
        h, w = frames.shape[1], frames.shape[2]
        codec = 'mp4v' if filename.lower().endswith('.mp4') else 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(filename, fourcc, self.fps, (w, h), isColor=True)
        for f in frames:
            frame_u8 = (f * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
            writer.write(frame_bgr)
        writer.release()

    def create_matplotlib_animation(self, frames, all_positions, show_trails=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # setting up image display: 
        im = ax1.imshow(frames[0], cmap='grey', vmin=0, vmax=1, origin='upper')
        ax1.set_title('Simulated Particles')
        ax1.set_xlabel(f'X (pixels) | {self.image_size[0]*self.pixel_size*1e6:.1f} micrometer total')
        ax1.set_ylabel(f'Y (pixels) | {self.image_size[1]*self.pixel_size*1e6:.1f} micrometer total')

        # trajectory plot. 
        ax2.set_xlim(0, self.image_size[0])
        ax2.set_ylim(0, self.image_size[1])
        ax2.set_aspect('equal')
        ax2.set_title('Particle Trajectories')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')

        n_particles = all_positions.shape[1]
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_particles, 10)))
        trail_lines = []
        particle_dots = []

        for i in range(min(n_particles, 10)):  # Show max 10 trails for clarity
            line, = ax2.plot([], [], '-', color=colors[i % len(colors)], alpha=0.7)
            dot, = ax2.plot([], [], 'o', color=colors[i % len(colors)], markersize=8)
            trail_lines.append(line)
            particle_dots.append(dot)
        
        frame_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                             fontsize=12, verticalalignment='top', 
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def animate(frame_idx):
            im.set_array(frames[frame_idx])

            # updating trajectories:

            if show_trails:
                for i, (line, dot) in enumerate(zip(trail_lines, particle_dots)):
                    if i < n_particles:
                        x_trail = all_positions[:frame_idx+1, i, 0]
                        y_trail = all_positions[:frame_idx+1, i, 1]
                        line.set_data(x_trail, y_trail)
                    
                    # current position 
                    if len(x_trail) > 0:
                        dot.set_data([x_trail[-1]], [y_trail[-1]])

            time_s = frame_idx/ self.fps
            frame_text.set_text(f'frame: {frame_idx}\nTime: {time_s:.2f}s')

            return [im] + trail_lines + particle_dots + [frame_text]
        ani = FuncAnimation(fig, animate, frames=len(frames), 
                          interval=1000/self.fps, blit=True, repeat=True)
        
        plt.tight_layout( )
        return ani
    
    def analyze_simulation(self, all_positions):
        if all_positions.ndim != 3 or all_positions.shape[2] != 2:
            raise ValueError(f"all_positions must be (T, N, 2); got {all_positions.shape}")
        T, N, _ = all_positions.shape # T = time steps, N = number of particles and _X,Y = 2d coordinates. 

        # calculating Meas Squared Displacement
        
        msd = np.zeros(T)
        for t in range(T):
            displacements = all_positions[t] - all_positions[0] # from initial positions. 
            # Convert to meters
            displacements_m = displacements * self.pixel_size
            # calcuating x^2 + y^2 for wach particle
            squared_disps = np.sum(displacements_m**2, axis=1)
            # averaging over all particles. 
            msd[t] = np.mean(squared_disps)

        # timepoints: 
        time_points = np.arange(T)/self.fps

        msd_theory = 4 * self.D * time_points
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # MSD comparison
        ax1.plot(time_points, msd * 1e12, 'b-', linewidth=2, label='Simulation')
        ax1.plot(time_points, msd_theory * 1e12, 'r--', linewidth=2, label='Theory')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('MSD (mm^2)')
        ax1.set_title('Mean Squared Displacement')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        n_show = min(5, N) # we chose to track 5 particles. 
        for i in range(n_show):
            x_um = all_positions[:, i, 0] * self.pixel_size * 1e6
            y_um = all_positions[:, i, 1] * self.pixel_size * 1e6
            ax2.plot(x_um, y_um, alpha=0.7, linewidth=1)
        ax2.set_xlabel('X (µm)')
        ax2.set_ylabel('Y (µm)')
        ax2.set_title(f'Sample Trajectories (showing {n_show})')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')

        all_steps = []
        for p in range(N): # for all the particles. 
            steps_pixels = np.diff(all_positions[:, p, :], axis=0) # calculating positive differences between consecutive positions (n+1 - (n))
            steps_m = steps_pixels * self.pixel_size # converting pixels to meters
            step_sizes = np.sqrt(np.sum(steps_m**2, axis=1)) # calculating the magnitude of step. sqrt(x^2 + y^2)
            all_steps.extend(step_sizes) # making all the steps into one list. 
        
        ax3.hist(np.array(all_steps) * 1e9, bins=50, density=True, alpha=0.7)
        ax3.set_xlabel('Step size (nm)')
        ax3.set_ylabel('Probability density')
        ax3.set_title('Step Size Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Final frame with scale bar
        final_frame = self.render_frame(all_positions[-1])
        ax4.imshow(final_frame, cmap='gray', vmin=0, vmax=1)
        
        # Add scale bar
        scale_bar_length_um = 5  # 5 µm scale bar
        scale_bar_pixels = scale_bar_length_um * 1e-6 / self.pixel_size
        bar_x = self.image_size[0] * 0.8
        bar_y = self.image_size[1] * 0.9
        
        ax4.plot([bar_x, bar_x + scale_bar_pixels], [bar_y, bar_y], 
                'white', linewidth=4)
        ax4.text(bar_x + scale_bar_pixels/2, bar_y - 20, f'{scale_bar_length_um} µm',
                ha='center', va='top', color='white', fontsize=10, fontweight='bold')
        
        ax4.set_title('Final Frame with Scale')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nSimulation results:")
        print(f"Theoretical step size: {self.sigma_pixels:.2f} pixels ({self.sigma_meters*1e9:.1f} nm)")
        print(f"Observed mean step size: {np.mean(all_steps)*1e9:.1f} nm")
        print(f"Particle radius: {self.particle_radius_pixels:.1f} pixels ({self.r*1e6:.1f} µm)")

# Example usage
def run_example_simulation():
    """Run an example simulation matching typical experimental conditions"""
    
    # Create simulator
    simulator = VPS(
        image_size=(512, 512),      # 512x512 pixel image
        pixel_size=100e-9,          # 100 nm per pixel
        temperature=298,            # Room temperature
        viscosity=0.001,            # Waterx
        particle_radius=500e-9,     # 500 nm particles
        fps=30,                     # 30 fps video
        particle_brightness=0.7,    # Bright particles
        background_noise=0.02,      # Small amount of noise
        blur_sigma=0.5              # Slight optical blur
    )
    
    # Run simulation
    print("\nRunning example simulation:")
    frames, positions = simulator.simulate_video(
        n_particles=20,
        n_frames=300,  # 10 seconds at 30 fps
        output_filename="brownian_particles.mp4",
        particle_distribution='random'
    )
    
    # Create interactive animation
    ani = simulator.create_matplotlib_animation(frames, positions, show_trails=True)
    #ani.save('matplotlib_animation.mp4', writer='ffmpeg', fps=simulator.fps)
    
    # Analyze results
    simulator.analyze_simulation(positions)
    
    plt.show()
    
    return simulator, frames, positions

if __name__ == "__main__":
    # Run example
    simulator, frames, positions = run_example_simulation()
    
    print("\n VIdeo results:")
    print("- Video saved as 'brownian_particles.mp4'")
    print("- Position data shape:", positions.shape)
    print("- Frames shape:", frames.shape)


    

        


        