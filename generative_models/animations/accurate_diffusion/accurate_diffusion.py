from manim import *
import numpy as np

class ForwardDiffusion(Scene):
    def construct(self):
        # Configuration
        n_iterations = 100
        beta = 0.05  # Noise schedule parameter for diffusion
        
        # Create coordinate system
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=10,
            y_length=10,
            axis_config={"include_numbers": True, "font_size": 20},
        )
        axes.add_coordinates()
        
        # Add grid
        grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=10,
            y_length=10,
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        )
        
        self.add(grid, axes)
        self.wait(0.5)
        
        # Step 1: Create 3 clusters of 5 points each in the first quadrant
        cluster_centers = [
            np.array([2, 2.5, 0]),
            np.array([3, 1, 0]),
            np.array([1, 0.8, 0])
        ]
        
        clusters = []
        all_dots = VGroup()
        
        for center in cluster_centers:
            cluster_points = []
            for _ in range(5):
                # Generate points around cluster center
                offset = np.random.randn(3) * 0.3
                offset[2] = 0  # Keep z=0
                point = center + offset
                cluster_points.append(point)
                dot = Dot(point, color=BLUE, radius=0.08)
                all_dots.add(dot)
            clusters.append(cluster_points)
        
        self.play(Create(all_dots))
        self.wait(0.5)
        
        # Step 2: Select 1 point from each cluster and show images
        selected_indices = [2, 1, 3]  # Select specific points from each cluster
        selected_dots = VGroup()
        selected_positions = []
        images = Group()
        
        for i, (cluster, idx) in enumerate(zip(clusters, selected_indices)):
            point = cluster[idx]
            selected_positions.append(point.copy())
            
            # Highlight selected dot
            dot_index = i * 5 + idx
            selected_dot = all_dots[dot_index]
            selected_dots.add(selected_dot)
            
            # Create image placeholder above the point
            image = ImageMobject(f"media/images/forward_diffusion/image_{i+1}.jpg").scale(0.1)
            image.move_to(point + UP * 1.2)
            
            images.add(image)
        
        # Highlight selected dots
        self.play(
            *[dot.animate.set_color(RED).scale(1.5) for dot in selected_dots]
        )
        self.wait(0.3)
        
        # Show images
        self.play(FadeIn(images))
        self.wait(1)
        
        # Step 3: Fade out images
        self.play(FadeOut(images))
        self.wait(0.5)
        
        # Step 4 & 5: Forward diffusion iterations
        # Create paths for ALL points
        paths = [TracedPath(dot.get_center, stroke_color=YELLOW, stroke_width=1.5, stroke_opacity=0.4) 
                 for dot in all_dots]
        
        for path in paths:
            self.add(path)
        
        # Iteration counter
        iteration_text = Text(f"Iteration: 0 / {n_iterations}", font_size=24)
        iteration_text.to_edge(DOWN)
        self.add(iteration_text)
        
        # Perform diffusion iterations on ALL points
        # Flatten cluster positions
        all_positions = []
        for cluster in clusters:
            all_positions.extend([pos.copy() for pos in cluster])
        
        current_positions = all_positions
        
        for t in range(n_iterations):
            new_positions = []
            animations = []
            
            for i, (dot, pos) in enumerate(zip(all_dots, current_positions)):
                # Forward diffusion: x_{t+1} = sqrt(1 - beta) * x_t + sqrt(beta) * epsilon
                # This converges to isotropic Gaussian N(0, I) at the origin
                epsilon = np.random.randn(3)
                epsilon[2] = 0  # Keep z=0
                
                new_pos = np.sqrt(1 - beta) * pos + np.sqrt(beta) * epsilon
                new_positions.append(new_pos)
                
                # Animate movement
                animations.append(dot.animate.move_to(new_pos))
            
            # Update iteration counter
            new_text = Text(f"Iteration: {t+1} / {n_iterations}", font_size=24)
            new_text.to_edge(DOWN)
            animations.append(Transform(iteration_text, new_text))
            
            # Run all animations
            self.play(*animations, run_time=0.3)
            
            current_positions = new_positions
        
        self.wait(2)