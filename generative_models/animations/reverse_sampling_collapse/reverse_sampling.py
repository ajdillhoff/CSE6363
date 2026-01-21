from manim import *
import numpy as np

class ReverseSamplingCollapse(Scene):
    def construct(self):
        # Create coordinate system
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=9,
            y_length=9,
            axis_config={"include_numbers": True, "font_size": 20},
        )
        
        grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=9,
            y_length=9,
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).move_to(axes.get_center())
        
        self.add(grid, axes)
        
        # Title
        title = Text("Reverse Sampling: With vs Without Variance", font_size=32)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Part 1: WITHOUT time-varying variance (deterministic)
        subtitle1 = Text("Without Time-Varying Variance (Deterministic)", font_size=24, color=RED)
        subtitle1.next_to(title, DOWN)
        self.play(Write(subtitle1))
        self.wait(1)
        
        # Start with multiple noisy samples around origin
        n_samples = 8
        np.random.seed(42)
        initial_noise_scale = 2.5
        
        noisy_positions = []
        noisy_dots = VGroup()
        paths_deterministic = []
        
        for i in range(n_samples):
            pos = np.random.randn(3) * initial_noise_scale
            pos[2] = 0
            noisy_positions.append(pos)
            
            dot = Dot(pos, color=RED, radius=0.1)
            noisy_dots.add(dot)
            
            # Create traced paths
            path = TracedPath(dot.get_center, stroke_color=RED, stroke_width=2, stroke_opacity=0.6)
            paths_deterministic.append(path)
            self.add(path)
        
        self.play(Create(noisy_dots))
        
        label = Text("Starting from noise", font_size=20, color=RED)
        label.to_edge(DOWN).shift(UP * 0.5)
        self.play(FadeIn(label))
        self.wait(1)
        
        # Simulate reverse process WITHOUT variance
        # All samples move toward the same mean prediction
        n_steps = 25
        target_mean = np.array([1.5, 1.5, 0])  # The learned mean
        
        current_positions = [pos.copy() for pos in noisy_positions]
        
        for step in range(n_steps):
            animations = []
            new_positions = []
            
            for i, (dot, pos) in enumerate(zip(noisy_dots, current_positions)):
                # Deterministic: just move toward the mean with no noise
                # x_{t-1} = μ_θ(x_t, t) with NO variance term
                alpha = 0.15  # Step size toward mean
                new_pos = pos + alpha * (target_mean - pos)
                new_positions.append(new_pos)
                
                animations.append(dot.animate.move_to(new_pos))
            
            self.play(*animations, run_time=0.2)
            current_positions = new_positions
        
        collapse_label = Text("All samples collapse to the mean!", font_size=22, color=RED)
        collapse_label.to_edge(DOWN).shift(UP * 0.5)
        self.play(Transform(label, collapse_label))
        self.wait(2)
        
        # Highlight the collapse
        mean_circle = Circle(radius=0.3, color=YELLOW, stroke_width=4)
        mean_circle.move_to(target_mean)
        self.play(Create(mean_circle))
        self.wait(1)
        
        # Clear for part 2
        self.play(
            FadeOut(noisy_dots),
            FadeOut(label),
            FadeOut(mean_circle),
            *[FadeOut(path) for path in paths_deterministic]
        )
        self.wait(0.5)
        
        # Part 2: WITH time-varying variance (stochastic)
        subtitle2 = Text("With Time-Varying Variance (Stochastic)", font_size=24, color=GREEN)
        subtitle2.next_to(title, DOWN)
        self.play(Transform(subtitle1, subtitle2))
        self.wait(1)
        
        # Reset samples
        noisy_positions_2 = []
        noisy_dots_2 = VGroup()
        paths_stochastic = []
        
        np.random.seed(42)  # Same initial positions
        for i in range(n_samples):
            pos = np.random.randn(3) * initial_noise_scale
            pos[2] = 0
            noisy_positions_2.append(pos)
            
            dot = Dot(pos, color=GREEN, radius=0.1)
            noisy_dots_2.add(dot)
            
            # Create traced paths
            path = TracedPath(dot.get_center, stroke_color=GREEN, stroke_width=2, stroke_opacity=0.6)
            paths_stochastic.append(path)
            self.add(path)
        
        self.play(Create(noisy_dots_2))
        
        label2 = Text("Starting from noise", font_size=20, color=GREEN)
        label2.to_edge(DOWN).shift(UP * 0.5)
        self.play(FadeIn(label2))
        self.wait(1)
        
        # Simulate reverse process WITH variance
        # Samples maintain diversity through stochastic sampling
        current_positions_2 = [pos.copy() for pos in noisy_positions_2]
        
        for step in range(n_steps):
            animations = []
            new_positions = []
            
            # Time-varying variance (decreases as we denoise)
            t = 1.0 - (step / n_steps)  # Goes from 1.0 to 0.0
            variance_scale = 0.3 * t  # Variance decreases with time
            
            for i, (dot, pos) in enumerate(zip(noisy_dots_2, current_positions_2)):
                # Stochastic: x_{t-1} = μ_θ(x_t, t) + σ_t * z, where z ~ N(0, I)
                alpha = 0.15
                mean_pred = pos + alpha * (target_mean - pos)
                
                # Add time-varying noise
                noise = np.random.randn(3) * variance_scale
                noise[2] = 0
                new_pos = mean_pred + noise
                
                new_positions.append(new_pos)
                animations.append(dot.animate.move_to(new_pos))
            
            self.play(*animations, run_time=0.2)
            current_positions_2 = new_positions
        
        diversity_label = Text("Samples maintain diversity!", font_size=22, color=GREEN)
        diversity_label.to_edge(DOWN).shift(UP * 0.5)
        self.play(Transform(label2, diversity_label))
        self.wait(2)
        
        # Show the cluster has spread
        cluster_circle = Circle(radius=0.6, color=YELLOW, stroke_width=4)
        cluster_circle.move_to(target_mean)
        self.play(Create(cluster_circle))
        self.wait(1)