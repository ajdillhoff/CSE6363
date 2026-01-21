from manim import *
import numpy as np

class DiffusionTrainingObjective(Scene):
    def construct(self):
        # Create coordinate system
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=8,
            y_length=8,
            axis_config={"include_numbers": True, "font_size": 20},
        )
        axes.to_edge(LEFT, buff=0.5)
        
        grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=8,
            y_length=8,
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).move_to(axes.get_center())
        
        self.add(grid, axes)
        
        # Title
        title = MathTex(
            r"\text{Training Objective: } \arg\min_{f} \mathbb{E}_{x_{t-1}, \eta} \left[ \| f(x_{t-1} + \eta_t) - x_{t-1} \|^2 \right]",
            font_size=28
        )
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Legend area on the right
        legend_y = 2
        legend_x = 5.5
        
        # Show multiple training examples
        n_examples = 5
        np.random.seed(42)
        
        for example in range(n_examples):
            # Generate a clean data point x_{t-1} from first quadrant
            x_clean = np.array([
                np.random.uniform(0.5, 2.5),
                np.random.uniform(0.5, 2.5),
                0
            ])
            
            # Step 1: Show clean point x_{t-1}
            clean_dot = Dot(x_clean, color=GREEN, radius=0.12)
            clean_label = MathTex(r"x_{t-1}", color=GREEN, font_size=30)
            clean_label.next_to(clean_dot, UP, buff=0.2)
            
            description = Text(
                f"Example {example + 1}: Start with clean data point",
                font_size=24,
                color=GREEN
            )
            description.to_edge(RIGHT).shift(UP * 2)
            
            self.play(
                Create(clean_dot),
                Write(clean_label),
                FadeIn(description)
            )
            self.wait(1)
            
            # Step 2: Sample noise η_t
            noise = np.random.randn(3) * 0.8
            noise[2] = 0
            
            noise_arrow = Arrow(
                clean_dot.get_center(),
                clean_dot.get_center() + noise,
                color=YELLOW,
                buff=0,
                stroke_width=4
            )
            noise_label = MathTex(r"\eta_t \sim \mathcal{N}(0, I)", color=YELLOW, font_size=28)
            noise_label.next_to(noise_arrow, RIGHT, buff=0.1)
            
            step_desc = Text(
                "Add Gaussian noise η_t",
                font_size=24,
                color=YELLOW
            )
            step_desc.to_edge(RIGHT).shift(UP * 1)
            
            self.play(
                FadeOut(description),
                GrowArrow(noise_arrow),
                Write(noise_label),
                FadeIn(step_desc)
            )
            self.wait(1)
            
            # Step 3: Create noisy point x_t = x_{t-1} + η_t
            x_noisy = x_clean + noise
            noisy_dot = Dot(x_noisy, color=RED, radius=0.12)
            noisy_label = MathTex(r"x_t = x_{t-1} + \eta_t", color=RED, font_size=30)
            noisy_label.next_to(noisy_dot, DOWN, buff=0.2)
            
            noisy_desc = Text(
                "Noisy observation x_t",
                font_size=24,
                color=RED
            )
            noisy_desc.to_edge(RIGHT).shift(UP * 0)
            
            self.play(
                FadeOut(step_desc),
                Create(noisy_dot),
                Write(noisy_label),
                FadeIn(noisy_desc)
            )
            self.wait(1)
            
            # Step 4: Network prediction f(x_t)
            # Simulate imperfect denoising
            prediction_noise = np.random.randn(3) * 0.3
            prediction_noise[2] = 0
            x_pred = x_clean + prediction_noise
            
            pred_dot = Dot(x_pred, color=BLUE, radius=0.12)
            pred_label = MathTex(r"f(x_t)", color=BLUE, font_size=30)
            pred_label.next_to(pred_dot, LEFT, buff=0.2)
            
            pred_arrow = Arrow(
                noisy_dot.get_center(),
                pred_dot.get_center(),
                color=BLUE,
                buff=0.12,
                stroke_width=4
            )
            
            pred_desc = Text(
                "Network predicts clean point f(x_t)",
                font_size=24,
                color=BLUE
            )
            pred_desc.to_edge(RIGHT).shift(DOWN * 1)
            
            self.play(
                FadeOut(noisy_desc),
                Create(pred_dot),
                Write(pred_label),
                GrowArrow(pred_arrow),
                FadeIn(pred_desc)
            )
            self.wait(1)
            
            # Step 5: Show error/loss
            error_arrow = Arrow(
                pred_dot.get_center(),
                clean_dot.get_center(),
                color=PURPLE,
                buff=0.12,
                stroke_width=6
            )
            
            # Calculate loss
            error_vec = x_clean - x_pred
            loss_value = np.linalg.norm(error_vec[:2])**2
            
            loss_text = MathTex(
                r"\| f(x_t) - x_{t-1} \|^2 = " + f"{loss_value:.3f}",
                color=PURPLE,
                font_size=32
            )
            loss_text.to_edge(RIGHT).shift(DOWN * 2)
            
            error_desc = Text(
                "Loss: Distance from prediction to truth",
                font_size=24,
                color=PURPLE
            )
            error_desc.to_edge(RIGHT).shift(DOWN * 3)
            
            self.play(
                FadeOut(pred_desc),
                GrowArrow(error_arrow),
                Write(loss_text),
                FadeIn(error_desc)
            )
            self.wait(2)
            
            # Clear for next example
            if example < n_examples - 1:
                fade_out_desc = Text(
                    "Training on next example...",
                    font_size=24,
                    color=WHITE
                )
                fade_out_desc.to_edge(RIGHT)
                
                self.play(
                    FadeOut(VGroup(
                        clean_dot, clean_label,
                        noise_arrow, noise_label,
                        noisy_dot, noisy_label,
                        pred_dot, pred_label, pred_arrow,
                        error_arrow, loss_text, error_desc
                    )),
                    FadeIn(fade_out_desc)
                )
                self.wait(0.5)
                self.play(FadeOut(fade_out_desc))
        
        # Final summary
        summary = VGroup(
            Text("Training Process:", font_size=28, color=YELLOW),
            Text("1. Sample clean point x_{t-1} from data", font_size=20),
            Text("2. Add Gaussian noise η_t", font_size=20),
            Text("3. Network f predicts clean point", font_size=20),
            Text("4. Minimize ||f(x_t) - x_{t-1}||²", font_size=20),
            Text("5. Repeat for many examples", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        summary.to_edge(RIGHT)
        
        self.play(FadeIn(summary))
        self.wait(3)