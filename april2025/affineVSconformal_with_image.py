from manimlib import *
import numpy as np
from numpy.linalg import inv
import os

class AffineVsConformalImageDemo(Scene):
    def construct(self):
        # Title
        title = Text("Affine vs Conformal Transformations with Image", font="sans-serif")
        title.scale(1.2)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Load the image using relative path
        image_path = "april2025/CaffetteriaMango.jpg"
        
        # Create placeholders first in case image loading fails
        image = Rectangle(height=3, width=4, color=WHITE, fill_opacity=0.2)
        image.to_edge(LEFT)
        image_text = Text("Image", font="sans-serif").scale(0.5)
        image_text.move_to(image.get_center())
        image = VGroup(image, image_text)
        
        # Now try to load the actual image
        try:
            loaded_image = ImageMobject(image_path)
            loaded_image.scale(0.5)
            loaded_image.to_edge(LEFT)
            image = loaded_image
            print("Image loaded successfully")
        except Exception as e:
            print(f"Error loading image: {e}")
            # We'll use the placeholder created above
        
        # Create a grid for demonstrating transformations
        grid = NumberPlane(
            x_range=(-5, 5, 1),
            y_range=(-3, 3, 1),
            background_line_style={
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        )
        
        # Create some shapes for demonstration
        square = Square(side_length=1, color=BLUE).shift(RIGHT * 2)
        circle = Circle(radius=0.5, color=RED).shift(RIGHT * 3.5)
        
        # Create crosses to demonstrate angle preservation
        cross = VGroup(
            Line(LEFT * 0.5, RIGHT * 0.5, color=YELLOW),
            Line(DOWN * 0.5, UP * 0.5, color=YELLOW)
        )
        crosses = VGroup(
            cross.copy().shift(RIGHT * 2 + UP * 1),
            cross.copy().shift(RIGHT * 3.5 + DOWN * 0.5)
        )
        
        shapes = VGroup(square, circle, crosses)
        
        # Display initial grid, shapes, and image
        self.play(ShowCreation(grid), ShowCreation(shapes), FadeIn(image))
        self.wait()
        
        # Section 1: Affine Transformations
        affine_title = Text("Affine Transformations", font="sans-serif")
        affine_title.scale(0.9)
        affine_title.next_to(title, DOWN)
        self.play(Write(affine_title))
        self.wait()
        
        # ===== Translation =====
        translation_label = Text("Translation", font="sans-serif")
        translation_label.scale(0.7)
        translation_label.to_edge(DOWN)
        self.play(Write(translation_label))
        
        translation_vector = np.array([1, 0.5, 0])
        self.play(
            grid.animate.shift(translation_vector),
            shapes.animate.shift(translation_vector),
            image.animate.shift(translation_vector),
            run_time=2
        )
        self.wait()
        
        # Reset position
        self.play(
            grid.animate.shift(-translation_vector),
            shapes.animate.shift(-translation_vector),
            image.animate.shift(-translation_vector),
            run_time=1
        )
        self.play(FadeOut(translation_label))
        
        # ===== Rotation =====
        rotation_label = Text("Rotation (Affine & Conformal)", font="sans-serif")
        rotation_label.scale(0.7)
        rotation_label.to_edge(DOWN)
        self.play(Write(rotation_label))
        
        rotation_angle = PI/6  # 30 degrees
        self.play(
            grid.animate.rotate(rotation_angle),
            shapes.animate.rotate(rotation_angle),
            image.animate.rotate(rotation_angle),
            run_time=2
        )
        self.wait()
        
        # Reset rotation
        self.play(
            grid.animate.rotate(-rotation_angle),
            shapes.animate.rotate(-rotation_angle),
            image.animate.rotate(-rotation_angle),
            run_time=1
        )
        self.play(FadeOut(rotation_label))
        
        # ===== Scaling =====
        scaling_label = Text("Scaling (Affine & Conformal)", font="sans-serif")
        scaling_label.scale(0.7)
        scaling_label.to_edge(DOWN)
        self.play(Write(scaling_label))
        
        scale_factor = 0.7
        self.play(
            grid.animate.scale(scale_factor),
            shapes.animate.scale(scale_factor),
            image.animate.scale(scale_factor),
            run_time=2
        )
        self.wait()
        
        # Reset scaling
        self.play(
            grid.animate.scale(1/scale_factor),
            shapes.animate.scale(1/scale_factor),
            image.animate.scale(1/scale_factor),
            run_time=1
        )
        self.play(FadeOut(scaling_label))
        
        # ===== Shearing =====
        shearing_label = Text("Shearing (Affine but NOT Conformal)", font="sans-serif")
        shearing_label.scale(0.7)
        shearing_label.to_edge(DOWN)
        self.play(Write(shearing_label))
        
        # Shearing matrix
        shear_matrix = np.array([
            [1, 0.5, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # Apply shearing transformation
        self.play(
            grid.animate.apply_matrix(shear_matrix),
            shapes.animate.apply_matrix(shear_matrix),
            image.animate.apply_matrix(shear_matrix),
            run_time=2
        )
        self.wait()
        
        # Add angle distortion note
        angle_note = Text("Notice angles are NOT preserved!", font="sans-serif", color=YELLOW)
        angle_note.scale(0.6)
        angle_note.next_to(shearing_label, UP)
        self.play(Write(angle_note))
        self.wait(2)
        
        # Reset shearing
        inv_shear_matrix = inv(shear_matrix)
        self.play(
            grid.animate.apply_matrix(inv_shear_matrix),
            shapes.animate.apply_matrix(inv_shear_matrix),
            image.animate.apply_matrix(inv_shear_matrix),
            run_time=1
        )
        self.play(FadeOut(shearing_label), FadeOut(angle_note))
        
        # Section 2: Conformal Transformations
        self.play(FadeOut(affine_title))
        conformal_title = Text("Conformal Transformations", font="sans-serif")
        conformal_title.scale(0.9)
        conformal_title.next_to(title, DOWN)
        self.play(Write(conformal_title))
        self.wait()
        
        # Basic conformal transformation (simplified)
        conformal_label = Text("Conformal Transformation Example", font="sans-serif")
        conformal_label.scale(0.7)
        conformal_label.to_edge(DOWN)
        self.play(Write(conformal_label))
        
        # Create a copy of objects to transform
        grid_copy = grid.copy()
        shapes_copy = shapes.copy()
        image_copy = image.copy()
        
        # Apply a simple conformal-like transformation (not mathematically precise but visually demonstrative)
        self.play(
            grid_copy.animate.scale(0.7).shift(RIGHT * 0.5),
            shapes_copy.animate.scale(0.7).shift(RIGHT * 0.5),
            image_copy.animate.scale(0.7).shift(RIGHT * 0.5),
            run_time=2
        )
        
        grid_copy.generate_target()
        shapes_copy.generate_target()
        image_copy.generate_target()
        
        # Distort in a way that still preserves angles locally
        grid_copy.target.apply_function(
            lambda p: np.array([
                p[0] + 0.1 * np.sin(p[1]), 
                p[1] + 0.1 * np.sin(p[0]), 
                p[2]
            ])
        )
        shapes_copy.target.apply_function(
            lambda p: np.array([
                p[0] + 0.1 * np.sin(p[1]), 
                p[1] + 0.1 * np.sin(p[0]), 
                p[2]
            ])
        )
        image_copy.target.apply_function(
            lambda p: np.array([
                p[0] + 0.1 * np.sin(p[1]), 
                p[1] + 0.1 * np.sin(p[0]), 
                p[2]
            ])
        )
        
        self.play(
            MoveToTarget(grid_copy),
            MoveToTarget(shapes_copy),
            MoveToTarget(image_copy),
            run_time=2
        )
        self.wait()
        
        # Display note about angle preservation
        angle_preservation = Text("Angles are preserved locally", font="sans-serif", color=GREEN)
        angle_preservation.scale(0.6)
        angle_preservation.next_to(conformal_label, UP)
        self.play(Write(angle_preservation))
        self.wait(2)
        
        # Clean up and conclusion
        self.play(
            FadeOut(grid), FadeOut(shapes), FadeOut(image),
            FadeOut(grid_copy), FadeOut(shapes_copy), FadeOut(image_copy),
            FadeOut(conformal_label), FadeOut(angle_preservation),
            FadeOut(conformal_title), FadeOut(title)
        )
        
        # Conclusion
        conclusion = Text("Summary:", font="sans-serif")
        conclusion.scale(0.9)
        conclusion.to_edge(UP, buff=1)
        points = [
            "• Affine transforms preserve straight lines and parallelism",
            "• Conformal transforms preserve angles locally",
            "• Scaling and rotation are both affine and conformal",
            "• Shearing is affine but not conformal"
        ]
        
        conclusion_points = VGroup(*[Text(point, font="sans-serif") for point in points])
        for point in conclusion_points:
            point.scale(0.6)
        conclusion_points.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        conclusion_points.next_to(conclusion, DOWN, buff=0.5)
        
        self.play(Write(conclusion))
        for point in conclusion_points:
            self.play(Write(point), run_time=0.8)
        
        self.wait(2)
        self.play(FadeOut(conclusion), FadeOut(conclusion_points))
        
        final_message = Text("Thank you for watching!", font="sans-serif")
        final_message.scale(1.2)
        self.play(Write(final_message))
        self.wait(2) 