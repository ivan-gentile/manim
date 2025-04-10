from manimlib import *
import numpy as np
from numpy.linalg import inv

class AffineVsConformal(Scene):
    def construct(self):
        # Title
        title = Text("Affine vs Conformal Transformations in 2D", font="sans-serif")
        title.scale(1.5)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Create a grid and some shapes for demonstration
        grid = self.create_grid()
        shapes = self.create_shapes()
        
        # Display initial grid and shapes
        self.play(ShowCreation(grid), ShowCreation(shapes))
        self.wait()
        
        # Add labels for initial state
        initial_label = Text("Initial Configuration", font="sans-serif")
        initial_label.scale(0.8)
        initial_label.to_edge(DOWN)
        self.play(Write(initial_label))
        self.wait(2)
        self.play(FadeOut(initial_label))
        
        # Section 1: Affine Transformations
        affine_title = Text("Affine Transformations", font="sans-serif")
        affine_title.scale(1.2)
        affine_title.to_edge(UP, buff=1.5)
        affine_description = Text("Preserve: Straight lines, Parallel lines, and Distance ratios", 
                                 font="sans-serif")
        affine_description.scale(0.6)
        affine_description.next_to(affine_title, DOWN)
        self.play(Write(affine_title), Write(affine_description))
        self.wait()
        
        # Demonstrate different affine transformations
        self.demonstrate_affine_transformations(grid, shapes)
        
        # Remove affine content
        self.play(FadeOut(affine_title), FadeOut(affine_description),
                 FadeOut(grid), FadeOut(shapes))
        self.wait()
        
        # Reset grid and shapes
        grid = self.create_grid()
        shapes = self.create_shapes()
        self.play(ShowCreation(grid), ShowCreation(shapes))
        self.wait()
        
        # Section 2: Conformal Transformations
        conformal_title = Text("Conformal Transformations", font="sans-serif")
        conformal_title.scale(1.2)
        conformal_title.to_edge(UP, buff=1.5)
        conformal_description = Text("Preserve: Angles between curves", 
                                    font="sans-serif")
        conformal_description.scale(0.6)
        conformal_description.next_to(conformal_title, DOWN)
        self.play(Write(conformal_title), Write(conformal_description))
        self.wait()
        
        # Demonstrate different conformal transformations
        self.demonstrate_conformal_transformations(grid, shapes)
        
        # Section 3: Scale Invariance Comparison
        self.compare_scale_invariance()
        
        # Conclusion
        self.play(FadeOut(conformal_title), FadeOut(conformal_description))
        
        conclusion = Text("Summary:", font="sans-serif")
        conclusion.scale(0.9)
        conclusion.to_edge(UP, buff=1)
        points = [
            "• Affine transforms preserve straight lines and parallelism",
            "• Conformal transforms preserve angles locally",
            "• Scaling and rotation are both affine and conformal",
            "• Shearing is affine but not conformal",
            "• Möbius transforms are conformal but not affine",
            "• Conformal maps are scale invariant, affine maps are not"
        ]
        
        conclusion_points = VGroup(*[Text(point, font="sans-serif") for point in points])
        for point in conclusion_points:
            point.scale(0.6)
        conclusion_points.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        conclusion_points.next_to(conclusion, DOWN, buff=0.5)
        
        self.play(Write(conclusion))
        for point in conclusion_points:
            self.play(Write(point), run_time=0.8)
        
        self.wait(3)
        self.play(FadeOut(conclusion), FadeOut(conclusion_points), FadeOut(title))
        
        final_message = Text("Thank you for watching!", font="sans-serif")
        final_message.scale(1.2)
        self.play(Write(final_message))
        self.wait(2)
    
    def create_grid(self):
        # Create a 2D grid
        grid = NumberPlane(
            x_range=(-5, 5, 1),
            y_range=(-3, 3, 1),
            background_line_style={
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        )
        return grid
    
    def create_shapes(self):
        # Create various shapes to demonstrate the transformations
        square = Square(side_length=1, color=BLUE).shift(LEFT * 2)
        circle = Circle(radius=0.5, color=RED).shift(RIGHT * 2)
        triangle = Triangle(color=GREEN).scale(0.7).shift(UP * 1.5)
        
        # Create a cross to show angle preservation
        horizontal = Line(LEFT * 0.5, RIGHT * 0.5, color=YELLOW)
        vertical = Line(DOWN * 0.5, UP * 0.5, color=YELLOW)
        cross = VGroup(horizontal, vertical)
        crosses = VGroup(
            cross.copy().shift(LEFT * 2 + UP * 1),
            cross.copy().shift(RIGHT * 2 + DOWN * 1),
            cross.copy().shift(LEFT * 0.5 + DOWN * 1.5),
            cross.copy().shift(RIGHT * 0.5 + UP * 1.5)
        )
        
        shapes = VGroup(square, circle, triangle, crosses)
        return shapes
    
    def demonstrate_affine_transformations(self, grid, shapes):
        # 1. Translation
        translation_label = Text("Translation", font="sans-serif")
        translation_label.scale(0.8)
        translation_label.to_edge(DOWN)
        self.play(Write(translation_label))
        
        translation_vector = np.array([1, 0.5, 0])
        self.play(
            grid.animate.shift(translation_vector),
            shapes.animate.shift(translation_vector),
            run_time=2
        )
        self.wait()
        
        self.play(FadeOut(translation_label))
        self.play(
            grid.animate.shift(-translation_vector),
            shapes.animate.shift(-translation_vector),
            run_time=1
        )
        
        # 2. Rotation
        rotation_label = Text("Rotation (Affine & Conformal)", font="sans-serif")
        rotation_label.scale(0.8)
        rotation_label.to_edge(DOWN)
        self.play(Write(rotation_label))
        
        rotation_angle = PI/4  # 45 degrees
        self.play(
            grid.animate.rotate(rotation_angle),
            shapes.animate.rotate(rotation_angle),
            run_time=2
        )
        self.wait()
        
        self.play(FadeOut(rotation_label))
        self.play(
            grid.animate.rotate(-rotation_angle),
            shapes.animate.rotate(-rotation_angle),
            run_time=1
        )
        
        # 3. Scaling
        scaling_label = Text("Scaling (Affine & Conformal)", font="sans-serif")
        scaling_label.scale(0.8)
        scaling_label.to_edge(DOWN)
        self.play(Write(scaling_label))
        
        scale_factor = 0.7
        self.play(
            grid.animate.scale(scale_factor),
            shapes.animate.scale(scale_factor),
            run_time=2
        )
        self.wait()
        
        self.play(FadeOut(scaling_label))
        self.play(
            grid.animate.scale(1/scale_factor),
            shapes.animate.scale(1/scale_factor),
            run_time=1
        )
        
        # 4. Shearing (Affine but not Conformal)
        shearing_label = Text("Shearing (Affine but NOT Conformal)", font="sans-serif")
        shearing_label.scale(0.8)
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
            run_time=2
        )
        self.wait()
        
        # Add an annotation about angle distortion
        angle_note = Text("Notice that angles are NOT preserved!", 
                         font="sans-serif", color=YELLOW)
        angle_note.scale(0.7)
        angle_note.next_to(shearing_label, UP)
        self.play(Write(angle_note))
        self.wait(2)
        
        self.play(FadeOut(shearing_label), FadeOut(angle_note))
        
        # Reset by applying inverse matrix
        inv_shear_matrix = inv(shear_matrix)
        self.play(
            grid.animate.apply_matrix(inv_shear_matrix),
            shapes.animate.apply_matrix(inv_shear_matrix),
            run_time=1
        )
        
        # 5. General linear transformation (Affine)
        linear_label = Text("General Linear Transformation (Affine)", font="sans-serif")
        linear_label.scale(0.8)
        linear_label.to_edge(DOWN)
        self.play(Write(linear_label))
        
        # General linear transformation matrix
        linear_matrix = np.array([
            [0.8, 0.3, 0],
            [-0.2, 1.1, 0],
            [0, 0, 1]
        ])
        
        self.play(
            grid.animate.apply_matrix(linear_matrix),
            shapes.animate.apply_matrix(linear_matrix),
            run_time=2
        )
        self.wait()
        
        parallel_note = Text("Parallel lines remain parallel", 
                           font="sans-serif", color=GREEN)
        parallel_note.scale(0.7)
        parallel_note.next_to(linear_label, UP)
        self.play(Write(parallel_note))
        self.wait(2)
        
        self.play(FadeOut(linear_label), FadeOut(parallel_note))
        
        # Reset to original configuration
        inv_linear_matrix = inv(linear_matrix)
        self.play(
            grid.animate.apply_matrix(inv_linear_matrix),
            shapes.animate.apply_matrix(inv_linear_matrix),
            run_time=1
        )
    
    def demonstrate_conformal_transformations(self, grid, shapes):
        # 1. First demonstrate transformations that are both affine and conformal
        both_label = Text("Transformations that are both Affine and Conformal:", 
                         font="sans-serif")
        both_label.scale(0.8)
        both_label.to_edge(DOWN)
        self.play(Write(both_label))
        self.wait()
        self.play(FadeOut(both_label))
        
        # 1.1 Rotation (already shown in affine section, but briefly mention)
        rotation_label = Text("Rotation", font="sans-serif")
        rotation_label.scale(0.8)
        rotation_label.to_edge(DOWN)
        self.play(Write(rotation_label))
        
        rotation_angle = PI/6  # 30 degrees
        self.play(
            grid.animate.rotate(rotation_angle),
            shapes.animate.rotate(rotation_angle),
            run_time=1.5
        )
        self.wait()
        
        self.play(FadeOut(rotation_label))
        self.play(
            grid.animate.rotate(-rotation_angle),
            shapes.animate.rotate(-rotation_angle),
            run_time=1
        )
        
        # 1.2 Uniform scaling
        scaling_label = Text("Uniform Scaling", font="sans-serif")
        scaling_label.scale(0.8)
        scaling_label.to_edge(DOWN)
        self.play(Write(scaling_label))
        
        scale_factor = 0.8
        self.play(
            grid.animate.scale(scale_factor),
            shapes.animate.scale(scale_factor),
            run_time=1.5
        )
        self.wait()
        
        self.play(FadeOut(scaling_label))
        self.play(
            grid.animate.scale(1/scale_factor),
            shapes.animate.scale(1/scale_factor),
            run_time=1
        )
        
        # 2. Complex function transformations (Conformal but not Affine)
        # We'll simulate Möbius transformations and other complex mappings
        
        complex_label = Text("Complex Function Transformations (Conformal)", 
                            font="sans-serif")
        complex_label.scale(0.8)
        complex_label.to_edge(DOWN)
        self.play(Write(complex_label))
        self.wait()
        
        # 2.1 Simulate a complex squaring function f(z) = z^2
        z_squared_label = Text("f(z) = z² (Complex Squaring)", 
                              font="sans-serif")
        z_squared_label.scale(0.7)
        z_squared_label.next_to(complex_label, UP)
        self.play(Write(z_squared_label))
        
        # Custom function to transform a point as if it were z^2 in the complex plane
        def complex_square(point):
            x, y, z = point
            # Treat (x,y) as a complex number and square it
            x_new = x*x - y*y
            y_new = 2*x*y
            # Scale down to keep points in reasonable range
            scale = 0.5
            return np.array([x_new * scale, y_new * scale, z])
        
        # Create a smooth transition function for the complex square transformation
        def smooth_complex_square(t):
            def transform(point):
                x, y, z = point
                # Interpolate between identity and complex square
                x_new = (1-t)*x + t*(x*x - y*y)
                y_new = (1-t)*y + t*(2*x*y)
                # Apply scaling factor for better visibility
                scale = 0.5 + 0.5*(1-t)
                return np.array([x_new * scale, y_new * scale, z])
            return transform
        
        # Animate the transition smoothly
        grid_copy = self.create_grid()
        shapes_copy = self.create_shapes()
        
        self.play(
            FadeOut(grid),
            FadeOut(shapes),
            ShowCreation(grid_copy),
            ShowCreation(shapes_copy)
        )
        
        n_steps = 30
        for i in range(n_steps + 1):
            t = i / n_steps
            new_grid = self.create_grid()
            new_shapes = self.create_shapes()
            new_grid.apply_function(smooth_complex_square(t))
            new_shapes.apply_function(smooth_complex_square(t))
            
            if i == 0:
                self.add(new_grid, new_shapes)
            else:
                self.play(
                    Transform(grid_copy, new_grid),
                    Transform(shapes_copy, new_shapes),
                    run_time=0.1,
                    rate_func=linear
                )
        
        self.wait()
        
        angle_note = Text("Notice that angles are preserved locally", 
                         font="sans-serif", color=YELLOW)
        angle_note.scale(0.7)
        angle_note.next_to(z_squared_label, UP)
        self.play(Write(angle_note))
        self.wait(2)
        
        self.play(FadeOut(z_squared_label), FadeOut(angle_note))
        
        # Reset the grid and shapes
        self.play(
            FadeOut(grid_copy),
            FadeOut(shapes_copy)
        )
        grid = self.create_grid()
        shapes = self.create_shapes()
        self.play(ShowCreation(grid), ShowCreation(shapes))
        
        # 2.2 Simulate a Möbius transformation
        mobius_label = Text("Möbius Transformation", font="sans-serif")
        mobius_label.scale(0.7)
        mobius_label.next_to(complex_label, UP)
        self.play(Write(mobius_label))
        
        # Simulate a Möbius transformation (az+b)/(cz+d)
        # Using a simple example: (z+0.5)/(z-1)
        def mobius_transform(point):
            x, y, z = point
            # Treat (x,y) as a complex number
            # Applying (z+0.5)/(z-1)
            denom = (x - 1)**2 + y**2
            if denom < 0.1:  # Avoid division by near-zero
                return point
            
            x_new = ((x + 0.5) * (x - 1) + y * y) / denom
            y_new = (y * (x - 1) - (x + 0.5) * y) / denom
            
            # Scale to keep points in view
            scale = 1.5
            if abs(x_new) > 5 or abs(y_new) > 5:
                return point
            
            return np.array([x_new, y_new, z])
        
        # Create a smooth transition function for Möbius transformation
        def smooth_mobius_transform(t):
            def transform(point):
                x, y, z = point
                if t == 0:
                    return point
                
                # Calculating the Möbius transform (z+0.5)/(z-1)
                denom = (x - 1)**2 + y**2
                if denom < 0.1:  # Avoid division by near-zero
                    return point
                
                x_new = ((x + 0.5) * (x - 1) + y * y) / denom
                y_new = (y * (x - 1) - (x + 0.5) * y) / denom
                
                # Scale to keep points in view
                if abs(x_new) > 5 or abs(y_new) > 5:
                    return point
                
                # Interpolate between identity and Möbius transform
                return np.array([
                    (1-t)*x + t*x_new,
                    (1-t)*y + t*y_new,
                    z
                ])
            return transform
        
        # Animate the Möbius transformation smoothly
        grid_copy = self.create_grid()
        shapes_copy = self.create_shapes()
        
        self.play(
            FadeOut(grid),
            FadeOut(shapes),
            ShowCreation(grid_copy),
            ShowCreation(shapes_copy)
        )
        
        n_steps = 30
        for i in range(n_steps + 1):
            t = i / n_steps
            new_grid = self.create_grid()
            new_shapes = self.create_shapes()
            new_grid.apply_function(smooth_mobius_transform(t))
            new_shapes.apply_function(smooth_mobius_transform(t))
            
            if i == 0:
                self.add(new_grid, new_shapes)
            else:
                self.play(
                    Transform(grid_copy, new_grid),
                    Transform(shapes_copy, new_shapes),
                    run_time=0.1,
                    rate_func=linear
                )
        
        self.wait()
        
        conformal_note = Text("Conformal but NOT affine: straight lines become curves", 
                             font="sans-serif", color=RED)
        conformal_note.scale(0.7)
        conformal_note.next_to(mobius_label, UP)
        self.play(Write(conformal_note))
        self.wait(2)
        
        self.play(FadeOut(mobius_label), FadeOut(complex_label), FadeOut(conformal_note))
        
        # Comparison Section
        comparison_title = Text("Direct Comparison", font="sans-serif")
        comparison_title.scale(0.9)
        comparison_title.to_edge(UP, buff=1.5)
        self.play(Write(comparison_title))
        
        comparison_notes = VGroup(
            Text("Affine: Preserves straight lines and parallelism", font="sans-serif", color=BLUE),
            Text("Conformal: Preserves angles between curves", font="sans-serif", color=RED)
        ).scale(0.7).arrange(DOWN, aligned_edge=LEFT, buff=0.5).to_edge(DOWN, buff=1)
        
        self.play(Write(comparison_notes))
        self.wait(3)
        
        self.play(FadeOut(comparison_title), FadeOut(comparison_notes), 
                 FadeOut(grid_copy), FadeOut(shapes_copy))
    
    def compare_scale_invariance(self):
        # Title for the section
        section_title = Text("Scale Invariance Comparison", font="sans-serif")
        section_title.scale(1.2)
        section_title.to_edge(UP, buff=1.5)
        
        section_desc = Text("Affine transformations are NOT scale invariant, but conformal ones ARE", 
                           font="sans-serif")
        section_desc.scale(0.6)
        section_desc.next_to(section_title, DOWN)
        
        self.play(Write(section_title), Write(section_desc))
        self.wait()
        
        # Create two grids - one large and one small
        large_grid = self.create_grid()
        large_grid.shift(LEFT * 3)
        
        small_grid = self.create_grid()
        small_grid.scale(0.5)
        small_grid.shift(RIGHT * 3)
        
        # Labels for each grid
        large_label = Text("Original Scale", font="sans-serif").scale(0.6)
        large_label.next_to(large_grid, UP)
        
        small_label = Text("Reduced Scale (0.5x)", font="sans-serif").scale(0.6)
        small_label.next_to(small_grid, UP)
        
        self.play(
            ShowCreation(large_grid),
            ShowCreation(small_grid),
            Write(large_label),
            Write(small_label)
        )
        self.wait()
        
        # 1. First demonstrate how affine transforms are NOT scale invariant
        affine_title = Text("Affine Transformation: Shearing", font="sans-serif")
        affine_title.scale(0.8)
        affine_title.to_edge(DOWN)
        self.play(Write(affine_title))
        
        # Define shearing transformation
        shear_matrix = np.array([
            [1, 0.7, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # Apply to both grids
        self.play(
            large_grid.animate.apply_matrix(shear_matrix),
            small_grid.animate.apply_matrix(shear_matrix),
            run_time=2
        )
        
        # Create indicators to highlight the differences
        large_highlight = SurroundingRectangle(large_grid, color=YELLOW, buff=0.2)
        small_highlight = SurroundingRectangle(small_grid, color=YELLOW, buff=0.2)
        
        highlight_label = Text("Notice the difference in angles after scaling", 
                              font="sans-serif", color=YELLOW).scale(0.7)
        highlight_label.next_to(affine_title, UP)
        
        self.play(
            ShowCreation(large_highlight),
            ShowCreation(small_highlight),
            Write(highlight_label)
        )
        self.wait(2)
        
        # Explanation of what happened
        explanation = Text(
            "Shearing by 0.7 at different scales produces\ndifferent angles at smaller scales", 
            font="sans-serif"
        ).scale(0.6)
        explanation.next_to(highlight_label, UP)
        
        self.play(Write(explanation))
        self.wait(2)
        
        # Clean up and reset
        self.play(
            FadeOut(large_highlight),
            FadeOut(small_highlight),
            FadeOut(highlight_label),
            FadeOut(explanation),
            FadeOut(affine_title)
        )
        
        # Reset grids
        self.play(FadeOut(large_grid), FadeOut(small_grid))
        
        large_grid = self.create_grid()
        large_grid.shift(LEFT * 3)
        
        small_grid = self.create_grid()
        small_grid.scale(0.5)
        small_grid.shift(RIGHT * 3)
        
        self.play(
            ShowCreation(large_grid),
            ShowCreation(small_grid)
        )
        
        # 2. Now demonstrate how conformal transforms ARE scale invariant
        conformal_title = Text("Conformal Transformation: Möbius", font="sans-serif")
        conformal_title.scale(0.8)
        conformal_title.to_edge(DOWN)
        self.play(Write(conformal_title))
        
        # Define a conformal transformation (Möbius)
        def mobius_simple(point):
            x, y, z = point
            # A simpler Möbius transform: 1/(z-0.5)
            denom = (x - 0.5)**2 + y**2
            if denom < 0.1:
                return point
                
            x_new = (x - 0.5) / denom
            y_new = -y / denom
            
            # Scale to keep in view
            if abs(x_new) > 5 or abs(y_new) > 5:
                return point
                
            return np.array([x_new, y_new, z])
        
        # Apply to both grids using a smooth transition
        n_steps = 20
        for i in range(n_steps + 1):
            t = i / n_steps
            
            def transition_func(point):
                x, y, z = point
                mob_point = mobius_simple(point)
                return np.array([
                    (1-t)*x + t*mob_point[0],
                    (1-t)*y + t*mob_point[1],
                    z
                ])
            
            new_large_grid = self.create_grid().shift(LEFT * 3)
            new_small_grid = self.create_grid().scale(0.5).shift(RIGHT * 3)
            
            new_large_grid.apply_function(transition_func)
            new_small_grid.apply_function(transition_func)
            
            if i == 0:
                large_copy = new_large_grid.copy()
                small_copy = new_small_grid.copy()
                self.add(large_copy, small_copy)
            else:
                self.play(
                    Transform(large_copy, new_large_grid),
                    Transform(small_copy, new_small_grid),
                    run_time=0.1,
                    rate_func=linear
                )
        
        # Create indicators to highlight the similarities
        large_highlight = SurroundingRectangle(large_copy, color=GREEN, buff=0.2)
        small_highlight = SurroundingRectangle(small_copy, color=GREEN, buff=0.2)
        
        highlight_label = Text("Notice how angles are preserved at both scales", 
                              font="sans-serif", color=GREEN).scale(0.7)
        highlight_label.next_to(conformal_title, UP)
        
        self.play(
            ShowCreation(large_highlight),
            ShowCreation(small_highlight),
            Write(highlight_label)
        )
        self.wait(2)
        
        # Explanation of what happened
        explanation = Text(
            "Local angles are preserved regardless of scale -\nthis is a key property of conformal maps", 
            font="sans-serif"
        ).scale(0.6)
        explanation.next_to(highlight_label, UP)
        
        self.play(Write(explanation))
        self.wait(2)
        
        # Clean up
        self.play(
            FadeOut(large_label),
            FadeOut(small_label),
            FadeOut(large_highlight),
            FadeOut(small_highlight),
            FadeOut(highlight_label),
            FadeOut(explanation),
            FadeOut(conformal_title),
            FadeOut(section_title),
            FadeOut(section_desc),
            FadeOut(large_copy),
            FadeOut(small_copy)
        )


# Example class demonstrating different angles in each transformation
class AnglePreservationDemo(Scene):
    def construct(self):
        # Title
        title = Text("Angle Preservation in Transformations", font="sans-serif")
        title.scale(1.5)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Introduction
        intro_text = Text(
            "In this demo, we'll visualize how different transformations\naffect angles between intersecting curves", 
            font="sans-serif"
        ).scale(0.7)
        intro_text.next_to(title, DOWN)
        self.play(Write(intro_text))
        self.wait(2)
        self.play(FadeOut(intro_text))
        
        # Create a grid and crossing lines to show angles
        grid = NumberPlane(x_range=(-5, 5), y_range=(-3, 3))
        
        # Create multiple crosses at different locations
        crosses = self.create_crosses()
        
        # Display initial grid and crosses
        self.play(ShowCreation(grid), ShowCreation(crosses))
        self.wait()
        
        # Add labels for initial state
        initial_label = Text("Initial Configuration", font="sans-serif")
        initial_label.scale(0.8)
        initial_label.to_edge(DOWN)
        self.play(Write(initial_label))
        
        # Add explanation about crosses
        cross_explanation = Text(
            "Each cross shows the angle between two lines.\nWe'll observe how these angles change after transformations.", 
            font="sans-serif"
        ).scale(0.6)
        cross_explanation.next_to(initial_label, UP)
        self.play(Write(cross_explanation))
        self.wait(2)
        self.play(FadeOut(initial_label), FadeOut(cross_explanation))
        
        # 1. Show affine non-conformal: Shearing
        shear_label = Text("Shearing (Affine, NOT Conformal)", font="sans-serif")
        shear_label.scale(0.8)
        shear_label.to_edge(DOWN)
        self.play(Write(shear_label))
        
        # Add mathematical description
        shear_math = Text("Transformation matrix: [[1, 0.7], [0, 1]]", font="sans-serif")
        shear_math.scale(0.6)
        shear_math.next_to(shear_label, UP)
        self.play(Write(shear_math))
        self.wait()
        
        # Shearing matrix
        shear_matrix = np.array([
            [1, 0.7, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # Apply shearing with smooth transition
        n_steps = 20
        for i in range(n_steps + 1):
            t = i / n_steps
            partial_matrix = np.array([
                [1, 0.7 * t, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            
            if i == 0:
                grid_copy = grid.copy()
                crosses_copy = crosses.copy()
                self.add(grid_copy, crosses_copy)
            else:
                new_grid = NumberPlane(x_range=(-5, 5), y_range=(-3, 3))
                new_crosses = self.create_crosses()
                
                new_grid.apply_matrix(partial_matrix)
                new_crosses.apply_matrix(partial_matrix)
                
                self.play(
                    Transform(grid_copy, new_grid),
                    Transform(crosses_copy, new_crosses),
                    run_time=0.1,
                    rate_func=linear
                )
        
        self.highlight_angles(crosses_copy)
        self.wait()
        
        angle_note = Text("Angles are NOT preserved!", font="sans-serif", color=YELLOW)
        angle_note.scale(0.7)
        angle_note.next_to(shear_math, UP)
        self.play(Write(angle_note))
        
        # Additional explanation about what's happening
        shear_explanation = Text(
            "Shearing distorts angles by different amounts\ndepending on their original orientation", 
            font="sans-serif"
        ).scale(0.6)
        shear_explanation.next_to(angle_note, UP)
        self.play(Write(shear_explanation))
        self.wait(2)
        
        self.play(
            FadeOut(shear_label), 
            FadeOut(angle_note), 
            FadeOut(shear_math),
            FadeOut(shear_explanation)
        )
        
        # Reset by applying inverse matrix
        inv_shear_matrix = inv(shear_matrix)
        self.play(
            grid_copy.animate.apply_matrix(inv_shear_matrix),
            crosses_copy.animate.apply_matrix(inv_shear_matrix),
            run_time=1
        )
        
        # 2. Show conformal: Complex function
        conformal_label = Text("Complex Function (Conformal)", font="sans-serif")
        conformal_label.scale(0.8)
        conformal_label.to_edge(DOWN)
        self.play(Write(conformal_label))
        
        # Add mathematical description
        conf_math = Text("Complex function: f(z) = e^z", font="sans-serif")
        conf_math.scale(0.6)
        conf_math.next_to(conformal_label, UP)
        self.play(Write(conf_math))
        
        # Apply a conformal transformation (simulated) with smooth transition
        def complex_exp(point, t):
            x, y, z = point
            # Scale inputs to avoid large outputs
            x, y = x*0.3, y*0.3
            # exp(z) = e^x * (cos(y) + i*sin(y))
            magnitude = np.exp(x)
            x_new = magnitude * np.cos(y)
            y_new = magnitude * np.sin(y)
            
            # Interpolate between original and transformed
            x_result = (1-t)*x + t*x_new*0.8
            y_result = (1-t)*y + t*y_new*0.8
            
            return np.array([x_result, y_result, z])
        
        n_steps = 30
        for i in range(n_steps + 1):
            t = i / n_steps
            
            if i == 0:
                new_grid = NumberPlane(x_range=(-5, 5), y_range=(-3, 3))
                new_crosses = self.create_crosses()
                
                transformed_grid = new_grid.copy()
                transformed_crosses = new_crosses.copy()
                
                self.play(
                    Transform(grid_copy, transformed_grid),
                    Transform(crosses_copy, transformed_crosses)
                )
            else:
                new_grid = NumberPlane(x_range=(-5, 5), y_range=(-3, 3))
                new_crosses = self.create_crosses()
                
                # Apply transformation with partial effect
                new_grid.apply_function(lambda p: complex_exp(p, t))
                new_crosses.apply_function(lambda p: complex_exp(p, t))
                
                self.play(
                    Transform(grid_copy, new_grid),
                    Transform(crosses_copy, new_crosses),
                    run_time=0.1,
                    rate_func=linear
                )
        
        self.highlight_angles(crosses_copy)
        
        angle_note = Text("Angles are preserved locally!", font="sans-serif", color=GREEN)
        angle_note.scale(0.7)
        angle_note.next_to(conf_math, UP)
        self.play(Write(angle_note))
        
        # Additional explanation
        conf_explanation = Text(
            "Complex analytic functions preserve angles locally,\neven though they distort the global shape", 
            font="sans-serif"
        ).scale(0.6)
        conf_explanation.next_to(angle_note, UP)
        self.play(Write(conf_explanation))
        self.wait(2)
        
        self.play(
            FadeOut(conformal_label), 
            FadeOut(angle_note),
            FadeOut(conf_math),
            FadeOut(conf_explanation),
            FadeOut(grid_copy), 
            FadeOut(crosses_copy)
        )
        
        # Conclusion with mathematical insight
        conclusion = Text("Key Insight:", font="sans-serif")
        conclusion.scale(0.9)
        conclusion.to_edge(UP, buff=1)
        
        math_insight = Text(
            "Conformal maps preserve angles because their Jacobian\nmatrix at any point is a scalar multiple of a rotation matrix", 
            font="sans-serif"
        ).scale(0.6)
        math_insight.next_to(conclusion, DOWN, buff=0.3)
        
        explanation = Text(
            "In conformal maps, the local shape is preserved\n"
            "even though the global structure may change dramatically.",
            font="sans-serif"
        ).scale(0.7).next_to(math_insight, DOWN, buff=0.5)
        
        self.play(Write(conclusion))
        self.play(Write(math_insight))
        self.play(Write(explanation))
        
        self.wait(3)
        self.play(FadeOut(conclusion), FadeOut(explanation), FadeOut(math_insight), FadeOut(title))
    
    def create_crosses(self):
        # Create crosses at different positions to show angle preservation
        crosses = VGroup()
        positions = [
            LEFT * 3 + UP * 1.5,
            RIGHT * 3 + UP * 1.5,
            LEFT * 3 + DOWN * 1.5,
            RIGHT * 3 + DOWN * 1.5,
            LEFT * 1.5,
            RIGHT * 1.5,
            UP * 1.5,
            DOWN * 1.5
        ]
        
        angles = [PI/6, PI/4, PI/3, PI/2, PI/8, 3*PI/8, 2*PI/5, PI/5]
        
        for pos, angle in zip(positions, angles):
            horiz = Line(LEFT * 0.5, RIGHT * 0.5, color=YELLOW)
            vert = Line(DOWN * 0.5, UP * 0.5, color=YELLOW).rotate(angle)
            cross = VGroup(horiz, vert).shift(pos)
            crosses.add(cross)
        
        return crosses
    
    def highlight_angles(self, crosses):
        # Highlight the angles in the crosses more dramatically
        highlights = VGroup()
        
        for cross in crosses:
            # Create an arc to show the angle
            angle = cross[1].get_angle()
            arc = Arc(
                radius=0.3,
                angle=angle,
                color=RED,
            ).shift(cross.get_center())
            
            # Add a small dot at the center
            dot = Dot(cross.get_center(), color=WHITE, radius=0.05)
            
            # Add the angle value as text
            angle_value = Text(f"{int(angle * 180 / PI)}°", font="sans-serif")
            angle_value.scale(0.3)
            angle_value.next_to(arc, UR, buff=0.1)
            
            group = VGroup(arc, dot, angle_value)
            highlights.add(group)
        
        self.play(ShowCreation(highlights))
        self.wait()
        self.play(FadeOut(highlights))


# Run the animations
if __name__ == "__main__":
    scenes = [
        AffineVsConformal(),
        AnglePreservationDemo()
    ]
    
    for scene in scenes:
        scene.render()