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
        
        # Conclusion
        self.play(FadeOut(conformal_title), FadeOut(conformal_description),
                 FadeOut(grid), FadeOut(shapes))
        
        conclusion = Text("Summary:", font="sans-serif")
        conclusion.scale(0.9)
        conclusion.to_edge(UP, buff=1)
        points = [
            "• Affine transforms preserve straight lines and parallelism",
            "• Conformal transforms preserve angles locally",
            "• Scaling and rotation are both affine and conformal",
            "• Shearing is affine but not conformal",
            "• Möbius transforms are conformal but not affine"
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
        
        # Create UpdateFromFunc for each movable element
        def update_grid_square(grid):
            grid.become(self.create_grid())
            grid.apply_function(complex_square)
            return grid
        
        def update_shapes_square(shapes):
            shapes.become(self.create_shapes())
            shapes.apply_function(complex_square)
            return shapes
        
        self.play(
            UpdateFromFunc(grid, update_grid_square),
            UpdateFromFunc(shapes, update_shapes_square),
            run_time=3
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
            FadeOut(grid),
            FadeOut(shapes)
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
        
        def update_grid_mobius(grid):
            grid.become(self.create_grid())
            grid.apply_function(mobius_transform)
            return grid
        
        def update_shapes_mobius(shapes):
            shapes.become(self.create_shapes())
            shapes.apply_function(mobius_transform)
            return shapes
        
        self.play(
            UpdateFromFunc(grid, update_grid_mobius),
            UpdateFromFunc(shapes, update_shapes_mobius),
            run_time=3
        )
        
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
                 FadeOut(grid), FadeOut(shapes))


# Example class demonstrating different angles in each transformation
class AnglePreservationDemo(Scene):
    def construct(self):
        # Title
        title = Text("Angle Preservation in Transformations", font="sans-serif")
        title.scale(1.5)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
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
        self.wait(2)
        self.play(FadeOut(initial_label))
        
        # 1. Show affine non-conformal: Shearing
        shear_label = Text("Shearing (Affine, NOT Conformal)", font="sans-serif")
        shear_label.scale(0.8)
        shear_label.to_edge(DOWN)
        self.play(Write(shear_label))
        
        # Shearing matrix
        shear_matrix = np.array([
            [1, 0.7, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # Apply shearing and highlight angle changes
        self.play(
            grid.animate.apply_matrix(shear_matrix),
            crosses.animate.apply_matrix(shear_matrix),
            run_time=2
        )
        
        self.highlight_angles(crosses)
        self.wait()
        
        angle_note = Text("Angles are NOT preserved!", font="sans-serif", color=YELLOW)
        angle_note.scale(0.7)
        angle_note.next_to(shear_label, UP)
        self.play(Write(angle_note))
        self.wait(2)
        
        self.play(FadeOut(shear_label), FadeOut(angle_note))
        
        # Reset by applying inverse matrix
        inv_shear_matrix = inv(shear_matrix)
        self.play(
            grid.animate.apply_matrix(inv_shear_matrix),
            crosses.animate.apply_matrix(inv_shear_matrix),
            run_time=1
        )
        
        # 2. Show conformal: Complex function
        conformal_label = Text("Complex Function (Conformal)", font="sans-serif")
        conformal_label.scale(0.8)
        conformal_label.to_edge(DOWN)
        self.play(Write(conformal_label))
        
        # Apply a conformal transformation (simulated)
        def complex_exp(point):
            x, y, z = point
            # Scale inputs to avoid large outputs
            x, y = x*0.3, y*0.3
            # exp(z) = e^x * (cos(y) + i*sin(y))
            magnitude = np.exp(x)
            x_new = magnitude * np.cos(y)
            y_new = magnitude * np.sin(y)
            # Scale to keep in reasonable view
            scale = 0.8
            return np.array([x_new * scale, y_new * scale, z])
        
        def update_grid_exp(grid):
            grid.become(NumberPlane(x_range=(-5, 5), y_range=(-3, 3)))
            grid.apply_function(complex_exp)
            return grid
        
        def update_crosses_exp(crosses):
            crosses.become(self.create_crosses())
            crosses.apply_function(complex_exp)
            return crosses
        
        self.play(
            UpdateFromFunc(grid, update_grid_exp),
            UpdateFromFunc(crosses, update_crosses_exp),
            run_time=3
        )
        
        self.highlight_angles(crosses)
        
        angle_note = Text("Angles are preserved locally!", font="sans-serif", color=GREEN)
        angle_note.scale(0.7)
        angle_note.next_to(conformal_label, UP)
        self.play(Write(angle_note))
        self.wait(2)
        
        self.play(FadeOut(conformal_label), FadeOut(angle_note))
        self.play(FadeOut(grid), FadeOut(crosses))
        
        # Conclusion
        conclusion = Text("Key Insight:", font="sans-serif")
        conclusion.scale(0.9)
        conclusion.to_edge(UP, buff=1)
        explanation = Text(
            "In conformal maps, the local shape is preserved\n"
            "even though the global structure may change dramatically.",
            font="sans-serif"
        ).scale(0.7).next_to(conclusion, DOWN, buff=0.5)
        
        self.play(Write(conclusion))
        self.play(Write(explanation))
        
        self.wait(3)
        self.play(FadeOut(conclusion), FadeOut(explanation), FadeOut(title))
    
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
        # Highlight the angles in the crosses
        highlights = VGroup()
        
        for cross in crosses:
            # Create an arc to show the angle
            arc = Arc(
                radius=0.3,
                angle=cross[1].get_angle(),
                color=RED,
            ).shift(cross.get_center())
            highlights.add(arc)
        
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