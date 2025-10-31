from manim import *

class PhotonToMagnon(Scene):
    def construct(self):
        # Create the magnetic material rectangle
        material = Rectangle(width=4, height=2, color=BLUE, fill_opacity=0.5)
        material.move_to(RIGHT * 2)
        label = Text("MAGNETIC MATERIAL", font_size=30).move_to(material.get_center())

        # Photon: red arrow from left
        photon_arrow = Arrow(start=LEFT * 6, end=LEFT * 2.5, color=RED, stroke_width=8)
        photon_label = Text("Photon", font_size=24, color=RED).next_to(photon_arrow, UP)

        # Add objects to scene
        self.play(FadeIn(material), Write(label))
        self.play(GrowArrow(photon_arrow), Write(photon_label))

        # Move the photon arrow toward the material
        self.play(photon_arrow.animate.shift(RIGHT * 3.5), run_time=2)
        self.wait(0.5)

        # Create magnon ripple effect
        ripples = VGroup()
        for r in range(1, 6):
            ripple = Circle(radius=0.2 * r, color=YELLOW).move_to(material.get_center())
            ripple.set_stroke(width=2)
            ripples.add(ripple)

        # Animate ripples
        self.play(*[Create(r) for r in ripples], run_time=1) 
        self.play(*[r.animate.set_opacity(0).scale(1.5) for r in ripples], run_time=2)
        self.wait()

        # Fade everything out
        self.play(FadeOut(Group(material, label, photon_arrow, photon_label, ripples)))
