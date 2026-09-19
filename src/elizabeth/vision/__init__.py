"""Everything that comes in through the camera.

Empty of policy on purpose: `camera.py` produces frames and says what it
cost, and nothing in here decides what a frame MEANS. v2's gesture
recogniser and v3's vision work both sit on top of this, so the seam
between "getting pixels" and "interpreting pixels" exists before either
one is written.
"""
