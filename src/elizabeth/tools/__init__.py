"""Elizabeth's control surface over the machine.

The rule everything here rests on: **the model never emits a string that
reaches a command line.** It emits an integer index into a list Elizabeth
produced in the same turn, or a `Literal` enum member. See egress.py for
why that is a hard requirement on this machine rather than caution.
"""
