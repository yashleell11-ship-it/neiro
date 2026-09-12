"""Training-side code: the dataset manifest and, later, the fine-tuning
recipes. Nothing in here is imported by the runtime daemon — training
happens on the 3090 Ti, once, and the runtime only ever loads the
resulting weights.
"""
