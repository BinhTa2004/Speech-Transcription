class CaptionMemory:
    def __init__(self):
        self.captions = []

    def add_caption(self, caption):
        self.captions.append(caption)

    def get_captions(self):
        return self.captions

    def clear(self):
        self.captions = []

memory = CaptionMemory()
