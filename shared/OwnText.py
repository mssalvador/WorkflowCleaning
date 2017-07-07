from ipywidgets import widgets

class OwnText(widgets.Text):
    '''
    Inherrent class for FloatSlider
    '''

    def __init__(self, **kwargs):

        self.name = kwargs.pop("name", "widget")
        super().__init__(**kwargs)