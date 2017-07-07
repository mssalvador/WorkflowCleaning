from ipywidgets import widgets

class OwnIntSlider(widgets.IntSlider):
    '''
    Inherrent class for FloatSlider
    '''

    def __init__(self, **kwargs):

        self.name = kwargs.pop("name", "widget")
        super().__init__(**kwargs)