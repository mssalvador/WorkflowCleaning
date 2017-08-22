from ipywidgets import widgets


class OwnIntSingleSlider(widgets.IntSlider):
    '''
    Inherrent class for FloatSlider
    '''

    def __init__(self, **kwargs):


        kwargs['continuous_update'] = False
        self.name = kwargs.pop("name", "widget")
        super().__init__(**kwargs)
