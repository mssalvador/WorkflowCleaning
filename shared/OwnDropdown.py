from ipywidgets import widgets

class OwnDropdown(widgets.Dropdown):
    '''
    Inherrent class for FloatSlider
    '''

    def __init__(self, **kwargs):

        self.name = kwargs.pop("name", "widget")
        super().__init__(**kwargs)