from ipywidgets import widgets

class OwnCheckBox(widgets.Checkbox):
    '''
    Inherrent class for Checkbox
    '''

    def __init__(self, **kwargs):

        self.name = kwargs.pop("name", "widget")
        super().__init__(**kwargs)
