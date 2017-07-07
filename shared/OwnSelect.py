from ipywidgets import widgets

class OwnSelect(widgets.Select):
    '''
    Inherrent class for Select
    '''

    def __init__(self, **kwargs):

        self.name = kwargs.pop("name", "widget")
        super().__init__(**kwargs)