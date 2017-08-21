from ipywidgets import widgets

class OwnFloatRangeSlider(widgets.FloatRangeSlider):
    '''
    Inherrent class for FloatSlider
    '''

    def __init__(self, **kwargs):
        r'''

        :param \**kwargs:


        :keyword
            name:
        '''
        kwargs['continuous_update'] = False
        self.name = kwargs.pop("name", "widget")
        super().__init__(**kwargs)
