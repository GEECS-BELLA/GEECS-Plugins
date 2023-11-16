from badger import interface


class Interface(interface.Interface):

    name = 'default'

    def __init__(self, params=None):
        super().__init__(params)

        self.states = {}

    @staticmethod
    def get_default_params():
        return None

    def get_value(self, channel: str):
        try:
            value = self.states[channel]
        except KeyError:
            self.states[channel] = value = 0

        return value

    def set_value(self, channel: str, value):
        self.states[channel] = value
