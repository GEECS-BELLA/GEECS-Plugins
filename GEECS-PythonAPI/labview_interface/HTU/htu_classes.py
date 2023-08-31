from labview_interface.lv_interface import Bridge


class UserInterface:
    @staticmethod
    def report(message: str):
        Bridge.labview_call('ui', 'report', ['error'], sync=False, message=message)
