from labview_interface.lv_interface import Bridge


class UserInterface:
    @staticmethod
    def report(message: str):
        Bridge.labview_call('ui', 'report', [], sync=False, message=message)


class Handler:
    @staticmethod
    def send_results(source: str, results: dict):
        Bridge.labview_call('handler', 'results', [], sync=False, source=source, results=str(results))
