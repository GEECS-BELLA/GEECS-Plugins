from typing import Optional, Any
from labview_interface.lv_interface import Bridge


class UserInterface:
    @staticmethod
    def report(message: str):
        Bridge.labview_call('ui', 'report', [], sync=False, message=message)


class Handler:
    @staticmethod
    def send_results(source: str, results: list):
        Bridge.labview_call('handler', 'results', [], sync=False, source=source, results=results)

    @staticmethod
    def question(message: str, possible_answers: list) -> Optional[Any]:
        success, answer = Bridge.labview_call('handler', 'question', ['answer'], sync=True, timeout_sec=600.,
                                              message=message, possible_answers=possible_answers)
        if success:
            return answer['answer']
        else:
            return None

    @staticmethod
    def request_value(message: str, integer: bool = False) -> Optional[Any]:
        success, answer = Bridge.labview_call('handler', 'value', ['answer'], sync=True, timeout_sec=600.,
                                              message=message, integer=integer)
        if success:
            return answer['answer']
        else:
            return None
