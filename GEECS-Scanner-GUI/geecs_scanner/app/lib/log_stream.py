"""
Custom classes to handle the printing out of system messages into both the normal console and the text box on the GUI

TODO Need to return to implementing this for the GUI.  Still a work in progress...
"""


class EmittingStream:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        if text.endswith('\n'):
            text = text[:-1]

        scrollbar = self.text_edit.verticalScrollBar()
        at_bottom = scrollbar.value() == scrollbar.maximum() or scrollbar.maximum() == 0

        self.text_edit.appendPlainText(text)

        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def flush(self):
        pass


class MultiStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()