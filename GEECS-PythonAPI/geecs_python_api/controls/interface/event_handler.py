import time


class SubscriberExample:
    def __init__(self, name):
        self.name = name

    def print_message(self, message):
        print(f'handler "{self.name}" received message: "{message}"')


class EventHandler:
    def __init__(self, events_names: iter):
        self.events = {event_name: dict() for event_name in events_names}  # dict (events) of dict (subscribers)

    def add_events(self, events_names: iter):
        for event_name in events_names:
            if event_name not in self.events:
                self.events[event_name] = dict()

    def delete_events(self, events_names: iter):
        for event_name in events_names:
            self.events.pop(event_name, None)

    def get_subscribers(self, event_name: str) -> dict:
        return self.events[event_name]

    def register(self, event_name: str, subscriber_name: str, subscriber_method=None):
        self.get_subscribers(event_name)[subscriber_name] = subscriber_method

    def unregister(self, event_name: str, subscriber_name: str):
        self.get_subscribers(event_name).pop(subscriber_name, None)

    def unregister_all(self):
        for event_name, subscriptions in self.events:
            subscriptions.clear()

    def publish(self, event_name: str, *args, **kwargs):
        for subscriber_name, subscriber_method in self.get_subscribers(event_name).items():
            if subscriber_method is not None:
                subscriber_method(*args, **kwargs)

    def publish_all(self, *args, **kwargs):
        for event_name, subscriptions in self.events.items():
            for subscriber_name, subscriber_method in subscriptions.items():
                if subscriber_method is not None:
                    subscriber_method(*args, **kwargs)


def var_to_name(var):
    # noinspection PyTypeChecker
    dict_vars = dict(globals().items())

    var_string = None

    for name in dict_vars.keys():
        if dict_vars[name] is var:
            var_string = name
            break

    return var_string


if __name__ == "__main__":
    print('creating publisher...')
    registration = EventHandler(['new message'])  # only 1 event "new message"

    print('creating handler...')
    message_handler = SubscriberExample('Handler')  # some method to call

    print('registering for incoming "new message"...')
    # registration.register('new message', 'message handler', message_handler.print_message)
    registration.register('new message', var_to_name(message_handler), message_handler.print_message)

    time.sleep(1)
    registration.publish('new message', 'this is a test message!')
    registration.publish_all('this is another test message!')
