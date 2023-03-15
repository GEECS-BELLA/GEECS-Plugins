import queue
import inspect
import threading
from typing import Optional
from geecs_api.interface.geecs_errors import ErrorAPI, api_error
from geecs_api.interface.event_handler import EventHandler


class NetworkMessage:
    def __init__(self, tag: str = '', stamp: str = '', msg: str = '', err: ErrorAPI = ErrorAPI()):
        self.tag = tag
        self.stamp = stamp
        self.msg = msg
        self.err = err


def next_msg(a_queue: queue.Queue, block=False, timeout: Optional[float] = None) -> NetworkMessage:
    try:
        net_msg: NetworkMessage = a_queue.get(block=block, timeout=timeout)
        a_queue.task_done()
    except queue.Empty:
        net_msg = NetworkMessage()
    except Exception as ex:
        net_msg = NetworkMessage()
        api_error.error('Failed to dequeue next message\n' + str(ex),
                        f'Method "{inspect.stack()[1].function}" calling "next_msg"')

    return net_msg


def flush_queue(a_queue: queue.Queue) -> list:
    flushed_msgs = list()

    try:
        while not a_queue.empty():
            try:
                this_element: NetworkMessage = a_queue.get(block=False)
                flushed_msgs.append(this_element)
                a_queue.task_done()
            except Exception:
                continue

    except Exception:
        pass
    # finally:
    #     a_queue = queue.Queue()

    return flushed_msgs


# Legacy functions
def async_msg_handler(message: NetworkMessage, a_queue: Optional[queue.Queue] = None):
    try:
        if a_queue:
            next_msg(a_queue)  # tmp (to be handled by device) Queue.get() call to dequeue message

        if message.err.is_error or message.err.is_warning:
            print(message.err)
        elif message.stamp:
            print(f'Asynchronous UDP response to "{message.tag}":\n\t{message.stamp}\n\t{message.msg}')
        else:
            print(f'Asynchronous UDP response to "{message.tag}":\n\tno timestamp\n\t{message.msg}')

    except Exception as ex:
        err = ErrorAPI(str(ex), 'Module message_handling, method "async_msg_handler"')
        print(err)


def broadcast_msg(net_msg: NetworkMessage,
                  notifier: Optional[threading.Condition] = None,
                  queue_msgs: Optional[queue.Queue] = None,
                  publisher: Optional[EventHandler] = None,
                  event_name: str = ''):
    """ Queue, notify & publish message """

    if notifier:
        with notifier:
            if queue_msgs:
                queue_msgs.put(net_msg)

            notifier.notify_all()

            if publisher and event_name.strip():
                publisher.publish(event_name, net_msg, queue_msgs)
    else:
        if queue_msgs:
            queue_msgs.put(net_msg)
        if publisher and event_name.strip():
            publisher.publish(event_name, net_msg, queue_msgs)
