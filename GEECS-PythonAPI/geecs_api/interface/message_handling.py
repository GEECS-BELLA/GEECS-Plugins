import queue
import inspect
from typing import Optional
from geecs_api.interface.geecs_errors import ErrorAPI


def next_msg(a_queue: queue.Queue, err_msg: Optional[tuple[str, str]] = None,
             block=False, timeout: Optional[float] = None) -> tuple[str, ErrorAPI]:
    err = ErrorAPI()

    try:
        tag, msg, err = a_queue.get(block=block, timeout=timeout)
        a_queue.task_done()
    except queue.Empty:
        msg = ''
    except Exception:
        msg = ''
        if err_msg:
            err = ErrorAPI(err_msg[0], err_msg[1])
        else:
            err = ErrorAPI('Failed to dequeue next message',
                           f'method "{inspect.stack()[1].function}" calling "next_msg"')

    return msg, err


def flush_queue(a_queue: queue.Queue) -> list:
    flushed_msgs = list()

    try:
        while not a_queue.empty():
            try:
                this_element = a_queue.get(block=False)
                flushed_msgs.append(this_element)
                a_queue.task_done()
            except Exception:
                continue

    except Exception:
        pass
    # finally:
    #     a_queue = queue.Queue()

    return flushed_msgs


def async_msg_handler(message: tuple[any, str, ErrorAPI], a_queue: Optional[queue.Queue] = None):
    try:
        cmd_tag, udp_msg, error = message
        if a_queue:
            next_msg(a_queue)  # tmp (to be handled by device) Queue.get() call to dequeue message

        if error.is_error:
            err_str = f'Error:\n\tDescription: {error.error_msg}\n\tSource: {error.error_src}'
        elif error.is_warning:
            err_str = f'Warning:\n\tDescription: {error.error_msg}\n\tSource: {error.error_src}'
        else:
            err_str = ''

        if err_str:
            print(err_str)
        else:
            print(f'Asynchronous UDP response to "{cmd_tag}":\n\t{udp_msg}')

    except Exception as ex:
        err = ErrorAPI(str(ex), 'UdpServer class, method "async_msg_handler"')
        print(err)
