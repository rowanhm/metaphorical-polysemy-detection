import os
import random
from threading import Thread, Lock, Event

from src.shared.global_variables import seed
from src.shared.common import get_file_list, info, open_pickle
from src.training.utils.trainer_utils import reformat_data

random.seed(seed)


class StreamingDataLoader:

    def __init__(self, files_dir, batch_size, shuffle, collate_fn, max_elements=16384, n_threads=3):
        self.all_files = get_file_list(files_dir, end='.pkl')
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.all_files)
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.max_elements = max_elements
        self.n_threads = n_threads
        self.iterator_id = 0
        self.next_iterator = self.new_iterator()
        self.current_iterator = None

    def __iter__(self):
        self.current_iterator = self.next_iterator
        self.next_iterator = self.new_iterator()
        assert self.current_iterator.iterator_id + 1 == self.next_iterator.iterator_id
        info(f'Launching data streamer {self.current_iterator.dataset}:{self.current_iterator.iterator_id}')
        return self.current_iterator

    def new_iterator(self):
        iterator = StreamerIterable(self.all_files.copy(), self.batch_size, self.shuffle, self.collate_fn,
                                    self.max_elements, self.n_threads, iterator_id=self.iterator_id)
        self.iterator_id += 1
        return iterator

    def terminate(self):
        del self.current_iterator
        del self.next_iterator

class StreamerIterable:

    def __init__(self, all_files, batch_size, shuffle, collate_fn, max_elements, n_threads, iterator_id):

        self.iterator_id = iterator_id
        self.dataset = os.path.basename(os.path.dirname(all_files[0]))

        info(f'Initialising data streamer {self.dataset}:{self.iterator_id}')
        self.shuffle = shuffle
        self.max_elements = max_elements

        # Shared elements
        self.file_queue = all_files
        self.file_lock = Lock()

        self.completed = False

        self.buffer_lock = Lock()
        self.buffer = []

        self.running_low = Event()
        self.running_low.set()
        self.buffer_continue = Event()
        self.buffer_continue.clear()

        self.threads = []
        for i in range(n_threads):
            t = Thread(target=self.queue_management, args=())
            t.daemon = True
            t.start()
            self.threads.append(t)

        self.collate_fn = collate_fn
        self.batch_size = batch_size

    def __next__(self):

        # Wait until the buffer has elements (or is completed)
        buffer_depleted = False
        if not self.buffer_continue.is_set():
            info(f'Buffer depleted of streamer {self.dataset}:{self.iterator_id}')
            buffer_depleted = True
        self.buffer_continue.wait()
        if buffer_depleted:
            info(f'Buffer replenished of streamer {self.dataset}:{self.iterator_id}')

        # NB, other thread could now take the lock and edit buffer, but only inserts items

        if self.buffer:  # if the buffer now has elements

            # Take a lock on the buffer and extract a batch
            with self.buffer_lock:
                data = self.buffer[:self.batch_size]
                self.buffer = self.buffer[self.batch_size:]
                if len(self.buffer) == 0 and not self.completed:  # Only stops the execution next time if it not completed
                    self.buffer_continue.clear()
                if len(self.buffer) < self.max_elements:
                    self.running_low.set()

            # Process and return the batch
            return self.collate_fn(data)

        else:
            assert self.completed
            info(f"Streamer {self.dataset}:{self.iterator_id} empty; terminating")
            # self.terminate()
            raise StopIteration

    # def terminate(self):
    #
    #     for i, t in enumerate(self.threads):
    #         info(f'Terminating thread {i+1}/{len(self.threads)}')
    #         t.join()
    #     info(f"Terminated threads of streamer {self.dataset}:{self.iterator_id}")

    def queue_management(self):
        while True:

            # if len(self.buffer) > self.max_elements, wait
            self.running_low.wait()

            # Get a file to process
            last_batch = False
            with self.file_lock:
                if len(self.file_queue) == 0:
                    return
                else:
                    assert len(self.file_queue) > 0
                    file = self.file_queue.pop()
                    if len(self.file_queue) == 0:
                        last_batch = True

            # Load items and process + shuffle them
            data = open_pickle(file)
            data = reformat_data(data)
            if self.shuffle:
                random.shuffle(data)

            # Take a lock on the buffer and append them, then mark completed
            with self.buffer_lock:
                self.buffer += data
                if last_batch:
                    self.completed = True
                self.buffer_continue.set()
                if len(self.buffer) >= self.max_elements and not self.completed:
                    self.running_low.clear()

    # def __del__(self):
    #     info(f'Streamer {self.dataset}:{self.iterator_id} being garbage collected')
    #     with self.file_lock:
    #         self.file_queue = []
    #     self.running_low.set()
    #     self.terminate()
