"""
Logging to stdout, CSV files and Tensorboard

Adapted from OpenAI SpinningUp (MIT License)
https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py
"""
import atexit
import pathlib
from typing import Any, Dict, Iterable, Union

import gin
import numpy as np
import tqdm


class Logger:
    def __init__(self,
                 output_dir: str = None,
                 log_to_csv: bool = True,
                 log_to_tensorboard: bool = True,
                 csv_headers: Iterable[str] = None):
        """
        :param output_dir: A directory for saving results to. If `None`, no
            files are written to disk, but logging to stdout is still performed
        :param log_to_csv: If `True`, log results to a `.csv`-file
        :param log_to_tensorboard: If `True`, log results to Tensorboard event
            files
        :param csv_headers: List of keys of results to log to `.csv` file. If
            `None`, infer list from set of first values that are logged
        """
        self.log_file = None
        self.csv_file = None
        self.tensorboard_writer = None

        if output_dir is not None:
            self.output_dir = pathlib.Path(output_dir)

            self.log_file = open(output_dir / 'log.txt', 'w')
            atexit.register(self.log_file.close)

            if log_to_csv:
                self.new_csv_file('data.csv')

            if log_to_tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.tensorboard_writer = SummaryWriter(output_dir)
                    atexit.register(self.tensorboard_writer.close)
                except ImportError:
                    print(('Warning: could not import Tensorboard. '
                           'Disabled Tensorboard logging.'))
        else:
            self.output_dir = None

        self.first_row = True
        self.infer_csv_headers = log_to_csv and csv_headers is None
        self.csv_headers = [] if csv_headers is None else csv_headers

        self.log_current_row = {}
        self.log_key_to_tensorboard = set()
        self.max_key_len = 0

        if hasattr(tqdm.tqdm, 'write'):
            # We need to check if it exists because we might have disabled
            # tqdm by replacing `tqdm.tqdm`
            self.stdout_print = tqdm.tqdm.write
        else:
            self.stdout_print = print

    def new_csv_file(self, filename):
        if self.output_dir is not None:
            if self.csv_file is not None:
                self.csv_file.close()
                atexit.unregister(self.csv_file.close)
            self.csv_file = open(self.output_dir / filename, 'w')
            atexit.register(self.csv_file.close)
        self.first_row = True

    def log_message(self, msg: str, color: str = None):
        """Print a colorized message to stdout and write it to log file"""
        if self.log_file is not None:
            self.log_file.write(msg + '\n')
            self.log_file.flush()

        if color is not None:
            msg = colorize(msg, color, bold=True)

        self.stdout_print(msg)

    def log_tabular(self, key: str, value: Any, log_to_tensorboard=False):
        """Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using `log_tabular` to store values for each diagnostic,
        make sure to call `dump_tabular` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row and self.infer_csv_headers:
            self.csv_headers.append(key)
        assert key not in self.log_current_row, \
            (f'You already set {key} this iteration. Maybe you forgot to call '
             f'dump_tabular()')
        self.log_current_row[key] = value
        if log_to_tensorboard:
            self.log_key_to_tensorboard.add(key)

    def dump_tabular(self, step: int, log_stdout=True):
        """Write all of the diagnostics from the current iteration.

        Writes both to stdout, to the output file and Tensorboard

        :param step: Step parameter to associate with current set of
            diagnostics for Tensorboard
        """
        if len(self.log_current_row) > 0:
            if log_stdout:
                self._dump_to_stdout()
            self._dump_to_csv()
            self._dump_to_tensorboard(step)
            self.log_current_row.clear()
            self.first_row = False

    def _dump_to_stdout(self):
        key_lens = (len(key) for key in self.log_current_row)
        self.max_key_len = max(self.max_key_len, 15, max(key_lens))

        key_str = '{:' + str(self.max_key_len) + '}'
        fmt = '| ' + key_str + ' | {:19} |\n'
        n_dashes = 26 + self.max_key_len

        out = '-' * n_dashes + '\n'
        for key, value in self.log_current_row.items():
            val_str = f'{value:8.3g}' if hasattr(value, '__float__') else value
            out += fmt.format(key, val_str)
        out += '-' * n_dashes
        self.log_message(out)

    def _dump_to_csv(self):
        if self.csv_file is not None:
            if self.first_row:
                self.csv_file.write('\t'.join(self.csv_headers) + '\n')
            values = (str(self.log_current_row.get(key, ''))
                      for key in self.csv_headers)
            self.csv_file.write('\t'.join(values) + '\n')
            self.csv_file.flush()

    def _dump_to_tensorboard(self, step: int):
        if self.tensorboard_writer is not None:
            for key, value in self.log_current_row.items():
                if key in self.log_key_to_tensorboard and hasattr(value,
                                                                  '__float__'):
                    self.tensorboard_writer.add_scalar(key, value, step)


class EpochLogger(Logger):
    """A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you
    would use

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in kwargs.items():
            values = self.epoch_dict.get(k)
            if values is None:
                self.epoch_dict[k] = [v]
            else:
                values.append(v)

    def log_tabular(self, key: str, value: Any = None,
                    log_to_tensorboard=False,
                    log_min=False,
                    log_max=False,
                    log_std=False):
        """Log a value or the mean/std/min/max values of a diagnostic.

        :param key: The name of the diagnostic. If you are logging a diagnostic
            whose state has previously been saved with `store`, the key here
            has to match the key you used there.
        :param value: A value for the diagnostic. If you have previously saved
            values for this key via `store`, do *not* provide a `value` here.
        :param log_to_tensorboard:
        :param log_min: If `True`, log minimum of diagnostic
        :param log_max: If `True`, log maximum of diagnostic
        :param log_std: If `True`, log standard deviation of diagnostic
        """
        if value is not None:
            super().log_tabular(key, value)
        else:
            if key not in self.epoch_dict:
                return

            values = self.epoch_dict[key]

            super().log_tabular(key, np.mean(values), log_to_tensorboard)

            if log_min:
                super().log_tabular(key + 'Min', np.min(values),
                                    log_to_tensorboard)
            if log_max:
                super().log_tabular(key + 'Max', np.max(values),
                                    log_to_tensorboard)
            if log_std:
                super().log_tabular(key + 'Std', np.std(values),
                                    log_to_tensorboard)

            del self.epoch_dict[key]


@gin.configurable('logging', blacklist=['output_dir'])
class ConfigurableLogger(EpochLogger):
    """EpochLogger that allows to configure if and how values are logged"""
    def __init__(self, output_dir,
                 log_to_csv=True,
                 log_to_tensorboard=True,
                 keys: Iterable[Union[str, Dict[str, Any]]] = None):
        """
        :param output_dir: A directory for saving results to. If `None`, no
            files are written to disk, but logging to stdout is still performed
        :param log_to_csv: If `True`, log results to a `.csv`-file
        :param log_to_tensorboard: If `True`, log results to Tensorboard event
            files
        :param keys: If `None`, log everything by default. If not `None`,
            expects a list with keys to be logged. Instead of a key, the entry
            can also be a dictionary specifying options to be applied to the
            value. In this case, the key has to be specified under an entry
            `key`. The possible options are

            - `log_csv`: Whether to log this value to csv (default: true)
            - `log_tensorboard: Whether to log this value to Tensorboard
                (default: true)
            - `log_min`: Whether to log the minimum value (default: false)
            - `log_max`: Whether to log the maximum value (default: false)
            - `log_std`: Whether to log the standard deviation of the value
                (default: false)

            Note that when specifying `None`, the keys to use for the
                `.csv`-file are inferred from the first set of keys logged. If
                this behavior is undesired, keys have to be specified here.
        """
        if keys is not None:
            keys_csv = []
            self.keys = set()
            self.keys_tb = set()
            self.keys_min = set()
            self.keys_max = set()
            self.keys_std = set()

            for entry in keys:
                if isinstance(entry, dict):
                    assert 'key' in entry, \
                        'If using an options dict, need to specify a `key`'
                    key = entry['key']

                    def add_key(collection, postfix=''):
                        collection.add(key)
                        if entry.get('log_csv', True):
                            keys_csv.append(key + postfix)
                        if entry.get('log_tensorboard', True):
                            self.keys_tb.add(key + postfix)

                    add_key(self.keys)
                    if entry.get('log_min', False):
                        add_key(self.keys_min, 'Min')
                    if entry.get('log_max', False):
                        add_key(self.keys_max, 'Max')
                    if entry.get('log_std', False):
                        add_key(self.keys_std, 'Std')
                else:
                    key = entry
                    self.keys.add(key)
                    keys_csv.append(key)
                    self.keys_tb.add(key)
        else:
            self.keys = None
            keys_csv = None

        super().__init__(output_dir,
                         log_to_csv,
                         log_to_tensorboard,
                         keys_csv)

    def log_tabular(self, key, value, **kwargs):
        if self.keys is None:
            super().log_tabular(key, value, **kwargs)
        elif key in self.keys:
            super_kwargs = {
                'log_to_tensorboard': key in self.keys_tb,
                'log_min': key in self.keys_min,
                'log_max': key in self.keys_max,
                'log_std': key in self.keys_std
            }
            super_kwargs.update(kwargs)
            super().log_tabular(key, value, **super_kwargs)

    def dump_tabular(self, step: int, log_stdout=True):
        for key in sorted(list(self.epoch_dict)):
            self.log_tabular(key, value=None)

        super().dump_tabular(step, log_stdout)


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """Colorize a string"""
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')

    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)
