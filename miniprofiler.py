"""
    
    Mini profiling tool for your projects
    https://github.com/kadir014/miniprofiler

"""

from dataclasses import dataclass
from contextlib import contextmanager
from time import perf_counter
from statistics import stdev


__version__ = "0.0.1"


def percentile(samples: list[float], percent: float) -> float:
    """ Calculate the Nth percentile given timing samples. """
    
    n = len(samples)
    if n == 0:
        return 0.0

    samples = sorted(samples)
    k = (n - 1.0) * (percent / 100.0)
    f = int(k)
    c = int(min(f + 1, n - 1))

    if f == c:
        return samples[int(k)]

    return samples[f] + (samples[c] - samples[f]) * (k - f)


@dataclass
class ProfiledStat:
    """
    Timings of a single stat over a period of samples.

    Attributes
    ----------
    avg
        Average of timings in seconds.
    min
        Minimum of timings in seconds.
    max
        Maximum of timings in seconds.
    p95
        95th percentile timing in seconds.
    p99
        99th percentile timing in seconds.
    stdev
        Standard deviation of timings in seconds.
    last
        Latest timing sample in seconds.
    samples
        Number of timing samples the profiler used to calculate.
    name
        Name of the profiled stat.
    """

    avg: float
    min: float
    max: float
    p95: float
    p99: float
    stdev: float
    last: float
    samples: int
    name: str


class Profiler:
    """
    Profiler and timing storage class.
    """
    
    def __init__(self, accumulation_limit: int = 30) -> None:
        """
        Parameters
        ----------
        accumulation_limit
            Maximum number of samples to store per each stat.
            If you're calling the `profile` context manager method each frame,
            this would be equal to N frames to accumulate timing samples for.
        """
        self.__timings = dict()

        self.accumulate_limit = accumulation_limit

    def __getitem__(self, stat: str) -> ProfiledStat:
        """ Return a profiled stat. """

        return ProfiledStat(
            avg = self.__timings[stat]["avg"],
            min = self.__timings[stat]["min"],
            max = self.__timings[stat]["max"],
            p95 = self.__timings[stat]["p95"],
            p99 = self.__timings[stat]["p99"],
            stdev = self.__timings[stat]["stdev"],
            last = self.__timings[stat]["last"],
            samples = len(self.__timings[stat]["acc"]),
            name=stat
        )

    def register(self, stat: str) -> None:
        """
        Register a stat to the profiler.
        
        Parameters
        ----------
        stat
            Stat name to register.
        """

        self.__timings[stat] = {
            "acc": [],
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "last": 0.0,
            "stdev": 0.0
        }

    @contextmanager
    def profile(self, stat: str):
        """
        Profile piece of code.

        ```
        profiler = Profiler(...)
        profiler.register("my_stat_name")
        with profiler.profile("my_stat_name"):
            # your code here...
        ```
        
        Parameters
        ----------
        stat
            Stat name to store the profiled code as.
        """

        start = perf_counter()
        
        try: yield None

        finally:
            elapsed = perf_counter() - start
            self.accumulate(stat, elapsed)

    def accumulate(self, stat: str, value: float) -> None:
        """
        Accumulate stat value.

        You don't really have a reason to call this manually, instead use
        the `profile` context manager method.
        
        Parameters
        ----------
        stat
            Stat name to accumulate.
        value
            Stat value to accumulate.
        """

        acc = self.__timings[stat]["acc"]
        acc.append(value)

        if len(acc) > self.accumulate_limit:
            acc.pop(0)

        n = len(acc)

        if n < 2:
            s_stdev = 0.0
        else:
            s_stdev = stdev(acc)

        self.__timings[stat]["avg"] = sum(acc) / n
        self.__timings[stat]["min"] = min(acc)
        self.__timings[stat]["max"] = max(acc)
        self.__timings[stat]["p95"] = percentile(acc, 95.0)
        self.__timings[stat]["p99"] = percentile(acc, 99.0)
        self.__timings[stat]["last"] = value
        self.__timings[stat]["stdev"] = s_stdev