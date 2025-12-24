import datetime
from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from signal_sources.generated import GeneratedSource


def plot_signals(*sources, lifetime: timedelta = timedelta(seconds=5), interval=30):
    fig, ax = plt.subplots()

    lines = []
    for source in sources:
        line, = ax.plot([], [], label=source.title)
        lines.append(line)

    ax.set_xlabel("Время")
    ax.set_ylabel("Амплитуда")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    def update(frame):
        any_data = False

        for src, line in zip(sources, lines):
            buffer = src.get_buffer()
            if not buffer:
                continue

            ts, values = zip(*buffer)
            line.set_data(ts, values)
            any_data = True

        if any_data:
            now = datetime.datetime.now()
            ax.set_xlim(now - lifetime, now)
            ax.relim()
            ax.autoscale_view()

        return lines

    ani = FuncAnimation(fig, update, interval=interval)
    plt.show()


if __name__ == "__main__":
    sources = []
    global_livetime = timedelta(seconds=5)


    def sin_by_time(dt: datetime):
        import numpy as np

        ms = dt.second * 1000 + dt.microsecond / 1000

        phase_ms = ms % 5_000

        value = np.sin(2 * np.pi * phase_ms / 5_000)

        return value


    generator_source = GeneratedSource(
        sin_by_time,
        interval=0.01,
        livetime=global_livetime
    )

    sources.append(generator_source)

    try:
        for source in sources:
            source.start()

        plot_signals(*sources, lifetime=global_livetime)
    except Exception as e:
        print("Starting error: {}".format(e))

    for source in sources:
        source.stop()
