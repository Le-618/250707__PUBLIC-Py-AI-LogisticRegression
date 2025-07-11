# Basically the custom class
#
# That works just like standart PRINT function
#   // print(*args, sep, end) // log(*args, sep, end)
#
# But allows to make the indentation levels in some places using the "WITH" keyword:
#
# // log ("mama", "papa", sep=",")              >>>mama,papa
# // with log:
# //     log("baby", end=" :)")                 >>>  baby
# //     with log:
# //         log("toy 1", "toy 2", sep="\n")    >>>    toy 1
# //                                            >>>    toy 2
# //     log("second baby", end=" :D")          >>>  second baby :D
# // log("granny")                              >>>granny
#
#

import datetime


class Log(object):
    _lv: int

    def __init__(self):
        self._lv = 0
        self._line_ended = True
        self._smallest_indent = "    "

    def __enter__(self):
        self._lv += 1
        if not self._line_ended:
            print()
            self._line_ended = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            raise exc_type(exc_val)
        self._lv -= 1
        if not self._line_ended:
            print()
            self._line_ended = True
        return self

    def __call__(self, *args, sep: str = " ", end: str = "\n"):
        res = sep.join(map(str, args)) + end
        lines = res.split("\n")
        for line_index in range(len(lines)):
            line = lines[line_index]
            if line_index == 0:
                indent = self._smallest_indent * self._lv     if self._line_ended else ""
                print(indent, line.replace("\r", "\r" + indent), sep="", end="\n" if len(lines) != 1 else "")
                self._line_ended = not self._line_ended and line.find("\r") != -1
            elif line_index != len(lines) - 1:
                indent = self._smallest_indent * self._lv
                print(indent, line.replace("\r", "\r" + indent), sep="", end="\n")
            else:
                self._line_ended = (line == "")
                indent = self._smallest_indent * self._lv if not self._line_ended else ""
                print(indent, line.replace("\r", "\r" + indent), sep="", end="")
        return self


log = Log()


def log_timer_start(do_logging: bool = True) -> datetime.datetime:
    global log
    start_time = datetime.datetime.now()
    if do_logging: log(f" Starting time: {start_time:%b %d %H:%M:%S.%f}")
    return start_time

def log_timer_end(start_time) -> None:
    global log
    finish_time = datetime.datetime.now()
    log(f" Finish time:   {finish_time:%b %d %H:%M:%S.%f}")
    total_time = finish_time - start_time
    log(f" Total time:     ",
        "{0:>8}".format("{0:02d}".format(int(total_time.total_seconds()//3600))),
        f":{int(total_time.total_seconds()%3600//60):02d}:",
        f"{int(total_time.total_seconds()%60):02d}",
        f"{total_time.total_seconds()%1:6f}"[1:], sep="")



if __name__ == "__main__":
    y=1
    with log:
        log(9)
    with log:
        log(y.a)
    log ("mama", "papa", sep=",")
    with log:
        log("baby", end=" :D")
        with log:
            log("toy 1", end="\n")
            log("toy 2", end="")
            log("toy 3")
        log("second baby", end=" :)")
    log("granny")
    with log:
        log("testing \\r:")
        for i in range(10):
            log("\r", end="")
            log(f"epoch: {i}", ")", end="")
    log("testing complete")