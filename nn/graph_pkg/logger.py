import time


class Logger:

    def __init__(self, log_file: str):
        self.log_path = log_file

    def __call__(self, msg: str):
        with open(self.log_path, "a") as fp:
            fp.write(msg + "\n")
        print(msg)


def make_run_id(model_name: str, task_name: str, run_name=None) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    if run_name is not None:
        return run_name
    else:
        return "%s_%s__%s" % (model_name, task_name, time.strftime("%Y-%m-%d_%H-%M-%S"))
