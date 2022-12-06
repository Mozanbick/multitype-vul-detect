import time


class Logger:

    def __init__(self, log_file: str, patience=None):
        self.log_path = log_file
        if patience and patience <= 0:
            patience = None
        self.patience = patience

    def __call__(self, msg: str, step=None):
        if step and self.patience:
            if (step + 1) % self.patience == 0:
                with open(self.log_path, "a") as fp:
                    fp.write(msg + "\n")
                print(msg)
        else:
            with open(self.log_path, "a") as fp:
                fp.write(msg + "\n")
            print(msg)


def make_run_id(model_name: str, task_name: str, run_name=None) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    if run_name is not None:
        return run_name
    else:
        return "%s_%s__%s" % (model_name, task_name, time.strftime("%Y-%m-%d_%H-%M-%S"))
