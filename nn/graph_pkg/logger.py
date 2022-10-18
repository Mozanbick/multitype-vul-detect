class Logger:

    def __init__(self, log_file: str):
        self.log_path = log_file

    def __call__(self, msg: str):
        with open(self.log_path, "a") as fp:
            fp.write(msg + "\n")
        print(msg)
