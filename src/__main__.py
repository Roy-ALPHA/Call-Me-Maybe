from .core import *

def run():
    fn_caller = ArgParser().parse_validate_args()
    fn_caller.ano()

if __name__ == "__main__":
    try:
        run()
    except ArgumentTypeError as e:
        print(e)

