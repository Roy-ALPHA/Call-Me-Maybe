from .core import *

def run():
    engine = ArgParser().parse_validate_args()
    engine.call_me_maybe()

if __name__ == "__main__":
    try:
        run()
    except ArgumentTypeError as e:
        print(e)

