from argparse import ArgumentParser

class MainCommand():

    @staticmethod
    def run():
        parser = ArgumentParser()
        parser.add_argument("--top_k", type=int, default=5, help="Top_k Accuracy")
        parser.add_argument("--erase", default=False, action="store_true", help="Erase all data")
        return parser.parse_args()