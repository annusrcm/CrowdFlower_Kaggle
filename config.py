import os


class ParamConfig:
    def __init__(self):
        self.num_labels = 4
        self.vocab_size = 5000
        self.batch_size = 100
        self.num_epochs = 10
        self.drop_html_flag = True

        self.data_folder = "/home/annu/Downloads/data/crowdflower-search-relevance/"
        self.model_folder = "/home/annu/Downloads/data/crowdflower-search-relevance/models/"
        self.logs_folder = "/home/annu/Downloads/data/crowdflower-search-relevance/logs/"

        self.train_path = os.path.join(self.data_folder, "small_train.csv")
        self.train_preprocessed_path = os.path.join(self.data_folder, "small_train_pp.csv")
        self.test_path = os.path.join(self.data_folder, "small_test.csv")
        self.test_preprocessed_path = os.path.join(self.data_folder, "small_test_pp.csv")
        self.synonyms_csv = os.path.join(self.data_folder, "synonyms.csv")
        self.model_name = os.path.join(self.model_folder, "small_keras_sequence.h5")
        self.tokenizer_name = os.path.join(self.model_folder, "small_keras_tokenizer.pickle")
        self.encoder_name = os.path.join(self.model_folder, "small_encoded_class.json")

        self.create_folder_if_not_present(self.model_folder)
        self.create_folder_if_not_present(self.logs_folder)

        # such dict is found by exploring the training data
        self.replace_dict = {
            "nutri system": "nutrisystem",
            "soda stream": "sodastream",
            "playstation's": "ps",
            "playstations": "ps",
            "playstation": "ps",
            "(ps 2)": "ps2",
            "(ps 3)": "ps3",
            "(ps 4)": "ps4",
            "ps 2": "ps2",
            "ps 3": "ps3",
            "ps 4": "ps4",
            "coffeemaker": "coffee maker",
            "k-cups": "k cup",
            "k-cup": "k cup",
            "4-ounce": "4 ounce",
            "8-ounce": "8 ounce",
            "12-ounce": "12 ounce",
            "ounce": "oz",
            "button-down": "button down",
            "doctor who": "dr who",
            "2-drawer": "2 drawer",
            "3-drawer": "3 drawer",
            "in-drawer": "in drawer",
            "hardisk": "hard drive",
            "hard disk": "hard drive",
            "harley-davidson": "harley davidson",
            "harleydavidson": "harley davidson",
            "e-reader": "ereader",
            "levi strauss": "levi",
            "levis": "levi",
            "mac book": "macbook",
            "micro-usb": "micro usb",
            "screen protector for samsung": "screen protector samsung",
            "video games": "videogames",
            "game pad": "gamepad",
            "western digital": "wd",
            "eau de toilette": "perfume",
        }

    def create_folder_if_not_present(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
