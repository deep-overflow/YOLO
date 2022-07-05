import yaml

class Config:
    def __init__(self, dict_config=None):
        self.set_attribute(dict_config)
    
    @staticmethod
    def from_yaml(path):
        # Load configs from yaml file.
        with open(path, 'r') as stream:
            return Config(yaml.load(stream, Loader=yaml.FullLoader))

    @staticmethod
    def from_dict(dict):
        # Load configs from dict.
        return Config(dict)

    @staticmethod
    def get_empty():
        # return empty instance
        return Config()

    def __getattr__(self, key):
        # return value corresponding to key
        return self.__dict__[key]
    
    def __setattr__(self, key, value):
        # set config
        self.set_attribute({key:value})

    def set_attribute(self, dict_config):
        # Set configs
        if dict_config is None:
            return
        
        for key in dict_config.keys():
            if isinstance(dict_config[key], dict):
                self.__dict__[key] = Config(dict_config[key])
            else:
                self.__dict__[key] = dict_config[key]

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    
    def __delitem__(self, key):
        del self.__dict__[key]
    
    def __contains__(self, key):
        return key in self.__dict__
    
    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def update(self, dict_config):
        for key in dict_config.keys():
            if key in self.__dict__.keys():
                if isinstance(dict_config[key], Config):
                    self.__dict__[key].update(dict_config[key])
                else:
                    self.__dict__[key] = dict_config[key]
            else:
                self.__setitem__(key, dict_config[key])

    @classmethod
    def extraction_dictionary(cls, config):
        out = {}
        for key in config.keys():
            if isinstance(config[key], Config):
                out[key] = cls.extraction_dictionary(config[key])
            else:
                out[key] = config[key]
        return out
    