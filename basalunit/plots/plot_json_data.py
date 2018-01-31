from pprint import pprint

#==============================================================================

class JsonData:
    """
    Saves data in JSON format
    Note: can be extended in future to provide more flexibility
    """

    def __init__(self, testObj):
        self.testObj = testObj
        self.filename = "json_score_summary"

    def create(self):
        def nested_set(dic, keys, value):
            for key in keys[:-1]:
                dic = dic.setdefault(key, {})
            dic[keys[-1]] = value

        save_data = {}
        save_data.update({"test_name": self.testObj.name})
        save_data.update({"model_name": self.testObj.model_name})

        for key_0 in self.testObj.observation:
            for key_1 in self.testObj.observation[key_0]:
                for key_2 in self.testObj.observation[key_0][key_1]:
                    temp_obs = self.testObj.observation[key_0][key_1][key_2]
                    o_mean = temp_obs[0]
                    o_std = temp_obs[1]
                    p_value = self.testObj.prediction[key_0][key_1][key_2]
                    score = self.testObj.score_dict[key_0][key_1][key_2]
                    feat_name = "{}.{}.{}".format(key_0, key_1, key_2)
                    nested_set(save_data, ["results", feat_name], {"obs_mean": o_mean, "obs_std": o_std, "pred_val": p_value, "score": score})
        save_data.update({"mean_score": self.testObj.score})
        save_data.update({"excluded_features": self.testObj.prob_list})

        filepath = self.testObj.path_test_output + "/" + self.filename + '.json'
        with open(filepath, 'w') as outfile:
            pprint(save_data, stream=outfile)
        return filepath
