from tabulate import tabulate

#==============================================================================

class TxtTable:
    """
    Displays data in table inside text file
    Note: can be extended in future to provide more flexibility
    """

    def __init__(self, testObj):
        self.testObj = testObj
        self.filename = "score_summary"

    def create(self, mid_keys = []):
        filepath = self.testObj.path_test_output + "/" + self.filename + '.txt'
        dataFile = open(filepath, 'w')
        dataFile.write("===================================================================================================\n")
        dataFile.write("Test Name: %s\n" % self.testObj.name)
        dataFile.write("Model Name: %s\n" % self.testObj.model_name)
        dataFile.write("Score Type: %s\n" % self.testObj.score.description)
        dataFile.write("---------------------------------------------------------------------------------------------------\n")
        header_list = ["Parameter", "Expt. mean", "Expt. std", "Model value", "Score"]
        row_list = []
        for key in self.testObj.observation.keys():
            for key2 in self.testObj.observation[key]["soma"].keys():
                temp_obs = self.testObj.observation[key]["soma"][key2]
                o_mean = temp_obs[0]
                o_std = temp_obs[1]
                # pred_feature_dict and score_dict currently use keys in different format
                mod_key = "{}.{}.{}".format(key, "soma", key2)
                p_value = self.testObj.pred_feature_dict[mod_key]
                score = self.testObj.score_dict[mod_key]
                row_list.append([mod_key, o_mean, o_std, p_value, score])
        dataFile.write(tabulate(row_list, headers=header_list, tablefmt='orgtbl'))
        dataFile.write("\n---------------------------------------------------------------------------------------------------\n")
        dataFile.write("Final Score: %s\n" % self.testObj.score)
        dataFile.write("===================================================================================================\n")
        dataFile.close()
        return filepath
