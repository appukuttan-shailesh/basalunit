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

    def create(self):
        filepath = self.testObj.path_test_output + "/" + self.filename + '.txt'
        dataFile = open(filepath, 'w')
        dataFile.write("===================================================================================================\n")
        dataFile.write("Test Name: %s\n" % self.testObj.name)
        dataFile.write("Model Name: %s\n" % self.testObj.model_name)
        # dataFile.write("Score Type: %s\n" % self.testObj.score.description)
        dataFile.write("---------------------------------------------------------------------------------------------------\n")
        header_list = ["Parameter", "Expt. mean", "Expt. std", "Model value", "Score"]
        row_list = []
        for key_0 in self.testObj.observation:
            for key_1 in self.testObj.observation[key_0]:
                for key_2 in self.testObj.observation[key_0][key_1]:
                    temp_obs = self.testObj.observation[key_0][key_1][key_2]
                    o_mean = temp_obs[0]
                    o_std = temp_obs[1]
                    p_value = self.testObj.prediction[key_0][key_1][key_2]
                    score = self.testObj.score_dict[key_0][key_1][key_2]
                    feat_name = "{}.{}.{}".format(key_0, key_1, key_2)
                    row_list.append([feat_name, o_mean, o_std, p_value, score])
        dataFile.write(tabulate(row_list, headers=header_list, tablefmt='orgtbl'))
        dataFile.write("\n---------------------------------------------------------------------------------------------------\n")
        dataFile.write("Mean Score: %s\n" % self.testObj.score)
        if len(self.testObj.prob_list) > 0:
            dataFile.write("\n---------------------------------------------------------------------------------------------------\n")
            dataFile.write("Features excluded (due to invalid scores):\n")
            for feat_name in self.testObj.prob_list:
                dataFile.write(">> %s\n" % feat_name)
        dataFile.write("===================================================================================================\n")
        dataFile.close()
        return filepath
