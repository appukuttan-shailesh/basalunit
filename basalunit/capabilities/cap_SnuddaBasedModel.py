import os
import json
from sciunit import Model, Capability
from snudda.init.init import SnuddaInit
from snudda.place import SnuddaPlace
from snudda.detect import SnuddaDetect, SnuddaPrune
from hbp_validation_framework import ModelCatalog

try:
    from clb_nb_utils import oauth
    have_collab_token_handler = True
except ImportError:
    have_collab_token_handler = False


class SnuddaBasedModel(Model, Capability):
    """For models developed using Snudda"""

    def __init__(self,
                 name="Snudda Based Model",
                 network_path=os.path.join(".", "networks"),
                 n_dSPN=120,
                 n_iSPN=120,
                 n_FS=20,
                 n_LTS=0,
                 n_ChIN=0,
                 volume_type="slice",
                 neuron_density=80500,
                 side_len=200e-6,
                 slice_depth=150e-6,
                 mod_files_path="./Snudda/snudda/data/neurons/mechanisms",
                 random_seed=None,
                 ebrains_username=None):

        Model.__init__(self, name=name)

        if ebrains_username != False:
            model_params = {
                "n_dSPN": n_dSPN,
                "n_iSPN": n_iSPN,
                "n_FS": n_FS,
                "n_LTS": n_LTS,
                "n_ChIN": n_ChIN,
                "volume_type": volume_type,
                "neuron_density": neuron_density,
                "side_len": neuron_density,
                "slice_depth": slice_depth,
                "mod_files_path": mod_files_path,
                "random_seed": random_seed
            }
            self.link_model_catalog(model_params, ebrains_username)

        if os.path.isfile(network_path):
            self.network_file = network_path
            self.network_path = os.path.dirname(network_path)
        else:
            self.network_file = os.path.join(
                network_path, "network-synapses.hdf5")
            self.network_path = network_path

        config_name = os.path.join(self.network_path, "network-config.json")
        cnc = SnuddaInit(struct_def={},
                         config_file=config_name,
                         random_seed=random_seed)
        cnc.define_striatum(num_dSPN=n_dSPN,
                            num_iSPN=n_iSPN,
                            num_FS=n_FS,
                            num_LTS=n_LTS,
                            num_ChIN=n_ChIN,
                            volume_type=volume_type,
                            side_len=side_len,
                            slice_depth=slice_depth,
                            neuron_density=neuron_density)

        dir_name = os.path.dirname(config_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        cnc.write_json(config_name)

        sp = SnuddaPlace(network_path=network_path)
        sp.place()

        sd = SnuddaDetect(network_path=network_path)
        sd.detect()

        spr = SnuddaPrune(network_path=network_path)
        spr.prune()
        spr = None

        self.compile_mod_files(mod_files_path)

    def link_model_catalog(self, model_params, ebrains_username):
        self.model_alias = "striatum_microcircuit"
        print("\nLink to model on model catalog: ")
        print("https://model-catalog.brainsimulation.eu/#model_alias." + self.model_alias)

        model_params_str = json.dumps(
            dict(sorted(model_params.items(), key=lambda item: item[0])))

        self.model_version = None
        mc = ModelCatalog(username=ebrains_username)

        list_model_instances = mc.list_model_instances(alias=self.model_alias)
        for model_inst in list_model_instances:
            if(model_inst["parameters"] == model_params_str):
                print("\nExisting model instance found with same parameters!") 
                print("Version = {}\n\n".format(model_inst["version"]))
                self.model_version = model_inst["version"]
                break

        if not self.model_version:
            print("\nNo existing model instance found for these parameters!")
            print("Create new model instance? Enter: y / n")
            choice = input().lower()
            valid_choices = {"yes": True, "y": True, "no": False, "n": False}
            if valid_choices[choice]:
                flag = True
                while flag:
                    print("\nPlease provide a short name (version) for this model instance: ")
                    model_inst_name = input()
                    flag = False
                    for model_inst in list_model_instances:
                        if(model_inst["version"] == model_inst_name):
                            print("This verison name already exists!")
                            flag = True
                            break
                print("\nCreating model instance...")
                new_model_inst = mc.add_model_instance(alias=self.model_alias,
                                                        source="https://github.com/Hjorthmedh/Snudda",
                                                        version=model_inst_name,
                                                        parameters=model_params_str,
                                                        description="Install PyPI package 'snudda' to use this model instance.")
                if new_model_inst:
                    print("Model instance created! UUID: {}\n\n".format(new_model_inst["id"]))
                    self.model_version = new_model_inst["version"]

    def compile_mod_files(self, mod_files_path):
        print("\nCompiling mod files...")
        if mod_files_path is None:
            raise Exception(
                "Please give the path to the mod files (eg. mod_files_path = \'/home/models/CA1_pyr/mechanisms/\') as an argument to the ModelLoader class")
        if os.path.isfile(os.path.join(".", 'x86_64/.libs/libnrnmech.so')) is False:
            os.system("nrnivmodl " + mod_files_path)
        else:
            print("mod files were previosuly compiled. Loading existing 'libnrnmech.so'\n\n")