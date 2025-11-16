from ai_eval.services.file import YAMLService

my_yaml = YAMLService(path="ai_eval/config/input_output.yaml")
io = my_yaml.doRead()

my_yaml2 = YAMLService(path="ai_eval/config/model_config.yaml")
model_list = my_yaml2.doRead()

# my_yaml3 = YAMLService(path="ai_eval/config/rag_config.yaml")
# rag_model_list = my_yaml3.doRead()
