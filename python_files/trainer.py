import pandas as pd
import numpy as np
from tqdm import *
import os
import pickle as pkl


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


#Hint utilize gep-mem from gep-llm
#Dummy class symbol
class Symbol(object):

    def __init__(self, name, arity, **kwargs):
        self.name = name
        self.arity = arity
        self.nice_name = kwargs.get('nice_name',name)

    def get(self):
        return self.name


def safe_convert(elem, defaults):
    try:
        elem_val = float(elem)
        if elem_val < 0:
            return defaults[0]
        elif elem_val > 0:
            return defaults[1]
        else:
            return "0.0"
    except ValueError:
        return elem


def tokenize(string_raw, defaults):
    list_string = string_raw.split(' ')
    ret_val = []
    for elem in list_string:
        ret_val.append(safe_convert(elem, defaults=defaults))
    return ret_val


def simplify_tree(geno_part, rounds=3, clean_rules={}):
    gene = ",".join(geno_part)
    for _ in range(rounds):
        for key,rule in clean_rules.items():
             gene = gene.replace(key, rule)
    return gene.split(",")


def get_pheno_part_from_gene(gene_, symtable):
    elements = []
    temp_arity = 0
    counter = 0
    while True:
        elem = gene_[counter]
        if elem !="":
            if symtable.get(elem,0) > 0 and len(elements) == 0:
                    temp_arity += symtable.get(elem,0)
                    elements.append(elem)
                    counter += 1
                    continue

            temp_arity += symtable.get(elem,0)
            elements.append(elem)
            temp_arity += -1


            if temp_arity <= 0:
                    break
        counter += 1
    return elements

input_features_token = {"I1":"I1","I2":"I2","N1":"N1","N2":"N2","N3":"N3"}
constant_token = {"0.0": "0.0"}
number_token = {"C1":"C1","C2":"C2"}
signs = {'+':'+','-':'-','/':'/','*':'*'}

gene_count = 4
head_len = 5
gene_len = head_len *2 +1
fun_cols = [0,1,2]
fit_col = 3
sym_dict = {"+": 2, "-": 2, "*": 2, "/": 2, "sign": 1, "sqrt": 1, "sin": 1,
            "cos": 1, "sinh": 1, "sqr": 1, "log": 1, "cosh": 1, "inv": 1}

simplification_rules = {
    "+,C1,C1" : "C1",
    "-,C1,C1" : "C1",
    "/,C1,C1" : "C1",
    "*,C1,C1" : "C1",
    "+,C2,C2" : "C2",
    "-,C2,C2" : "C2",
    "/,C2,C2" : "C2",
    "*,C2,C2" : "C2",
    "+,C2,C1" : "C1",
    "-,C2,C1" : "C2",
    "/,C2,C1" : "C2",
    "*,C2,C1" : "C2",
    "+,C1,C2" : "C1",
    "-,C1,C2" : "C1",
    "/,C1,C2" : "C2",
    "*,C1,C2" : "C2",

    "+,0.0,C1" : "C1",
    "-,0.0,C1" : "C1",
    "/,0.0,C1" : "0.0",
    "*,0.0,C1" : "0.0",
    "+,C1,0.0" : "C1",
    "-,C1,0.0" : "C1",
    "/,C1,0.0" : "0.0",
    "*,C1,0.0" : "0.0",

    "+,0.0,C2" : "C2",
    "-,0.0,C2" : "C2",
    "/,0.0,C2" : "0.0",
    "*,0.0,C2" : "0.0",
    "+,C2,0.0" : "C2",
    "-,C2,0.0" : "C2",
    "/,C2,0.0" : "0.0",
    "*,C2,0.0" : "0.0",

    "*,I1,0.0" : "0.0",
    "*,I2,0.0" : "0.0",
    "*,N1,0.0" : "0.0",
    "*,N2,0.0" : "0.0",
    "*,N3,0.0" : "0.0",

    "*,0.0,I1" : "0.0",
    "*,0.0,I2" : "0.0",
    "*,0.0,N1" : "0.0",
    "*,0.0,N2" : "0.0",
    "*,0.0,N3" : "0.0",

    "+,I1,0.0" : "I1",
    "+,I2,0.0" : "I2",
    "+,N1,0.0" : "N1",
    "+,N2,0.0" : "N2",
    "+,N3,0.0" : "N3",

    "+,0.0,I1" : "I1",
    "+,0.0,I2" : "I2",
    "+,0.0,N1" : "N1",
    "+,0.0,N2" : "N2",
    "+,0.0,N3" : "N3",

    "-,I1,0.0" : "I1",
    "-,I2,0.0" : "I2",
    "-,N1,0.0" : "N1",
    "-,N2,0.0" : "N2",
    "-,N3,0.0" : "N3",


    "-,0.0,0.0" : "0.0",
    "+,0.0,0.0" : "0.0",
    "*,0.0,0.0" : "0.0",
    "/,0.0,0.0" : "0.0",
    
}


#Load your dataframe
frame = [pd.read_csv(os.path.join("expression_data",elem), header =None, index_col=False) for elem in os.listdir("expression_data") if "csv" in elem]

indis = pd.concat(frame,ignore_index=True)
print(len(indis))
indis = indis.drop_duplicates(subset=fit_col)
print(len(indis))

#re-arrange the dataframe

forms = indis[fun_cols]
for col in forms.columns:
    temp = forms[col].apply(
        lambda x: tokenize(x, ["C1", "C2"])
    )
    genes = []
    for elem in temp.values:
        links = elem[:gene_count-1]
        elem = elem[gene_count-1:]
        chromo = []
        for counter in range(len(elem)//gene_len):
            temp_gene = elem[counter:gene_len]
            elem = elem[gene_len:]
            temp_gene = simplify_tree(temp_gene, rounds=8, clean_rules=simplification_rules)
            temp_gene = get_pheno_part_from_gene(temp_gene,sym_dict)
            chromo.append(temp_gene)
        genes.append([chromo, links])
    indis[col] = genes

#select particular forms from the frame
min_fit = np.nanmin(indis[fit_col].values)*1.4
selection = indis[indis[fit_col]<min_fit]
indis_raw = selection[fun_cols]

print(f"Seclection len: {len(selection)}")

#For simple transfer learning we can just initilize embedding with ones ore zeros otherwise we use the flowfield data
#embed_inds = [np.ones((30, 1))]*len(forms)
flow_features = pd.read_csv("flow_field_data/flow_features_duct_180.csv")
n_points = 100 # e.g sample n points
embed_inds = []
for _ in enumerate(forms):
    embed_inds.append(
        flow_features.sample(n_points).values
    )

#create the training_set:
trainee = [[indi_raw, 1, embed, 0] for indi_raw,embed in zip(indis_raw[fun_cols].values, embed_inds)]


## Create all the memory stuff - 1. Vocab 2. Architecture

I1 = Symbol("I1", 0)
I2 = Symbol("I2", 0)
N1 = Symbol("N1", 0)
N2 = Symbol("N2", 0)
N3 = Symbol("N3", 0)
zero = Symbol("0.0", 0)
one = Symbol("1.0",0)

coeff1 = Symbol("C1", 0)
coeff2 = Symbol("C2", 0)

times = Symbol("*", 2)
plus = Symbol("+", 2)
minus = Symbol("-", 2)
symlist = [times, plus, minus, I1, I2, N1, N2, N3, coeff1, coeff2, zero, one]


#Architecture
test_conf = "c4"

names_tests = [test_conf]
params_to_test = {
        "head_size":{"c1":4, "c2":16},
        "blocks":{"c3":2, "c4":8},
        "embed_dim":{"c5":256,"c6":64},
        "train_epochs":{"c7":250,"c8":1000}
        }


for elem in names_tests:
    head_length = 5 # In case you only train, this can be any value, just important for inference
    gene_count = 4
    embedding_size = params_to_test["embed_dim"].get(elem,128)
    num_blocks = params_to_test["blocks"].get(elem,4)
    head_size = params_to_test["head_size"].get(elem,8)
    train_epochs = params_to_test["train_epochs"].get(elem,500)
    
    memory_config = MemConfig()
    memory_config.set_len(head_length). \
        set_symbol_set(symlist). \
        set_embed_dim(embedding_size).set_attention_head_size(head_size).set_number_expression_count(3). \
        set_number_k_sampling(2).set_num_blocks(num_blocks).set_feature_dim(n_featues).set_gene_count(gene_count).set_constant_symbols({"C1":-1, "C2":1})

    gene_memory = GenMemory(memory_config, input_len=128)
    model_wrapper = gene_memory.get_memory()


    hist = model_wrapper.train(trainee, max_y_size=30, batch_size=256, epochs=train_epochs, frequency=100,check_point_path=f"config_save_{elem}.save")
    handler = open(f"test_hist_{elem}.dat", 'wb')
    pkl.dump(hist,handler)
    handler.close()

    del model_wrapper