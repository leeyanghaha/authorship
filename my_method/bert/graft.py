from my_method.bert.run_classifier import DataProcessor


input_file = '/home/yangl/research/authorship/data/glue/MRPC/dev.tsv'
data_processor = DataProcessor()
lines = data_processor._read_tsv(input_file)
print(len(lines))
