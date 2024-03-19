# analyze data
import os
import glob
import pickle
import pandas as pd
import numpy as np

# load latests file

files = glob.glob('all_test_results*.pickle')
latest_file = max(files, key=os.path.getctime)
run_uuid = latest_file.split('uuid_')[-1].split('.pickle')[0]

files = glob.glob(f'*{run_uuid}*.pickle')

all_test_results = []
for f in files:
	all_test_results += pickle.load(open(f,'rb'))

def all_list_elements_identical(l:list)->bool:
	l_str = [str(x) for x in l]
	if len(set(l_str))>1:
		return False
	return True

df_metadata = pd.DataFrame([pd.Series(x['metadata']) for x in all_test_results])
# cols_highlight = df.columns[df.apply(lambda x: len(x.apply(str).unique()),axis=0)>1]
# cols_highlight = [c for c in cols_highlight if 'uuid' not in c]
cols_highlight = ['n_vec_index','n_vec_search','single_non_vector_field','results_reproducible']
df_metadata = df_metadata.sort_values(['n_vec_index','single_non_vector_field'])
print(df_metadata[cols_highlight])

reproducibility_fractions = []
for i, test_res in enumerate(all_test_results):
	res_df = pd.DataFrame(test_res['results'])

	same_idx = res_df.groupby(['index_rebuild_id','search_vectors_batch_id','search_vector_within_batch_id'])['res']\
        .apply(lambda x: all_list_elements_identical(x.to_list()))\
     .value_counts(normalize=True)
	same_idx.name = 'same_index_search_reproducibility'

	different_idx = res_df.groupby(['search_vectors_batch_id','search_vector_within_batch_id'])['res']\
        .apply(lambda x: all_list_elements_identical(x.to_list()))\
     .value_counts(normalize=True)
	different_idx.name = 'different_index_search_reproducibility'

	print(test_res['metadata'])
	print('#'*5 + ': Same index repeated search results reproducibility')
	print(same_idx)
	print('#'*5 + ': Different index repeated search results reproducibility')
	print(different_idx)
	sdf = pd.DataFrame([same_idx,different_idx])
	sdf['test_uuid'] = test_res['metadata']['test_uuid']
	reproducibility_fractions.append(sdf)

reproducibility_fractions = pd.concat(reproducibility_fractions).reset_index().pivot(index='test_uuid',values=True,columns='index')

df_summary_results = df_metadata.set_index('test_uuid').join(reproducibility_fractions)

cols_highlight += [same_idx.name,different_idx.name]

df_summary_results_display = df_summary_results[cols_highlight]\
								.rename(columns={'same_index_search_reproducibility':'fraction_reproducible_search_on_SAME_index',
						 						'different_index_search_reproducibility':'fraction_reproducible_search_on_REBUILT_index'})\
								.to_markdown(index=False)


print(df_summary_results_display)
