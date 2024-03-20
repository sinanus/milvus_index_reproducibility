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

def compare_two_hits_dict(hits0:dict, hits1: dict):
	intersecting_keys = set(list(hits0.keys())).intersection(list(hits1.keys()))
	n_intersecting_keys = len(intersecting_keys)
	intersecting_keys_value_deltas = []
	if n_intersecting_keys>0:
		for k in intersecting_keys:
			intersecting_keys_value_deltas.append(hits0[k]-hits1[k])
	if len(hits0.values()) == len(hits1.values()):
		ordered_keys_value_delta = [a-b for a,b in zip(hits0.values(),hits1.values())]
	else:
		n = len(hits0.values()) if len(hits0.values())<len(hits1.values()) else len(hits1.values())
		ordered_keys_value_delta = [hits0.values()[i] - hits1.values()[i] for i in range(n)]
	ordered_keys_MAE = np.mean(np.abs(ordered_keys_value_delta))
	return ordered_keys_MAE, n_intersecting_keys, intersecting_keys_value_deltas



df_metadata = pd.DataFrame([pd.Series(x['metadata']) for x in all_test_results])
# cols_highlight = df.columns[df.apply(lambda x: len(x.apply(str).unique()),axis=0)>1]
# cols_highlight = [c for c in cols_highlight if 'uuid' not in c]
cols_highlight = ['n_vec_index','n_vec_search','single_non_vector_field','results_reproducible']
df_metadata = df_metadata.sort_values(['n_vec_index','single_non_vector_field'])
print(df_metadata[cols_highlight])

reproducibility_fractions = []
delta_between_different_indices = []
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

	delta_results = []
	for g, ssdf in res_df.groupby(['search_vectors_batch_id','search_vector_within_batch_id'])['res']:
		delta_results.append(compare_two_hits_dict(ssdf.iloc[0],ssdf.iloc[-1]))
	delta_results = pd.DataFrame(data=delta_results,columns = ['MAE_returned_vector_distances', 'n_vectors_returned_in_both_search', 'disntace_delta_for_vecs_returned_in_both_search'])
	delta_between_different_indices.append(delta_results.n_vectors_returned_in_both_search.value_counts())

reproducibility_fractions = pd.concat(reproducibility_fractions).reset_index().pivot(index='test_uuid',values=True,columns='index')
test_uuids = [x['metadata']['test_uuid'] for x in all_test_results]
delta_between_different_indices_df = pd.DataFrame(delta_between_different_indices)
delta_between_different_indices_df.index = test_uuids
delta_between_different_indices_df.columns = pd.MultiIndex.from_tuples([(delta_between_different_indices_df.columns.name,c) for c in delta_between_different_indices_df])
df_summary_results = df_metadata.set_index('test_uuid').join(reproducibility_fractions)
df_summary_results = df_summary_results.rename(columns={'same_index_search_reproducibility':'fraction_reproducible_search_on_SAME_index',
						 						'different_index_search_reproducibility':'fraction_reproducible_search_on_REBUILT_index'})
df_summary_results.columns = pd.MultiIndex.from_arrays([['']*len(df_summary_results.columns),df_summary_results.columns])
df_summary_results = df_summary_results.join(delta_between_different_indices_df)


cols_highlight = [('','n_vec_index'),
				  ('','n_vec_search'),
				  ('','single_non_vector_field'),
				  ('','results_reproducible'),
				  ('','fraction_reproducible_search_on_SAME_index'),
				  ('','fraction_reproducible_search_on_REBUILT_index'),
				  ('n_vectors_returned_in_both_search', 0),
				('n_vectors_returned_in_both_search', 1),
				('n_vectors_returned_in_both_search', 2),
				('n_vectors_returned_in_both_search', 3)]

cols_highlight = [c for c in cols_highlight if c in df_summary_results.columns]

df_summary_results_display = df_summary_results[cols_highlight]\
								.to_markdown(index=False)


print(df_summary_results_display)
