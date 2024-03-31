# analyze data
import os
import glob
import pickle
import pandas as pd
import numpy as np
import time 
import itertools
# load latests file

files = glob.glob('all_test_results*.pickle')
latest_file = max(files, key=os.path.getctime)
run_uuid = latest_file.split('uuid_')[-1].split('.pickle')[0]

print(f'Analysing results from run: {run_uuid} from  {time.ctime(os.path.getctime(latest_file))}')

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

	output = dict(zip(
			['MAE_returned_vector_distances', 'n_vectors_returned_in_both_search', 'distance_delta_for_vecs_returned_in_both_search'],
			 [ordered_keys_MAE, n_intersecting_keys, intersecting_keys_value_deltas]
			 ))
	if (n_intersecting_keys < len(hits0.keys())) or (n_intersecting_keys < len(hits1.keys())):
		output.update({'hits0':str(hits0),
				 		'hits1':str(hits1)})
	return output



df_metadata = pd.DataFrame([pd.Series(x['metadata']) for x in all_test_results])
# cols_highlight = df.columns[df.apply(lambda x: len(x.apply(str).unique()),axis=0)>1]
# cols_highlight = [c for c in cols_highlight if 'uuid' not in c]
cols_highlight = ['n_vec_index','n_vec_search','results_reproducible']
df_metadata = df_metadata.sort_values(['n_vec_index'])
print(df_metadata[cols_highlight])

reproducibility_fractions = []

detailed_results_comparison_rebuilt_index = []
detailed_results_comparison_same_index = []

delta_between_different_indices = []
delta_different_indices_same_embeddings_distance_difference = []
delta_MAE_returned_vector_distances = []

for i, test_res in enumerate(all_test_results):
	res_df = pd.DataFrame(test_res['results'])

	# Check complete Reproducibility for same and rebuilt (different) index
	## same_idx reproducibility: repeated queries on the same index (without rebuilding) for all the rebuilds. 
	same_idx = res_df.groupby(['index_rebuild_id','search_vectors_batch_id','search_vector_within_batch_id'])['res']\
        .apply(lambda x: all_list_elements_identical(x.to_list()))\
     .value_counts(normalize=True)
	same_idx.name = 'same_index_search_reproducibility'

	## different_idx reproducibilit for rebuilt (with identical vectors and settings) for all of repeat searches
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

	# check detailed comparisons for same index and rebuilt index
	## same index
	same_idx_detailed = []
	group_cols = ['index_rebuild_id','search_vectors_batch_id','search_vector_within_batch_id']
	for group_vals, ssdf in res_df.groupby(group_cols)['res']:
		for (a,b) in itertools.combinations(range(ssdf.shape[0]),2):
			r = compare_two_hits_dict(ssdf.iloc[a],ssdf.iloc[b])
			r.update(dict(zip(group_cols,group_vals)))
			same_idx_detailed.append(r)

	different_idx_detailed = []
	group_cols = ['same_index_search_repeat_id','search_vectors_batch_id','search_vector_within_batch_id']
	for group_vals, ssdf in res_df.groupby(group_cols)['res']:
		for (a,b) in itertools.combinations(range(ssdf.shape[0]),2):
			r = compare_two_hits_dict(ssdf.iloc[a],ssdf.iloc[b])
			r.update(dict(zip(group_cols,group_vals)))
			different_idx_detailed.append(r)

	
	detailed_results_comparison_same_index.append(pd.DataFrame(same_idx_detailed))
	detailed_results_comparison_rebuilt_index.append(pd.DataFrame(different_idx_detailed))


reproducibility_fractions = pd.concat(reproducibility_fractions).reset_index().pivot(index='test_uuid',values=True,columns='index')
test_uuids = [x['metadata']['test_uuid'] for x in all_test_results]

for test_id, res in zip(test_uuids,detailed_results_comparison_same_index):
	try:
		res_stats_df = pd.DataFrame(
										data = res.MAE_returned_vector_distances\
												.describe().loc[['min','mean','max']]\
												.rename(lambda x: 'Repeat_search_on_SAME_index\nMAE_distances_'+x)\
												.rename(test_id)
												).T
		res_stats_df = res_stats_df.join(
										pd.DataFrame(
										data = res.n_vectors_returned_in_both_search\
												.value_counts()\
												.rename(lambda x: 'Repeat_search_on_SAME_index\nnum_vecs_in_topK_in_both_search_-_'+str(x))\
												.rename(test_id)
												).T

										)
		col_overlap = sorted(list(set(reproducibility_fractions.columns.to_list()).intersection(res_stats_df.columns.to_list())))
		col_diff = sorted(list(set(res_stats_df.columns.to_list()).difference(reproducibility_fractions.columns.to_list())))
		if len(col_overlap)>0:
			reproducibility_fractions.update(res_stats_df[col_overlap])
		if len(col_diff)>0:
			reproducibility_fractions = reproducibility_fractions.join(res_stats_df[col_diff])

	except:
		import pdb
		pdb.set_trace()


for test_id, res in zip(test_uuids,detailed_results_comparison_rebuilt_index):
	try:
		res_stats_df = pd.DataFrame(
										data = res.MAE_returned_vector_distances\
												.describe().loc[['min','mean','max']]\
												.rename(lambda x: 'Repeat_search_on_REBUILT_indices\nMAE_distances_'+x)\
												.rename(test_id)
												).T
		res_stats_df = res_stats_df.join(
										pd.DataFrame(
										data = res.n_vectors_returned_in_both_search\
												.value_counts()\
												.rename(lambda x: 'Repeat_search_on_REBUILT_indices\nnum_vecs_in_topK_in_both_search_-_'+str(x))\
												.rename(test_id)
												).T

										)
		col_overlap = sorted(list(set(reproducibility_fractions.columns.to_list()).intersection(res_stats_df.columns.to_list())))
		col_diff = sorted(list(set(res_stats_df.columns.to_list()).difference(reproducibility_fractions.columns.to_list())))
		if len(col_overlap)>0:
			reproducibility_fractions.update(res_stats_df[col_overlap])
		if len(col_diff)>0:
			reproducibility_fractions = reproducibility_fractions.join(res_stats_df[col_diff])

	except:
		import pdb
		pdb.set_trace()
	
df_summary_results = df_metadata.set_index('test_uuid').join(reproducibility_fractions)
df_summary_results = df_summary_results.rename(columns={'same_index_search_reproducibility':'fraction_reproducible_search_on_SAME_index',
						 						'different_index_search_reproducibility':'fraction_reproducible_search_on_REBUILT_index'})

cols_highlight = [c for c in df_summary_results.columns if 'uuid' not in c]

if df_summary_results.shape[0]<5:
	df_summary_results_display = df_summary_results[cols_highlight].T\
								.to_markdown(index=True)
else:
	df_summary_results_display = df_summary_results[cols_highlight].rename(columns={k:k.replace('\n',':') for k in cols_highlight})\
								.to_markdown(index=False)
print(df_summary_results_display)
with open(f'analysis_summary_run_uuid_{run_uuid}.md','w') as f:
	f.write(df_summary_results_display)
