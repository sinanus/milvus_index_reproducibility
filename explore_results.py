# analyze data

import glob
import pickle
import pandas as pd
import numpy as np

files = glob.glob('non_reproducibile_index_test_results_2024-03-08*.pickle')
all_test_results = [pickle.load(open(f,'rb')) for f in files]


def all_list_elements_identical(l:list)->bool:
	l_str = [str(x) for x in l]
	if len(set(l_str))>1:
		return False
	return True

N_examples = 3
for test_case in all_test_results:

		# check we have all repetitions
		assert test_case['metadata']['number_index_rebuild'] == len(test_case['results'])

		rep0 = test_case['results'][0]
		N_index = 0
		N_search = 0
		print('\n')
		print('Test conditions')
		for k,v in test_case['metadata'].items():
			print(f'{k}:{v}')

		if test_case['metadata']['all_results_same']:
			print(f'First {N_examples} examples where ALL SEARCH results are REPRODUCIBLE')
		else:
			print(f'First {N_examples} examples where INDEX results are NOT REPRODUCIBLE')
		all_same_index_search_repeats_are_reproducible = True
		for i_group, search_group in enumerate(rep0):
			for i_vector in range(len(search_group[0])):
				rebuild_index_repeated_vector_search_results = []
				same_index_search_repeats_reproducible = True
				for i_index_rep in range(test_case['metadata']['number_index_rebuild']):
					same_index_search_repeat_res = [x_search_rep[i_vector] for x_search_rep in test_case['results'][i_index_rep][i_group]]
					same_index_search_repeat_res_no_instance_id = [{k:v for k,v in x.items() if 'instance_id' not in k}for x in same_index_search_repeat_res]
					if not all_list_elements_identical(same_index_search_repeat_res_no_instance_id):
						same_index_search_repeats_reproducible = False
						all_same_index_search_repeats_are_reproducible = False
					rebuild_index_repeated_vector_search_results.append(same_index_search_repeat_res)
				different_index_search_repeats_reproducible = all_list_elements_identical(rebuild_index_repeated_vector_search_results)

				if (N_search<N_examples and not(same_index_search_repeats_reproducible)):
					print('#'*5 + f' Example {N_search} ' + '#'*5)
					print(f'different_index_search_repeats_reproducible: {different_index_search_repeats_reproducible}  ;  same_index_search_repeats_reproducible: {same_index_search_repeats_reproducible}')
					for r in rebuild_index_repeated_vector_search_results:
						for rr in r:
							print(rr)
					N_search +=1
				elif (N_index<N_examples) and (not(different_index_search_repeats_reproducible) or test_case['metadata']['all_results_same']):
					print('#'*5 + f' Example {N_index} ' + '#'*5)
					print(f'different_index_search_repeats_reproducible: {different_index_search_repeats_reproducible}  ;  same_index_search_repeats_reproducible: {same_index_search_repeats_reproducible}')
					for r in rebuild_index_repeated_vector_search_results:
						for rr in r:
							print(rr)
					N_index +=1
				else:
					break
			else:
				continue  
			break
		if all_same_index_search_repeats_are_reproducible:
			print('All vector searchs on identical index are reproducible for this test')

			



# display N_delta_examples of cases where search results are different

# N = 0
# for i,row in data[~data.all_results_same].iterrows():
# 	# check we have all repetitions
# 	assert row.number_of_repeats == len(row.results)

# 	rep0 = row.results[0]
# 	for i_group, search_group in enumerate(rep0):
# 		for i_vector in range(len(search_group)):
# 			repeated_vector_search_results = []
# 			for i_rep in range(1,row.number_of_repeats):
# 				repeated_vector_search_results.append(row.results[i_rep][i_group][i_vector])
# 			repeated_vector_search_results_str = [str(x) for x in repeated_vector_search_results]

# 			if np.unique(repeated_vector_search_results_str).shape[0]>1:
# 				if N<N_examples:
# 					print('#'*5 + f' Example {N} ' + '#'*5)
# 					for r in repeated_vector_search_results:
# 						print(r)
# 					N +=1