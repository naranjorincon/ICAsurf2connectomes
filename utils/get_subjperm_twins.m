function [train_list, true_val_list, test_list] = get_subjperm_twins(list1_path, list2_path)
% data is from running "python generate_subindex.py -o ./test_subidx.csv -c
% 1" ONLY ONCE. Need to remove brackers from first and last index for this
% to work.
% needs to be given two paths, traina nd validation taken from above with
% prop 0.8 and 0.5 respectively. I first do train at 80% and then
% validation at 50% which resulted in some overlap - you can check yourself
% here overlap in those two permutations. I basically use those
% permutations that respect family ID and split that into train/val/test.

train_path = list1_path; % './tools/training_subidx.csv'
validation_path = list2_path; % './tools/validation_subidx.csv'
train_list = readmatrix(train_path)';
train_list = sort(train_list + 1); % matlab idx starts at 0 so add 1 to match later indexing
hcp_sbj_count = 1003;
val_list = readmatrix(validation_path)';
val_list = val_list + 1;
diff_idx = ~ismember(val_list,train_list);
true_val_list = sort(val_list(diff_idx == 1));
size_of_subjlist = [1:hcp_sbj_count]'; % num of HCP subjecs

% split into apprx 80/10/10 train/validate/test
% then test is what is left
concat_both_forleftover = sort(cat(1, train_list, true_val_list));
assert(length(unique(concat_both_forleftover)) == (length(train_list) + length(true_val_list)), 'something wrong with train/val sampling go check')
% check non members and those should be left over for test
test_leftover_options = ~ismember(size_of_subjlist,concat_both_forleftover);

% use test_leftover_options as index for left over subidx values
test_list = size_of_subjlist(test_leftover_options == 1);

% last asserts, they should all sum to 1003 and be different
assert(length(train_list)+length(true_val_list)+length(test_list)==hcp_sbj_count,'incorrect total sum, go check!>:(')
assert(sum(ismember(train_list, true_val_list))==0, 'incorrect membership some overlap, go check! >:(')
assert(sum(ismember(train_list, test_list))==0, 'incorrect membership some overlap, go check! >:(')
assert(sum(ismember(true_val_list, test_list))==0, 'incorrect membership some overlap, go check! >:(')

return