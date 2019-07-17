"""
Main file to run to get all experiment results.
Result table will be put into latex_output/table.txt.
"""


from functionality import make_summary, \
                          make_smallest_connected_graph, \
                          make_stats_for_prediction, \
                          validation_for_dataset, \
                          final_table

directories = ['france', 'germany']

print("Start computing all stats that are needed for classifcation")

for d in directories:
    print("Start making stats for " + d)
    print("Make summary")
    make_summary(d)
    print("Make graph")
    make_smallest_connected_graph(d)
    print("Now compute all stats")
    make_stats_for_prediction(d)
    print("Now make classifcation")
    validation_for_dataset(d)

print("Finished experiments")
print("Now compute and write result table as latex tabular")
final_table()
