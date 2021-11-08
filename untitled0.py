# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:02:03 2020

@author: john
"""

#data
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import log as utils
from pm4py.statistics.start_activities.log.get import get_start_activities
from pm4py.statistics.end_activities.log.get import get_end_activities

#process mining
from pm4py.algo.discovery.alpha import factory as alpha_miner
from pm4py.algo.discovery.heuristics import factory as heuristics_miner
from pm4py.algo.discovery.inductive import factory as inductive_miner

#process discovery
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.petri.check_soundness import check_petri_wfnet_and_soundness
from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.evaluation.generalization import evaluator as generalization_evaluator
from pm4py.evaluation.simplicity import evaluator as simplicity_evaluator
from pm4py.evaluation import factory as evaluation_factory
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.evaluation.replay_fitness import factory as replay_fitness_factory
from pm4py.algo.conformance.alignments import algorithm as alignments
from pm4py.evaluation.replay_fitness import evaluator as replay_fitness
from pm4py.algo.conformance.alignments import factory as align_factory
from pm4py.objects.petri.align_utils import pretty_print_alignments
from pm4py.objects.conversion.dfg import converter as dfg_mining
from pm4py.objects.petri import utils

#decision mining
from pm4py.algo.enhancement.decision import algorithm as decision_mining

#statistics
from pm4py.util import constants
from pm4py.statistics.traces.log import case_statistics
from pm4py.algo.filtering.log.attributes import attributes_filter

#process tree
from pm4py.simulation.tree_generator import simulator as tree_gen

#viz
from pm4py.visualization.heuristics_net import factory as hn_vis_factory
from pm4py.visualization.petrinet import factory as vis_factory
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.visualization.petrinet import visualizer
from pm4py.visualization.graphs import visualizer as graphs_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.decisiontree import visualizer as tree_visualizer






#   διαβάζουμε το event log
log = xes_importer.apply('C:\\Users\\john\\Desktop\\διπλωματικη\\datasets\\Lfull.xes')


#print(log)

trace_key_list = []
event_key_list = []
event_count = 0 #counter για να μετρήσουμε το πλήθος των event
for trace in log:
    #βρίσκουμε τα keys κάθε trace και αν δεν υπάρχουν ήδη στη λίστα με τα key
    #δηλαδή την trace_key_list τα προσθέτουμε στη λίστα. 
    for trace_key in trace.attributes.keys():
        if trace_key not in trace_key_list:
            trace_key_list.append(trace_key)
    for event in trace:
        #κάνουμε το ίδιο και για τα keys των events
        for event_key in event.keys():
            if event_key not in event_key_list:
                event_key_list.append(event_key)
        event_count += 1 #κάθε φορά που μπαίνουμε στην for των events αυξάνουμε τον counter κατά 1
        
#   εμφάνιση του αριθμού των event και traces
print("Number of traces : " + str(len(log)))
print("Number of events : " + str(event_count))

#   εμφάνιση των διαφορετικών event
#unique_events = utils.get_event_labels(log,'concept:name')
#print("Events of log : " + str(unique_events))

#   εμφάνιση των πρώτων δραστηριοτήτων
start_activities = get_start_activities(log)
print("Starting activities: " + str(start_activities))

#   εμφάνιση των τελευταίων δραστηριοτήτων
end_activities = get_end_activities(log)
print("End activities" + str(end_activities))

#   δημιουργία dataframe και φόρτωση του log σε αυτό για καλύτερη ανάγνωση του
log_df = pd.DataFrame(columns = ["Case ID" , "Activity Name" , "Transition" , "Timestamp"])
for trace_id, trace in enumerate(log):
    for event_index, event in enumerate(trace):
        #φτιάχνουμε ένα dataframe στο οποίο ουσιαστικά φορτώνουμε τα στοιχεία
        #που θέλουμε από το τρέχον event, δηλαδή μια γραμμή του πίνακα
        #που σκοπεύουμε να δημιουργήσουμε
        row = pd.DataFrame({
            "Case ID" : [trace.attributes["concept:name"]],
            "Activity Name" : [event["concept:name"]],
            "Transition" : [event["lifecycle:transition"]],
            "Timestamp" : [event["time:timestamp"]]
            })
        #κάνουμε append την γραμμή που φτιάξαμε στο DataFrame που ορίσαμε έξω από την επανάληψη
        log_df = log_df.append(row, ignore_index = True)
        
log_df.to_csv('log_csv.csv', index = False)




'''
#table 1 kef5
variants_count = case_statistics.get_variant_statistics(log)
variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)
print(variants_count)

#table 2 kef5
evntlst=[]
for trace in log:
    for event in trace:
        evntlst.extend(event['concept:name'])
           
str1=""
str1 = ''.join(evntlst)
Sum=0
Sum=sum('b' in s for s in str1)
print(Sum)
'''


#---process discovery

'''
#alpha miner
net, initial_marking, final_marking = alpha_miner.apply(log)
print('Alpha Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz) 


#cycles = utils.get_cycles_petri_net_transitions(net)
#print(cycles)


#   evaluation of alpha miner
print("evaluation of alpha miner")
evaluation_result = evaluation_factory.apply(log, net, initial_marking,final_marking)
print("percentage of fit traces: ",evaluation_result['fitness']['perc_fit_traces'])
print("average trace fitness: ",evaluation_result['fitness']['average_trace_fitness'])
print("log fitness: ",evaluation_result['fitness']['log_fitness'])
print("precision: ",evaluation_result['precision'])
print("generalization: ",evaluation_result['generalization'])
print("simplicity: ",evaluation_result['simplicity'])
print("fscore: ",evaluation_result['fscore'])
print("overall score(metricsAverageWeight): ",evaluation_result['metricsAverageWeight'])

#or ...

print("///////")
fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
print("TOKEN BASED fitness: ", str(fitness))
fitness2 = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
print("ALIGNMENT BASED fitness: ", str(fitness2))
prec = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
print("TOKEN BASED precision: ", str(prec))
prec2 = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
print("alignments based precision: ", str(prec2))
gen = generalization_evaluator.apply(log, net, initial_marking, final_marking)
print("generalization: ", str(gen))
simp = simplicity_evaluator.apply(net)
print("simplicity: ", str(simp))


#   check if the petri net using alpha miner is sound

is_it_sound = check_petri_wfnet_and_soundness(net)
print("alpha miner petri net is sound? ",is_it_sound)

#
#soundness:
#1-For each possible state of the process model, it is possible to reach the end state
#2-When the process model reaches the end state, there are no tokens left behind
#3-Each transition in the process model can be enabled






#   conformance checking of alpha miner

#   replay token
replay_result = token_replay.apply(log, net, initial_marking, final_marking)
#print(replay_result)

fit_traces = 0
for trace in replay_result:
    if trace["trace_is_fit"]:
        fit_traces += 1
print("Number of  fit traces : ", fit_traces)

log_fitness = replay_fitness_factory.evaluate(replay_result, variant="token_replay")
print('log fitness', log_fitness)


#   alignments
alignments = align_factory.apply_log(log, net, initial_marking, final_marking)
#print(alignments)
#pretty_print_alignments(alignments)

log_fitness2 = replay_fitness_factory.evaluate(alignments, variant="alignments")
print('log fitness ',log_fitness2)
'''



'''
#   heuristics miner
net, initial_marking, final_marking = heuristics_miner.apply(log)
print('Heuristics Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz) 



cycles = utils.get_cycles_petri_net_transitions(net)
print(cycles)


#   evaluation of heuristic miner
print("evaluation of heuristic miner")
evaluation_result = evaluation_factory.apply(log, net, initial_marking,final_marking)
print("percentage of fit traces: ",evaluation_result['fitness']['perc_fit_traces'])
print("average trace fitness: ",evaluation_result['fitness']['average_trace_fitness'])
print("log fitness: ",evaluation_result['fitness']['log_fitness'])
print("precision: ",evaluation_result['precision'])
print("generalization: ",evaluation_result['generalization'])
print("simplicity: ",evaluation_result['simplicity'])
print("fscore: ",evaluation_result['fscore'])
print("overall score(metricsAverageWeight): ",evaluation_result['metricsAverageWeight'])


print("///////")
fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
print("TOKEN BASED fitness: ", str(fitness))
fitness2 = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
print("ALIGNMENT BASED fitness: ", str(fitness2))
prec = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
print("TOKEN BASED precision: ", str(prec))
prec2 = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
print("alignments based precision: ", str(prec2))
gen = generalization_evaluator.apply(log, net, initial_marking, final_marking)
print("generalization: ", str(gen))
simp = simplicity_evaluator.apply(net)
print("simplicity: ", str(simp))



#   check if the petri net using heuristic miner is sound

is_it_sound = check_petri_wfnet_and_soundness(net)
print("heuristic miner petri net is sound? ",is_it_sound)


#   conformance checking of heuristic miner

#   replay token
replay_result = token_replay.apply(log, net, initial_marking, final_marking)
#print(replay_result)

fit_traces = 0
for trace in replay_result:
    if trace["trace_is_fit"]:
        fit_traces += 1
print("Number of  fit traces : ", fit_traces)


log_fitness = replay_fitness_factory.evaluate(replay_result, variant="token_replay")
print('log fitness', log_fitness)


#   alignments
alignments = align_factory.apply_log(log, net, initial_marking, final_marking)
#print(alignments)
#pretty_print_alignments(alignments)

log_fitness2 = replay_fitness_factory.evaluate(alignments, variant="alignments")
print('log fitness ',log_fitness2)

'''



'''

#   inductive miner
net, initial_marking, final_marking = inductive_miner.apply(log)
print('Inductive Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz)

cycles = utils.get_cycles_petri_net_transitions(net)
print(cycles)


#   evaluation of inductive miner
print("evaluation of inductive miner")
evaluation_result = evaluation_factory.apply(log, net, initial_marking,final_marking)
print("percentage of fit traces: ",evaluation_result['fitness']['perc_fit_traces'])
print("average trace fitness: ",evaluation_result['fitness']['average_trace_fitness'])
print("log fitness: ",evaluation_result['fitness']['log_fitness'])
print("precision: ",evaluation_result['precision'])
print("generalization: ",evaluation_result['generalization'])
print("simplicity: ",evaluation_result['simplicity'])
print("fscore: ",evaluation_result['fscore'])
print("overall score(metricsAverageWeight): ",evaluation_result['metricsAverageWeight'])



print("///////")
fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
print("TOKEN BASED fitness: ", str(fitness))
fitness2 = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
print("ALIGNMENT BASED fitness: ", str(fitness2))
prec = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
print("TOKEN BASED precision: ", str(prec))
prec2 = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
print("alignments based precision: ", str(prec2))
gen = generalization_evaluator.apply(log, net, initial_marking, final_marking)
print("generalization: ", str(gen))
simp = simplicity_evaluator.apply(net)
print("simplicity: ", str(simp))



#   check if the petri net using inductive miner is sound

is_it_sound = check_petri_wfnet_and_soundness(net)
print("inductive miner petri net is sound? ",is_it_sound)

#   conformance checking of inductive miner

#   replay token
replay_result = token_replay.apply(log, net, initial_marking, final_marking)
#print(replay_result)

fit_traces = 0
for trace in replay_result:
    if trace["trace_is_fit"]:
        fit_traces += 1
print("Number of  fit traces : ", fit_traces)


log_fitness = replay_fitness_factory.evaluate(replay_result, variant="token_replay")
print('log fitness', log_fitness)


#   alignments
alignments = align_factory.apply_log(log, net, initial_marking, final_marking)
#print(alignments)
#pretty_print_alignments(alignments)

log_fitness2 = replay_fitness_factory.evaluate(alignments, variant="alignments")
print('log fitness ',log_fitness2)

'''


#   directly follows graphs

#   Directly-Follows graph decorated with the performance between the edges
dfg = dfg_discovery.apply(log, variant=dfg_discovery.Variants.PERFORMANCE)
gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.PERFORMANCE)
dfg_visualization.view(gviz)

#    Directly-Follows graph decorated with the frequency of activities
dfg = dfg_discovery.apply(log)
gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY)
dfg_visualization.view(gviz)


net, initial_marking, final_marking = dfg_mining.apply(dfg)

gviz = dfg_visualization.apply(net, initial_marking, final_marking, variant=dfg_visualization.Variants.FREQUENCY)
dfg_visualization.view(gviz)

cycles = utils.get_cycles_petri_net_transitions(net)
print(cycles)


#   replay token
replay_result = token_replay.apply(log, net, initial_marking, final_marking)
#print(replay_result)

fit_traces = 0
for trace in replay_result:
    if trace["trace_is_fit"]:
        fit_traces += 1
print("Number of  fit traces : ", fit_traces)


log_fitness = replay_fitness_factory.evaluate(replay_result, variant="token_replay")
print('log fitness', log_fitness)


#   alignments
alignments = align_factory.apply_log(log, net, initial_marking, final_marking)
#print(alignments)
#pretty_print_alignments(alignments)

log_fitness2 = replay_fitness_factory.evaluate(alignments, variant="alignments")
print('log fitness2 ',log_fitness2)


'''
#   evaluation of dfg 
print("evaluation of dfg")
evaluation_result = evaluation_factory.apply(log, net, initial_marking,final_marking)
print("percentage of fit traces: ",evaluation_result['fitness']['perc_fit_traces'])
print("average trace fitness: ",evaluation_result['fitness']['average_trace_fitness'])
print("log fitness: ",evaluation_result['fitness']['log_fitness'])
print("precision: ",evaluation_result['precision'])
print("generalization: ",evaluation_result['generalization'])
print("simplicity: ",evaluation_result['simplicity'])
print("fscore: ",evaluation_result['fscore'])
print("overall score(metricsAverageWeight): ",evaluation_result['metricsAverageWeight'])


print("///////")
fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
print("TOKEN BASED fitness: ", str(fitness))
fitness2 = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
print("ALIGNMENT BASED fitness: ", str(fitness2))
prec = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
print("TOKEN BASED precision: ", str(prec))
prec2 = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
print("alignments based precision: ", str(prec2))
gen = generalization_evaluator.apply(log, net, initial_marking, final_marking)
print("generalization: ", str(gen))
simp = simplicity_evaluator.apply(net)
print("simplicity: ", str(simp))
'''


'''
#   Convert Directly-Follows Graph to a Workflow Net

dfg = dfg_discovery.apply(log)
net, initial_marking, final_marking = dfg_mining.apply(dfg)
print('Convert Directly-Follows Graph to a Workflow Net\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz) 




# decision mining

net, initial_marking, final_marking = inductive_miner.apply(log)
gviz = visualizer.apply(net, initial_marking, final_marking, parameters={visualizer.Variants.WO_DECORATION.value.Parameters.DEBUG: True})
visualizer.view(gviz)

X, y, class_names = decision_mining.apply(log, net, initial_marking, final_marking, decision_point="p_10")

clf, feature_names, classes = decision_mining.get_decision_tree(log, net, initial_marking, final_marking, decision_point="p_10")
gviz = tree_visualizer.apply(clf, feature_names, classes)
visualizer.view(gviz)


#   statistics

#   throughput time
all_case_durations = case_statistics.get_all_casedurations(log, parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
#print(all_case_durations)

median_case_duration = case_statistics.get_median_caseduration(log, parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
print("median value: ",median_case_duration)  

Sum = sum(all_case_durations)
avg_time = Sum/len(all_case_durations)
print("average time of case: ",avg_time)

print("/////")



#   displaying graphs 

#   distribution of case duration
x, y = case_statistics.get_kde_caseduration(log, parameters={constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"})
gviz = graphs_visualizer.apply_plot(x, y, variant=graphs_visualizer.Variants.CASES)
graphs_visualizer.view(gviz)


#   distribution of events over time
x, y = attributes_filter.get_kde_date_attribute(log, attribute="time:timestamp")
gviz = graphs_visualizer.apply_plot(x, y, variant=graphs_visualizer.Variants.DATES)
graphs_visualizer.view(gviz)



# process tree

parameters = {}
tree = tree_gen.apply(parameters=parameters)
print(tree)
gviz = pt_visualizer.apply(tree, parameters={pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
pt_visualizer.view(gviz)
'''



'''
'''
'''
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------


############################
# pairnw to event log kai bgazw apo auto ta traces pou periexoun to event 'f' gia na bgalw ton kyklo apo to event log
# etsi ta sximata pou tha pairnw einai acyclic graphs
#isws etsi opws ebgala ta traces na xreiastei na kanw to idio gia na dw ta traces poy akolouthoun tin sigegrikeni diadromi
#px a->d>c>e>g ktlp gia kathe ena tropo diadromis???????
#  http://pm4py.pads.rwth-aachen.de/documentation/process-discovery/frequency-performance/
###########################

#   διαβάζουμε το event log
log = xes_importer.apply('C:\\Users\\john\\Desktop\\διπλωματικη\\datasets\\Lfull.xes')
#φιλτραρισμένο event log χωρίς κύκλους
tracefilter_log_pos = attributes_filter.apply(log, ["f"], parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})

activities = attributes_filter.get_attribute_values(tracefilter_log_pos, "concept:name")
#print(activities)



trace_key_list = []
event_key_list = []
event_count = 0 #counter για να μετρήσουμε το πλήθος των event
for trace in tracefilter_log_pos:
    #βρίσκουμε τα keys κάθε trace και αν δεν υπάρχουν ήδη στη λίστα με τα key
    #δηλαδή την trace_key_list τα προσθέτουμε στη λίστα. 
    for trace_key in trace.attributes.keys():
        if trace_key not in trace_key_list:
            trace_key_list.append(trace_key)
    for event in trace:
        #κάνουμε το ίδιο και για τα keys των events
        for event_key in event.keys():
            if event_key not in event_key_list:
                event_key_list.append(event_key)
        event_count += 1 #κάθε φορά που μπαίνουμε στην for των events αυξάνουμε τον counter κατά 1
        
#   εμφάνιση του αριθμού των event και traces
print("Number of traces : " + str(len(tracefilter_log_pos)))
print("Number of events : " + str(event_count))

#   εμφάνιση των διαφορετικών event
#unique_events = utils.get_event_labels(tracefilter_log_pos,'concept:name')
#print("Events of log : " + str(unique_events))

'''
'''


#alpha miner
net, initial_marking, final_marking = alpha_miner.apply(tracefilter_log_pos)
print('Alpha Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz) 
'''
'''
from pm4py.objects.petri import utils


#   heuristics miner
net, initial_marking, final_marking = heuristics_miner.apply(tracefilter_log_pos)
print('Heuristics Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz) 

cycles = utils.get_cycles_petri_net_places(net)
print(cycles)


#   inductive miner
net, initial_marking, final_marking = inductive_miner.apply(tracefilter_log_pos)
print('Inductive Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz)

cycles = utils.get_cycles_petri_net_places(net)
print(cycles)


from pm4py.algo.discovery.dfg import factory as dfg_factory
from pm4py.visualization.dfg import factory as dfg_vis_factory


#    Directly-Follows graph decorated with the frequency of activities
dfg = dfg_factory.apply(tracefilter_log_pos)
gviz = dfg_vis_factory.apply(dfg, log=tracefilter_log_pos, variant="frequency")
dfg_vis_factory.view(gviz)



net, initial_marking, final_marking = inductive_miner.apply(tracefilter_log_pos)
gviz = visualizer.apply(net, initial_marking, final_marking, parameters={visualizer.Variants.WO_DECORATION.value.Parameters.DEBUG: True})
visualizer.view(gviz)
'''
#decision mining

'''
print(decision_mining.get_decision_points(net, labels=True, pre_decision_points=None, parameters=None))


X, y, class_names = decision_mining.apply(log, net, initial_marking, final_marking, decision_point="p_9")
clf, feature_names, classes = decision_mining.get_decision_tree(log, net, initial_marking, final_marking, decision_point="p_9")
gviz = tree_visualizer.apply(clf, feature_names, classes)
visualizer.view(gviz)

X, y, class_names = decision_mining.apply(log, net, initial_marking, final_marking, decision_point="p_7")
clf, feature_names, classes = decision_mining.get_decision_tree(log, net, initial_marking, final_marking, decision_point="p_7")
gviz = tree_visualizer.apply(clf, feature_names, classes)
visualizer.view(gviz)


import networkx as nx
from matplotlib import pyplot as plt

graph = nx.DiGraph()

#graph.add_node("a")
#graph.add_node("b")
#graph.add_node("c")
#graph.add_node("d")
#graph.add_node("e")
#graph.add_node("g")
#graph.add_node("h")

graph.add_edges_from([("a","c"),("a","d"),("a","b"),("c","d"),("c","e"),("d","c"),("d","e"),("d","b"),("b","d"),("b","e"),("e","g"),("e","h")])
print("is the graph directed? ",nx.is_directed(graph))
print("is the graph directed acyclic?" ,nx.is_directed_acyclic_graph(graph))

nx.draw_networkx(graph, arrows=True)

'''

####################################    GIA NA VRW TO 80-20 ( PROETIMASIES )       ########################################################################


'''
#80% of filtered event log for bayesian network construction
new_lenght_of_log=len(tracefilter_log_pos)*0.8
new_lenght_of_log2 = int(new_lenght_of_log)
print(new_lenght_of_log2)



log_df_80 = pd.DataFrame(columns = ["Case ID" , "Activity Name" , "Transition" , "Timestamp"])
log_df_20 = pd.DataFrame(columns = ["Case ID" , "Activity Name" , "Transition" , "Timestamp"])
for trace_id, trace in enumerate(tracefilter_log_pos):
    for event_index, event in enumerate(trace):
        row = pd.DataFrame({
            "Case ID" : [trace.attributes["concept:name"]],
            "Activity Name" : [event["concept:name"]],
            "Transition" : [event["lifecycle:transition"]],
            "Timestamp" : [event["time:timestamp"]]
            })
        log_df_80 = log_df_80.append(row, ignore_index = True)
        if log_df_80['Case ID'].nunique() > new_lenght_of_log2:
            log_df_20 = log_df_20.append(row, ignore_index = True)
     
cond = log_df_80['Case ID'].isin(log_df_20['Case ID'])
log_df_80.drop(log_df_80[cond].index, inplace = True)         
     
#log_df_80.to_csv('log_csv_80%.csv', index = False)
#log_df_20.to_csv('log_csv_20%.csv', index = False)



print("unique sto pandas 80% " ,log_df_80['Case ID'].nunique() )
print("unique sto pandas 20% " ,log_df_20['Case ID'].nunique() )


#total traces = 1254
#80% of 1254 is 1003 
'''

#########################################################################################
################################################################################################
#########################################################################################




'''
from pm4py.objects.conversion.log import factory as conversion_factory
from pm4py.objects.log.exporter.xes import factory as xes_exporter

## ta dataframe apo panw poy eginan csv ta kanoyme .xes

log_csv_80 = pd.read_csv('C:\\Users\\john\\Desktop\\διπλωματικη\\code\\log_csv_80%.csv', sep=',')

#log_csv_80.rename(columns={'Case ID': 'case:concept:name'}, inplace=True)
#log_csv_80.rename(columns={'Activity Name': 'concept:name'}, inplace=True)
#log_csv_80.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
#log_csv_80.rename(columns={'Transition': 'lifecycle:transition'}, inplace=True)

parameters = {constants.PARAMETER_CONSTANT_CASEID_KEY: "Case ID",
              constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "Activity Name",
              constants.PARAMETER_CONSTANT_TRANSITION_KEY: "Transition",
              constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "Timestamp"}

log80 = conversion_factory.apply(log_df_80,parameters=parameters)
xes_exporter.export_log(log80, "log80%.xes")

'''''''
log_csv_20 = pd.read_csv('C:\\Users\\john\\Desktop\\διπλωματικη\\code\\log_csv_20%.csv', sep=',')
parameters = {constants.PARAMETER_CONSTANT_CASEID_KEY: "Case ID",
              constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "Activity Name",
              constants.PARAMETER_CONSTANT_TRANSITION_KEY: "Transition",
              constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "Timestamp"}
log20 = conversion_factory.apply(log_df_20, parameters=parameters)
xes_exporter.export_log(log20, "log20%.xes")
'''
############










#########################################################################################
################################################################################################
#########################################################################################






















'''

##################################  douleia me to 80% liga pragmata      ###########################################################################

log_for_80 = xes_importer.apply('C:\\Users\\john\\Desktop\\διπλωματικη\\code\\log80%.xes')

trace_key_list = []
event_key_list = []
event_count = 0 #counter για να μετρήσουμε το πλήθος των event
for trace in log_for_80:
    #βρίσκουμε τα keys κάθε trace και αν δεν υπάρχουν ήδη στη λίστα με τα key
    #δηλαδή την trace_key_list τα προσθέτουμε στη λίστα. 
    for trace_key in trace.attributes.keys():
        if trace_key not in trace_key_list:
            trace_key_list.append(trace_key)
    for event in trace:
        #κάνουμε το ίδιο και για τα keys των events
        for event_key in event.keys():
            if event_key not in event_key_list:
                event_key_list.append(event_key)
        event_count += 1 #κάθε φορά που μπαίνουμε στην for των events αυξάνουμε τον counter κατά 1
        
#   εμφάνιση του αριθμού των event και traces
print("Number of traces : " + str(len(log_for_80)))
print("Number of events : " + str(event_count))

#   εμφάνιση των διαφορετικών event
unique_events = utils.get_event_labels(log_for_80,'Activity Name')
print("Events of log : " + str(unique_events))


evntlst=[]
for trace in log_for_80:
    for event in trace:
        evntlst.extend(event['Activity Name'])
        
print("length of evntlst: ", len(evntlst))    
str1=""
str1 = ''.join(evntlst)
print("length of str1: ",len(str1))
#print(",".join([str1[i:i+5] for i in range(0, len(str1), 5)]))
str1 = ",".join([str1[i:i+5] for i in range(0, len(str1), 5)])
#print(str1)
#print("length of str1: ", len(str1))

#print(str1.count(",") +1)


str1_list = []
str1_list = str1.split(",")
#print(str1_list)


uniq = set(str1_list)
#word = 'adceg'
for item in uniq:
    print(item)
    print(str1_list.count(item))

substringAC = "ac"
substringAB = "ab"
substringAD = "ad"
substringCD = "cd"
substringCE = "ce"
substringDC = "cd"
substringDB = "db"

Sum=0
Sum=sum('ac' in s for s in str1_list)

print(Sum)

'''

'''

#alpha miner
net, initial_marking, final_marking = alpha_miner.apply(log_for_80, parameters = parameters)
print('Alpha Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz) 


from pm4py.objects.petri import utils

cycles = utils.get_cycles_petri_net_places(net)
print(cycles)



#   heuristics miner
net, initial_marking, final_marking = heuristics_miner.apply(log_for_80, parameters = parameters)
print('Heuristics Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz) 

cycles = utils.get_cycles_petri_net_places(net)
print(cycles)


#   inductive miner
net, initial_marking, final_marking = inductive_miner.apply(log_for_80, parameters = parameters)
print('Inductive Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz)

cycles = utils.get_cycles_petri_net_places(net)
print(cycles)


from pm4py.algo.discovery.dfg import factory as dfg_factory
from pm4py.visualization.dfg import factory as dfg_vis_factory


#    Directly-Follows graph decorated with the frequency of activities
dfg = dfg_factory.apply(log_for_80, parameters = parameters)
gviz = dfg_vis_factory.apply(dfg, log=log_for_80, variant="frequency", parameters=parameters)
dfg_vis_factory.view(gviz)


''''''

log_for_20 = xes_importer.apply('C:\\Users\\john\\Desktop\\διπλωματικη\\code\\log20%.xes')

trace_key_list = []
event_key_list = []
event_count = 0 #counter για να μετρήσουμε το πλήθος των event
for trace in log_for_20:
    #βρίσκουμε τα keys κάθε trace και αν δεν υπάρχουν ήδη στη λίστα με τα key
    #δηλαδή την trace_key_list τα προσθέτουμε στη λίστα. 
    for trace_key in trace.attributes.keys():
        if trace_key not in trace_key_list:
            trace_key_list.append(trace_key)
    for event in trace:
        #κάνουμε το ίδιο και για τα keys των events
        for event_key in event.keys():
            if event_key not in event_key_list:
                event_key_list.append(event_key)
        event_count += 1 #κάθε φορά που μπαίνουμε στην for των events αυξάνουμε τον counter κατά 1
        
#   εμφάνιση του αριθμού των event και traces
print("Number of traces : " + str(len(log_for_20)))
print("Number of events : " + str(event_count))

#   εμφάνιση των διαφορετικών event
unique_events = utils.get_event_labels(log_for_20,'Activity Name')
print("Events of log : " + str(unique_events))


evntlst=[]
for trace in log_for_20:
    for event in trace:
        evntlst.extend(event['Activity Name'])
        
print("length of evntlst: ", len(evntlst))    
str1=""
str1 = ''.join(evntlst)
print("length of str1: ",len(str1))
#print(",".join([str1[i:i+5] for i in range(0, len(str1), 5)]))
str1 = ",".join([str1[i:i+5] for i in range(0, len(str1), 5)])
#print(str1)
#print("length of str1: ", len(str1))

#print(str1.count(",") +1)


str1_list = []
str1_list = str1.split(",")
#print(str1_list)


uniq = set(str1_list)
#word = 'adceg'
for item in uniq:
    print(item)
    print(str1_list.count(item))

substringAC = "ac"
substringAB = "ab"
substringAD = "ad"
substringCD = "cd"
substringCE = "ce"
substringDC = "cd"
substringDB = "db"

Sum=0
Sum=sum('ac' in s for s in str1_list)

print(Sum)
'''

'''

#alpha miner
net, initial_marking, final_marking = alpha_miner.apply(log_for_20, parameters = parameters)
print('Alpha Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz) 


from pm4py.objects.petri import utils

cycles = utils.get_cycles_petri_net_places(net)
print(cycles)



#   heuristics miner
net, initial_marking, final_marking = heuristics_miner.apply(log_for_20, parameters = parameters)
print('Heuristics Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz) 

cycles = utils.get_cycles_petri_net_places(net)
print(cycles)


#   inductive miner
net, initial_marking, final_marking = inductive_miner.apply(log_for_20, parameters = parameters)
print('Inductive Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz)

cycles = utils.get_cycles_petri_net_places(net)
print(cycles)


from pm4py.algo.discovery.dfg import factory as dfg_factory
from pm4py.visualization.dfg import factory as dfg_vis_factory


#    Directly-Follows graph decorated with the frequency of activities
dfg = dfg_factory.apply(log_for_20, parameters = parameters)
gviz = dfg_vis_factory.apply(dfg, log=log_for_20, variant="frequency", parameters=parameters)
dfg_vis_factory.view(gviz)





'''









'''

import networkx as nx
from matplotlib import pyplot as plt

graph = nx.DiGraph()

#graph.add_node("a")
#graph.add_node("b")
#graph.add_node("c")
#graph.add_node("d")
#graph.add_node("e")
#graph.add_node("g")
#graph.add_node("h")

graph.add_edges_from([("a","c"),("a","d"),("a","b"),("c","d"),("c","e"),("d","c"),("d","e"),("d","b"),("b","d"),("b","e"),("e","g"),("e","h")])
print("is the graph directed? ",nx.is_directed(graph))
print("is the graph directed acyclic?" ,nx.is_directed_acyclic_graph(graph))

nx.draw_networkx(graph, arrows=True)

'''