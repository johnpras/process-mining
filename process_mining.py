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


log = xes_importer.apply(' ')

trace_key_list = []
event_key_list = []
event_count = 0 
for trace in log:
    for trace_key in trace.attributes.keys():
        if trace_key not in trace_key_list:
            trace_key_list.append(trace_key)
    for event in trace:
        for event_key in event.keys():
            if event_key not in event_key_list:
                event_key_list.append(event_key)
        event_count += 1
        
print("Number of traces : " + str(len(log)))
print("Number of events : " + str(event_count))

unique_events = utils.get_event_labels(log,'concept:name')
print("Events of log : " + str(unique_events))

start_activities = get_start_activities(log)
print("Starting activities: " + str(start_activities))
end_activities = get_end_activities(log)
print("End activities" + str(end_activities))

log_df = pd.DataFrame(columns = ["Case ID" , "Activity Name" , "Transition" , "Timestamp"])
for trace_id, trace in enumerate(log):
    for event_index, event in enumerate(trace):
        row = pd.DataFrame({
            "Case ID" : [trace.attributes["concept:name"]],
            "Activity Name" : [event["concept:name"]],
            "Transition" : [event["lifecycle:transition"]],
            "Timestamp" : [event["time:timestamp"]]
            })
        log_df = log_df.append(row, ignore_index = True) 
log_df.to_csv('log_csv.csv', index = False)


#---process discovery

#alpha miner
net, initial_marking, final_marking = alpha_miner.apply(log)
print('Alpha Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz) 

#cycles = utils.get_cycles_petri_net_transitions(net)
#print(cycles)

#---evaluation of alpha miner

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

#or
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


#is_it_sound = check_petri_wfnet_and_soundness(net)
#print("alpha miner petri net is sound? ",is_it_sound)


#---conformance checking of alpha miner

# replay token
replay_result = token_replay.apply(log, net, initial_marking, final_marking)
print(replay_result)

fit_traces = 0
for trace in replay_result:
    if trace["trace_is_fit"]:
        fit_traces += 1
print("Number of  fit traces : ", fit_traces)

log_fitness = replay_fitness_factory.evaluate(replay_result, variant="token_replay")
print('log fitness', log_fitness)


# alignments
alignments = align_factory.apply_log(log, net, initial_marking, final_marking)
print(alignments)
#pretty_print_alignments(alignments)

log_fitness2 = replay_fitness_factory.evaluate(alignments, variant="alignments")
print('log fitness ',log_fitness2)



#   heuristics miner
net, initial_marking, final_marking = heuristics_miner.apply(log)
print('Heuristics Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz) 

#cycles = utils.get_cycles_petri_net_transitions(net)
#print(cycles)


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


#or
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

#is_it_sound = check_petri_wfnet_and_soundness(net)
#print("heuristic miner petri net is sound? ",is_it_sound)


#   conformance checking of heuristic miner

#   replay token
replay_result = token_replay.apply(log, net, initial_marking, final_marking)
print(replay_result)

fit_traces = 0
for trace in replay_result:
    if trace["trace_is_fit"]:
        fit_traces += 1
print("Number of  fit traces : ", fit_traces)

log_fitness = replay_fitness_factory.evaluate(replay_result, variant="token_replay")
print('log fitness', log_fitness)


#   alignments
alignments = align_factory.apply_log(log, net, initial_marking, final_marking)
print(alignments)
#pretty_print_alignments(alignments)

log_fitness2 = replay_fitness_factory.evaluate(alignments, variant="alignments")
print('log fitness ',log_fitness2)


#   inductive miner
net, initial_marking, final_marking = inductive_miner.apply(log)
print('Inductive Miner PetriNet\n')
gviz= vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz)

#cycles = utils.get_cycles_petri_net_transitions(net)
#print(cycles)


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

#or
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

#s_it_sound = check_petri_wfnet_and_soundness(net)
#rint("inductive miner petri net is sound? ",is_it_sound)

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
print(alignments)
#retty_print_alignments(alignments)

log_fitness2 = replay_fitness_factory.evaluate(alignments, variant="alignments")
print('log fitness ',log_fitness2)


#   directly follows graphs
#Directly-Follows graph decorated with the performance between the edges
dfg = dfg_discovery.apply(log, variant=dfg_discovery.Variants.PERFORMANCE)
gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.PERFORMANCE)
dfg_visualization.view(gviz)

#Directly-Follows graph decorated with the frequency of activities
dfg = dfg_discovery.apply(log)
gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY)
dfg_visualization.view(gviz)

net, initial_marking, final_marking = dfg_mining.apply(dfg)
gviz = dfg_visualization.apply(net, initial_marking, final_marking, variant=dfg_visualization.Variants.FREQUENCY)
dfg_visualization.view(gviz)

#ycles = utils.get_cycles_petri_net_transitions(net)
#rint(cycles)

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

#or
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

X, y, class_names = decision_mining.apply(log, net, initial_marking, final_marking, decision_point=" ")

clf, feature_names, classes = decision_mining.get_decision_tree(log, net, initial_marking, final_marking, decision_point=" ")
gviz = tree_visualizer.apply(clf, feature_names, classes)
visualizer.view(gviz)


#   statistics
#throughput time
all_case_durations = case_statistics.get_all_casedurations(log, parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
print(all_case_durations)

median_case_duration = case_statistics.get_median_caseduration(log, parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
print("median value: ",median_case_duration)  

Sum = sum(all_case_durations)
avg_time = Sum/len(all_case_durations)
print("average time of case: ",avg_time)

#distribution of case duration
x, y = case_statistics.get_kde_caseduration(log, parameters={constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"})
gviz = graphs_visualizer.apply_plot(x, y, variant=graphs_visualizer.Variants.CASES)
graphs_visualizer.view(gviz)

#distribution of events over time
x, y = attributes_filter.get_kde_date_attribute(log, attribute="time:timestamp")
gviz = graphs_visualizer.apply_plot(x, y, variant=graphs_visualizer.Variants.DATES)
graphs_visualizer.view(gviz)
