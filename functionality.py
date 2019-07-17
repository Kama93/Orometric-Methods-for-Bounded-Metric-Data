# All functionality that is needed for the experiments is inside this script.

import json
import networkx as nx
from queue import PriorityQueue
import geopy.distance as geo
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import itertools
import os
import numpy as np

# Input output stuff


def load_data(path):
    """
    Reads json-data at ``path'' and returns it as dictionary/list.
    """
    with open(path) as file:
        data = json.load(file)
    return data


def write_data(data, path, indent=4, ensure_ascii=False):
    """
    Write ``data'' to json at ``path''.
    """
    with open(path, 'w') as file:
        json.dump(data, file, indent=indent, ensure_ascii=ensure_ascii)


def data_path(directory, output=False):
    """
    Gets experiment ``directory'' and returns full path to experiment data.
    """
    if directory[-1] != '/':
        directory += '/'
    path = 'dataset/' + directory
    if output:
        path += 'output/'
    return path


def wikidata_path(number):
    """
    Makefull wikidata property path from number
    """
    return "http://www.wikidata.org/entity/Q" + str(number)


wp = wikidata_path


def load_dataframe(directory):
    """
    Load stats for prediction into dataframe for easy synergy with scikit learn
    """
    path = data_path(directory=directory, output=True)
    stats = load_data(path + 'stats_for_prediction.json')
    columns = ["name", "population", "isolation", "prominence", "unis", "has_uni"]
    items = list(stats.items())
    data = [[value["name"], value["population"], value["isolation"],
             value["prominence"], value["unis"], value["unis"] > 0]
            for _, value in items]
    indices = [index for index, _ in items]
    return pd.DataFrame(data=data, columns=columns, index=indices)


def to_tex(dataframe, file, columns=None):
    """"
    Write dataframe to tex.
    """
    if not os.path.isdir('latex_output/'):
        os.makedirs('latex_output/')
    with open('latex_output/' + file + '.txt', 'w') as file:
        dataframe.to_latex(buf=file, columns=columns, float_format="%.4f")


# Normalization function


def normalize(data):
    """
    Returns a normalized version of data by converting the values to the range of
    0 to 1.
    """
    if isinstance(data, dict):
        one = max(data.values())
        zero = min(data.values())
        if one != zero:
            denominator = one - zero
            return {key: (data[key] - zero) / denominator for key in data.keys()}
        else:
            return {key: 1 for key in data.keys()}
    else:
        one = max(data)
        zero = min(data)
        if one != zero:
            denominator = one - zero
            return [(x - zero) / denominator for x in data]
        else:
            return [1 for _ in data]

# Summary functionaility: Put all relevant data in json dict in eye friendly format.


def make_summary(directory):
    """
    Write summary and make corrections for directory
    """
    path = data_path(directory)
    city_list = load_data(path + 'cities.json')
    uni_list = make_corrections(directory)
    write_data(uni_list, path + 'universities.json')
    s = make_city_dict(city_list)
    result = add_universities(s, uni_list)
    path += 'output/'
    if not os.path.isdir(path):
        os.makedirs(path)
    write_data(result, path + 'summary.json')


def make_corrections(directory):
    """
    Make corrections for universities in directory
    """
    path = data_path(directory)
    city_list = load_data(path + 'cities.json')
    uni_list = load_data(path + 'universities_uncorrected.json')
    uni_list_corrected = []
    corrections = load_data(path + 'corrections.json')
    city_dict = {}
    # Make city_dicts first
    for [_, _, city_id] in corrections:
        identifier = wp(city_id)
        for city in city_list:
            if city["city"] == identifier:
                city_dict[city["city"]] = city["cityLabel"]
    correction_dict = {wp(x): (wp(y), wp(z)) for [x, y, z] in corrections}
    for uni in uni_list:
        if uni["uni"] in correction_dict and uni["city"] == correction_dict[uni["uni"]][0]:
            uni_list_corrected.append({"city": correction_dict[uni["uni"]][1],
                                       "uni": uni["uni"],
                                       "cityLabel": city_dict[correction_dict[uni["uni"]][1]],
                                       "uniLabel": uni["uniLabel"]})
        else:
            uni_list_corrected.append(uni)
    return uni_list_corrected


def make_city_dict(city_list):
    """
    Helper for make summary
    """
    summary = {}
    for city in city_list:
        wikidata_id = city['city']
        summary[wikidata_id] = {'longitude': float(city['longitude']),
                                'latitude': float(city['latitude']),
                                'population': int(city['population']),
                                'name': city['cityLabel']}
    return summary


def add_universities(city_dict, universities):
    """
    Helper for make summary: Add all universites to summary.
    """
    summary = {key: city_dict[key] for key in city_dict.keys()}
    for city in summary.keys():
        summary[city]['universities'] = []
    for uni in universities:
        uni_id = uni['uni']
        uni_name = uni['uniLabel']
        city_id = uni['city']
        city = uni['cityLabel']
        if city_id in summary:
            summary[city_id]['universities'].append((uni_id, uni_name))
    for city_id in summary.keys():
        summary[city_id]['universities'] = [[uni_id, uni_name] for (uni_id, uni_name)
                                            in list(set(summary[city_id]['universities']))]
        summary[city_id]['#universities'] = len(summary[city_id]['universities'])
    return summary

###########################
# Prominence and Isolation#
###########################


def isolation(graph, node, height, distance, show_dominating_point=False):
    """
    Takes a nx-graph ``graph'' and computes the isolation of the node ``node'',
    based on the height-dict ``height'', and the distance-dict``distance''.
    """
    h = height[node]
    distances = [(distance[node][node1], node1)
                 for node1 in graph.nodes() if height[node1] >= h and node1 != node]
    if distances:
        if show_dominating_point:
            return min(distances)
        else:
            return min(distances)[0]
    else:
        result = max([distance[node][node1] for node1 in graph.nodes()])
        if show_dominating_point:
            result = [result]
        return result


def compute_all_isolations(graph, height, distance,  show_dominating_point=False):
    """
    Takes a nx-graph ``graph'' and computes the isolation  of all nodes based on the height
    dict ``height'' and the distance dict ``distance''.
    If no distance function is provided, an optimized version of isolation for
    the shortest path distance is used.
    """
    return {node: isolation(graph, node, height, distance, show_dominating_point)
            for node in graph.nodes()}

# Prominence


def prominence(graph, node, height):
    """
    Computes the prominence of ``node'' in ``graph'', based on the dict ``height''.
    """
    h = height[node]
    for m in graph.neighbors(node):
        if height[m] >= h:
            return 0
    queue = PriorityQueue()
    queue.put((0, node))
    closed = set()
    added = set()
    while not queue.empty():
        (maxHeightDiff, k) = queue.get()
        if k in closed:
            continue
        closed.add(k)
        for m in graph.neighbors(k):
            if m in closed or m in added:
                continue
            if height[m] >= h:
                return maxHeightDiff
            else:
                added.add(m)
                newDiff = h - height[m]
                if newDiff > maxHeightDiff:
                    queue.put((newDiff, m))
                else:
                    queue.put((maxHeightDiff, m))
    return height[node]


def compute_all_prominences(graph, height):
    """
    Returns the prominence of all ``nodes'' in ``graph''.
    """
    return {node: prominence(graph, node, height) for node in graph.nodes()}


#####################################
# Computation of minimal step-graph #
#####################################

# Helper stuff


def distance_graph(distances, epsilon):
    """
    Returns for a given dict of dict storing ``distances'' the graph,
    where edges are defined for pairs with distances not greater then ``epsilon''.
    """
    graph = nx.Graph()
    for node in distances.keys():
        graph.add_node(node)
        current_distances = distances[node]
        for node1 in current_distances.keys():
            if current_distances[node1] <= epsilon and node != node1:
                graph.add_edge(node, node1)
    return graph


def smallest_connected_graph(distances):
    """
    Returns for given points and there ``distances'', given as dict of dicts,
    the graph where two points have an edge, if there distance is smaller than epsilon.
    Epsilon is chosen minimal, that every node has at least one neighbor.
    """
    epsilon = max([min([distances[x][y] for y in distances.keys() if y != x])
                   for x in distances.keys()])
    return distance_graph(distances, epsilon)

# Main function for computing the graph.


def make_smallest_connected_graph(directory):
    """
    Make smallest graph where every node has a real neighbour for dataset
    in directory.
    """
    path = data_path(directory, True)
    summary = load_data(path + 'summary.json')
    distances = {city1: {city2: geo.great_circle((summary[city1]['latitude'],
                                                 summary[city1]['longitude']),
                                                 (summary[city2]['latitude'],
                                                 summary[city2]['longitude'])).km
                         for city2 in summary.keys()}
                 for city1 in summary.keys()}
    g = smallest_connected_graph(distances)
    write_data(nx.to_dict_of_lists(g), path + 'minimal_graph.json')

# Make stats in format for classsification


def make_stats_for_prediction(directory):
    """
    Compute all normalized stats. The generated file is used for
    classifcation.
    """
    path = data_path(directory, True)
    summary = load_data(path + 'summary.json')
    heights = {city: summary[city]['population'] for city in summary.keys()}
    distance = {city1: {city2: geo.great_circle((summary[city1]['latitude'],
                                                 summary[city1]['longitude']),
                                                (summary[city2]['latitude'],
                                                 summary[city2]['longitude'])).km
                        for city2 in summary.keys()}
                for city1 in summary.keys()}
    graph = nx.from_dict_of_lists(load_data(path+'minimal_graph.json'))
    isolations = compute_all_isolations(graph=graph,
                                        height=heights,
                                        distance=distance)
    prominences = compute_all_prominences(graph=graph,
                                          height=heights)
    heights = normalize(heights)
    isolations = normalize({key: isolations[key] for key in summary.keys()})
    prominences = normalize({key: prominences[key] for key in summary.keys()})
    stats_to_plot = {key: {'name': summary[key]['name'],
                           'population': heights[key],
                           'isolation': isolations[key],
                           'prominence': prominences[key],
                           'unis': summary[key]['#universities']}
                     for key in heights.keys()}
    write_data(stats_to_plot, path + 'stats_for_prediction.json')

############################
# Stuff for classsification#
############################


def validation(dataframe,
               label="has_uni",
               features=("isolation", "prominence", "population"),
               folds=5,
               classifier="SVM",
               iterations=100):
    """
    Make classification for one specific classifier and feature set.
    """
    result = {}
    for i in range(iterations):
        print(i)
        current_fold = []
        kf = KFold(n_splits=folds, shuffle=True, random_state=i)
        for train, test in kf.split(dataframe):
            d_train = dataframe.iloc[train]
            d_test = dataframe.iloc[test]
            values_train = d_train[list(features)]
            classes_train = d_train[label]
            values_test = d_test[list(features)]
            classes_test = d_test[label]
            weight = 'balanced'
            if classifier == "SVM":
                clf = SVC(kernel='rbf', gamma=1, random_state=i, class_weight=weight)
            elif classifier == "LR":
                clf = LogisticRegression(random_state=i,
                                         class_weight=weight,
                                         solver='liblinear')
            clf.fit(values_train, classes_train)
            classes_pred = clf.predict(values_test)
            conf_matrix = confusion_matrix(y_true=classes_test, y_pred=classes_pred,
                                           labels=[False, True])
            current_fold.append({'conf_matrix': conf_matrix.tolist()})
        result["fold_" + str(i)] = current_fold
    # Check if it has just one fold
    if len(result) == 1:
        result = result['fold_0']
    return result


def validation_for_dataset(directory, iterations=100,
                           classifiers=("SVM",
                                        "LR")):
    """
    Do classification for all classifiers and all feature combinations and write results.
    """
    path = data_path(directory=directory, output=True) + "results/"
    features = ["isolation", "prominence", "population"]
    feature_iterations = [combination for i in range(1, 4)
                          for combination in itertools.combinations(features, i)]
    # Load frame and sort it for deterministic behaiour in older python versions
    # where order preserving for dicts is not guaranteed
    dataframe = load_dataframe(directory).sort_values(by=['population',
                                                          'isolation',
                                                          'prominence',
                                                          'name'],
                                                      ascending=False)
    for features in feature_iterations:
        print("Make experiments for " + str(features))
        for classifier in classifiers:
            print("Start Classification via " + classifier)
            outputpath = path + classifier + '/' + '_'.join(features) + '/'
            result = validation(dataframe,
                                label="has_uni",
                                features=features,
                                folds=5,
                                classifier=classifier,
                                iterations=iterations)
            if not os.path.isdir(outputpath):
                os.makedirs(outputpath)
            write_data(result, outputpath + 'result.json')


def evaluate_classification(directory, combination, classifier):
    """
    Evaluate results of classifcation for specific feature combination
    and classifier.
    """
    path = data_path(directory, output=True) + 'results/'
    ex_path = path + classifier + '/' + combination + '/'
    data = load_data(ex_path + 'result.json')
    fold_stats = [pd.DataFrame(data=[x['conf_matrix'][0]+x['conf_matrix'][1]
                                     for x in fold],
                               columns=['TN', 'FP', 'FN', 'TP']).sum()
                  for fold in data.values()]
    fold_scores = pd.DataFrame(data=[[fold['TN']/(fold['TN']+fold['FP']),
                                      fold['TP']/(fold['TP']+fold['FN']),
                                      np.sqrt((fold['TN']/(fold['TN']+fold['FP'])) *
                                              (fold['TP']/(fold['TP']+fold['FN'])))]
                                     for fold in fold_stats],
                               columns=['acc-', 'acc+', 'g-mean'])
    return fold_scores


def final_table(directories=('france', 'germany'), classifiers=('SVM', 'LR')):
    """
    Make final table of all results.
    """
    features = ["isolation", "prominence", "population"]
    feature_iterations = ['_'.join(combination) for i in range(1, 4)
                          for combination in itertools.combinations(features, i)]
    columns = pd.MultiIndex.from_product([list(directories), list(classifiers),
                                          ['mean', 'std']],
                                         names=('Country', 'Classifier', 'Score'))
    index = pd.MultiIndex.from_product([feature_iterations, ['acc+', 'acc-', 'g-mean']])
    result = []
    for feature in feature_iterations:
        frames = [[evaluate_classification(directory,
                                           combination=feature,
                                           classifier=classifier)
                   for classifier in classifiers] for directory in directories]
        values = [[[a.mean(), a.std()]] for x in frames for a in x]
        current_l = [[x[row] for z in values for y in z for x in y]
                     for row in ['acc+', 'acc-', 'g-mean']]
        for x in current_l:
            result.append(x)
    frame = pd.DataFrame(data=result, columns=columns, index=index)
    to_tex(frame, 'table')
    return frame
