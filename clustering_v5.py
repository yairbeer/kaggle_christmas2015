"""move excess gifts into south
"""

from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np
import itertools
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.grid_search import ParameterGrid

AVG_EARTH_RADIUS = 6371  # in km


def haversine(point1, point2, miles=False):
    """ Calculate the great-circle distance bewteen two points on the Earth surface.
    :input: two 2-tuples, containing the latitude and longitude of each point
    in decimal degrees.
    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))
    :output: Returns the distance bewteen the two points.
    The default unit is kilometers. Miles can be returned
    if the ``miles`` parameter is set to True.
    """
    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(lng / 2) ** 2
    h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))
    if miles:
        return h * 0.621371  # in miles
    else:
        return h  # in kilometers

north_pole = (90, 0)
weight_limit = 1000
sleigh_weight = 10


def weighted_trip_length(stops, weights):
    tuples = [tuple(x) for x in stops.values]
    # adding the last trip back to north pole, with just the sleigh weight
    tuples.append(north_pole)
    weights.append(sleigh_weight)

    dist = 0.0
    prev_stop = north_pole
    prev_weight = sum(weights)
    for i in range(len(tuples)):
        dist += haversine(tuples[i], prev_stop) * prev_weight
        prev_stop = tuples[i]
        prev_weight = prev_weight - weights[i]
    return dist


def weighted_sub_trip_length(stops, weights, start, end):
    """
    :param stops:  places to put presents
    :param weights: weights of all the presents till the end of the WHOLE trip
    :param start: static starting point
    :param end: static end point
    :return: metric score
    """
    tuples = [tuple(x) for x in stops.values]
    # adding the last trip, with just the sleigh weight
    tuples.append(end)
    tmp_weights = list(weights)
    tmp_weights.append(sleigh_weight)

    dist = 0.0
    prev_stop = start
    prev_weight = sum(tmp_weights)
    for i in range(len(tuples)):
        dist += haversine(tuples[i], prev_stop) * prev_weight
        prev_stop = tuples[i]
        prev_weight = prev_weight - tmp_weights[i]
    return dist


def weighted_reindeer_weariness(all_trips):
    uniq_trips = all_trips.TripId.unique()
    if np.any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")
    dist = 0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId == t]
        dist = dist + weighted_trip_length(this_trip[['Latitude', 'Longitude']], this_trip.Weight.tolist())
    return dist


def trips_in_cluster_v2(gifts):
    """
    Use sorted in latitude trips in each cell
    """
    cur_trip = 0
    cur_weight = 0
    gifts.loc[:, 'TripId'] = np.ones((gifts.shape[0], 1)) * (-1)

    gifts = gifts.sort('Longitude', ascending=True)
    gift_index = list(gifts.index)
    for cur_index in gift_index:
        # add current weight
        if gifts['TripId'].loc[cur_index] == -1:
            if (cur_weight + gifts['Weight'].loc[cur_index]) <= 990:
                gifts['TripId'].at[cur_index] = cur_trip
                cur_weight += gifts['Weight'].loc[cur_index]
            else:
                # fill up trip
                gifts, cur_weight = fill_trip(gifts, cur_weight, cur_trip, gifts.loc[cur_index], 1.0)
                # add last weight
                # print 'For trip %d, the total weight was %f' % (cur_trip, cur_weight)
                cur_weight = 0
                cur_trip += 1
                gifts['TripId'].at[cur_index] = cur_trip
                cur_weight += gifts['Weight'].loc[cur_index]
    trips = []
    # print 'sorting'
    for trip in gifts['TripId'].unique():
        cur_trip = gifts[gifts['TripId'] == trip]
        cur_trip = cur_trip.sort('Latitude', ascending=False)
        trips.append(cur_trip)
    gifts = pd.concat(trips, axis=0)
    # print gifts
    return gifts


def fill_trip(gifts, cur_weight, cur_trip, cur_gift, long_limit):
    """
    Fill trips to the top
    """
    cur_long = cur_gift['Longitude']
    relevant_gifts = gifts[gifts['Longitude'] < (cur_long + long_limit)]
    relevant_gifts = relevant_gifts[gifts['TripId'] < 0]
    relevant_gifts = relevant_gifts.sort('Longitude', ascending=True)
    relevant_gifts_index = list(relevant_gifts.index)
    for cur_index in relevant_gifts_index:
        # add current weight
        if (cur_weight + relevant_gifts['Weight'].loc[cur_index]) <= 990:
            gifts['TripId'].at[cur_index] = cur_trip
            cur_weight += relevant_gifts['Weight'].loc[cur_index]
    return gifts, cur_weight


def trips_optimize_v4(gift_trips, batch_size, k_changes, changes_iterations):
    """
    Use optimized track in each trip
    :param gift_trips:
    :param batch_size:
    :return: optimized trips
    """
    trips = gift_trips['TripId'].unique()
    opt_trip = []
    # print gift_trips
    for trip_i in trips:
        # single iteration per trip
        # Working from the start
        cur_trip = gift_trips[gift_trips['TripId'] == trip_i]
        if cur_trip.shape[0] > batch_size:
            cur_improve = 1
            while cur_improve > 0:
                cur_trip_init_goal = weighted_trip_length(cur_trip[['Latitude', 'Longitude']], list(cur_trip['Weight']))
                print 'trip %d before optimization has %f weighted reindeer weariness' % \
                      (trip_i, weighted_trip_length(cur_trip[['Latitude', 'Longitude']], list(cur_trip['Weight'])))
                # print cur_trip
                # print cur_trip.shape
                # add first and last stop in the north pole
                north_trip_start = pd.DataFrame([[-1, 90, 0, 0, trip_i]],
                                                columns=["GiftId", "Latitude", "Longitude", "Weight", "TripId"])
                north_trip_end = pd.DataFrame([[-2, 90, 0, 10, trip_i]],
                                              columns=["GiftId", "Latitude", "Longitude", "Weight", "TripId"])
                cur_trip = pd.concat([north_trip_start, cur_trip, north_trip_end])
                cur_trip = k_change_optimize_dynamic(cur_trip, k_changes, changes_iterations)
                single_trip = []
                for batch_i in range(1, cur_trip.shape[0], batch_size):
                    if (batch_i + batch_size) < cur_trip.shape[0]:
                        # print 'norm batch'
                        # print cur_trip.iloc[batch_i - 1: batch_i + batch_size]
                        optimize_batch = batch_optimize_dynamic(cur_trip.iloc[batch_i - 1: batch_i + batch_size],
                                                                cur_trip['Weight'].iloc[batch_i - 1:])
                    else:
                        if cur_trip.iloc[batch_i - 1:].shape[0] > 3:
                            # print 'last batch opt'
                            # print cur_trip.iloc[batch_i - 1:]
                            optimize_batch = batch_optimize_dynamic(cur_trip.iloc[batch_i - 1:],
                                                                    cur_trip['Weight'].iloc[batch_i - 1:])
                        else:
                            # print 'last batch not opt'
                            # print cur_trip.iloc[batch_i:]
                            optimize_batch = cur_trip.iloc[batch_i:]
                    single_trip.append(optimize_batch)
                cur_trip = pd.concat(single_trip)

                # working from the middle of the  1st batch
                single_trip = [cur_trip.iloc[:(batch_size/2)]]
                cur_trip = cur_trip.iloc[((batch_size/2) - 1):]
                for batch_i in range(1, cur_trip.shape[0], batch_size):
                    if (batch_i + batch_size) < cur_trip.shape[0]:
                        # print 'norm batch'
                        # print cur_trip.iloc[batch_i - 1: batch_i + batch_size]
                        optimize_batch = batch_optimize_dynamic(cur_trip.iloc[batch_i - 1: batch_i + batch_size],
                                                                cur_trip['Weight'].iloc[batch_i - 1:])
                    else:
                        if cur_trip.iloc[batch_i - 1:].shape[0] > 3:
                            # print 'last batch opt'
                            # print cur_trip.iloc[batch_i - 1:]
                            optimize_batch = batch_optimize_dynamic(cur_trip.iloc[batch_i - 1:],
                                                                    cur_trip['Weight'].iloc[batch_i - 1:])
                        else:
                            # print 'last batch not opt'
                            # print cur_trip.iloc[batch_i:]
                            optimize_batch = cur_trip.iloc[batch_i:]
                    single_trip.append(optimize_batch)
                # remove the return to the north pole
                cur_trip = pd.concat(single_trip)
                cur_trip = cur_trip.iloc[:-1]
                cur_trip_final_goal = weighted_trip_length(cur_trip[['Latitude', 'Longitude']], list(cur_trip['Weight']))
                cur_improve = cur_trip_init_goal - cur_trip_final_goal
                print 'iteration improve:', cur_improve
        opt_trip.append(cur_trip)
    opt_trip = pd.concat(opt_trip)
    return opt_trip


def batch_optimize_dynamic(batch_gifts, batch_weights):
    """
    optimize a single batch. need to add sleigh weight
    :param batch_gifts: free parameters for optimizing, last point is static
    :return: optimized batch without start
    """

    batch_index = list(batch_gifts.index)
    n_batch = len(batch_index)

    # calculating all the edges
    batch_gifts_weights = list(batch_weights)
    haver_mat = np.ones((n_batch, n_batch))
    for i in range(haver_mat.shape[0]):
        for j in range(haver_mat.shape[0]):
            if i != j:
                haver_mat[i, j] = haversine(list(batch_gifts.loc[batch_index[i], ['Latitude', 'Longitude']]),
                                            list(batch_gifts.loc[batch_index[j], ['Latitude', 'Longitude']]))

    best_metric = weighted_sub_trip_length_dynamic(range(haver_mat.shape[0]), batch_gifts_weights, haver_mat)
    best_perm = range(haver_mat.shape[0])
    # print 'Before optimization %f' % weighted_sub_trip_length(batch_gifts[['Latitude', 'Longitude']],
    #                                                           weights, start, stop)

    tries = list(itertools.permutations(range(1, n_batch-1)))

    tries = map(lambda x: [0] + list(x) + [n_batch-1], tries)

    for perm in tries:
        cur_metric = weighted_sub_trip_length_dynamic(perm, batch_gifts_weights, haver_mat)
        if cur_metric < best_metric:
            best_metric = cur_metric
            best_perm = perm

    # from order to opt batch
    opt_batch_index = []
    for i in range(len(best_perm)):
        opt_batch_index.append(batch_index[best_perm[i]])

    opt_batch = batch_gifts.iloc[best_perm]
    opt_batch = opt_batch.iloc[1:]
    # print 'After optimization %f' % weighted_sub_trip_length(best_batch[['Latitude', 'Longitude']],
    #                                                          weights, start, stop)
    # if (best_metric - base_metric) < 0:
    #     print 'weariness gain: %f' % (best_metric - base_metric)
    return opt_batch


def k_change_optimize_dynamic(trip_gifts, k_changes, opt_iterations):
    """
    optimize a single batch. need to add sleigh weight
    :param batch_gifts: free parameters for optimizing, last point is static
    :return: optimized batch without start
    """
    trip_num = trip_gifts['TripId'].iloc[1]
    n_trip = trip_gifts.shape[0]
    trip_index = list(trip_gifts.index)
    trip_index[0] = -1
    trip_index[-1] = -2
    trip_gifts.index = trip_index

    # calculating all the edges
    gifts_weights = list(trip_gifts['Weight'])
    haver_mat = np.ones((n_trip, n_trip))
    for i in range(haver_mat.shape[0]):
        for j in range(haver_mat.shape[0]):
            if trip_num == 82:
                print list(trip_gifts.loc[trip_index[i], ['Latitude', 'Longitude']]), \
                    list(trip_gifts.loc[trip_index[j], ['Latitude', 'Longitude']])
            if i != j:
                haver_mat[i, j] = haversine(list(trip_gifts.loc[trip_index[i], ['Latitude', 'Longitude']]),
                                            list(trip_gifts.loc[trip_index[j], ['Latitude', 'Longitude']]))

    best_metric = base_metric = weighted_sub_trip_length_dynamic(range(haver_mat.shape[0]), gifts_weights, haver_mat)
    best_perm = range(haver_mat.shape[0])
    # print 'Before optimization %f' % weighted_sub_trip_length(batch_gifts[['Latitude', 'Longitude']],
    #                                                           weights, start, stop)

    tries = []
    for i in range(opt_iterations):
        tries.append(np.random.choice(np.arange(1, n_trip-1), k_changes, replace=False))

    for change_iter in tries:
        # add change of poles
        tmp_perm = list(best_perm)
        for j in range(change_iter.shape[0] - 1):
            tmp_perm[change_iter[j+1]] = best_perm[change_iter[j]]
        tmp_perm[change_iter[0]] = best_perm[change_iter[-1]]
        cur_metric = weighted_sub_trip_length_dynamic(tmp_perm, gifts_weights, haver_mat)
        if cur_metric < best_metric:
            best_metric = cur_metric
            best_perm = tmp_perm

    # # from order to opt trip
    # opt_batch_index = []
    # for i in range(len(best_perm)):
    #     opt_batch_index.append(batch_index[best_perm[i]])

    opt_trip = trip_gifts.iloc[best_perm]
    # print 'After optimization %f' % weighted_sub_trip_length(best_batch[['Latitude', 'Longitude']],
    #                                                          weights, start, stop)
    if (best_metric - base_metric) < 0:
        print 'weariness gain: %f' % (best_metric - base_metric)
    return opt_trip


def weighted_sub_trip_length_dynamic(stops, weights, haversine_matrix):
    """
    :param stops: list of index places to put presents including end point
    :param weights: weights of all the presents in the batch including end point
    :param haversine_matrix: array with haversine
    :return: metric score
    """

    tmp_weights = list(weights)

    dist = 0
    prev_weight = sum(tmp_weights) - tmp_weights[0]
    for i in range(1, len(stops)):
        dist += haversine_matrix[stops[i - 1], stops[i]] * prev_weight
        prev_weight = prev_weight - tmp_weights[stops[i]]
    return dist


def trips_in_cluster_v3(gifts):
    """
    Use sorted in latitude trips in each cell
    """
    cur_trip = 0
    cur_weight = 0
    gifts.loc[:, 'TripId'] = np.ones((gifts.shape[0], 1)) * (-1)

    gifts = gifts.sort('Longitude', ascending=True)
    gift_index = list(gifts.index)
    for cur_index in gift_index:
        # add current weight
        if gifts['TripId'].loc[cur_index] == -1:
            if (cur_weight + gifts['Weight'].loc[cur_index]) <= 990:
                gifts['TripId'].at[cur_index] = cur_trip
                cur_weight += gifts['Weight'].loc[cur_index]
            else:
                # fill up trip
                gifts, cur_weight = fill_trip(gifts, cur_weight, cur_trip, gifts.loc[cur_index], 1.0)
                # add last weight
                # print 'For trip %d, the total weight was %f' % (cur_trip, cur_weight)
                cur_weight = 0
                cur_trip += 1
                gifts['TripId'].at[cur_index] = cur_trip
                cur_weight += gifts['Weight'].loc[cur_index]
    trips = []
    # print 'sorting'
    for trip in gifts['TripId'].unique():
        cur_trip = gifts[gifts['TripId'] == trip]
        cur_trip = cur_trip.sort('Latitude', ascending=False)
        trips.append(cur_trip)
    gifts = pd.concat(trips, axis=0)
    # print gifts
    return gifts

"""
Start Main program
"""
# GiftId   Latitude   Longitude     Weight
gifts = pd.read_csv('gifts.csv')
# gifts = gifts.iloc[:1000]  # training


def solve(gifts):
    # Main parameters
    # n_gifts = gifts.shape[0]
    # print 'There are %d gifts to distribute' % n_gifts

    # print 'Starting to plan trips by longitude'
    gifts = trips_in_cluster_v2(gifts)
    # print(weighted_reindeer_weariness(gifts))

    # print 'Start in trip batch optimizing'
    gifts = trips_optimize_v4(gifts, 5, 4, 50)
    print(weighted_reindeer_weariness(gifts))

    return gifts

"""
clustering
"""
param_grid = {'eps': [20], 'min_samples': [1400]}
for params in ParameterGrid(param_grid):
    print params
    gifts_south = gifts[gifts['Latitude'] <= -70]
    gifts_north = gifts[gifts['Latitude'] > -70]

    gifts_north_clustering = np.array(gifts_north[['Latitude', 'Longitude']])
    db = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(gifts_north_clustering)
    labels = pd.Series(db.labels_)
    print labels.value_counts()
    labels_unique = labels.unique()

    gifts_south = pd.concat([gifts_south, gifts_north.loc[np.array(labels == (-1))]])

    # # plot north clusters
    # for ind in labels_unique:
    #     plt.plot(np.array(gifts_north['Longitude'].loc[np.array(labels == ind)]),
    #              np.array(gifts_north['Latitude'].loc[np.array(labels == ind)]), 'ro')
    #     plt.grid(True)
    #     plt.show()

    # Check if it is better to add last trip to south or not
    gifts_cluster = []
    for i in labels_unique:
        if i != (-1):
            # solve i trip
            gifts_i = gifts_north.loc[np.array(labels == i)]
            if gifts_cluster:
                gifts_cluster_trip_start = gifts_cluster[-1]['TripId'].iloc[-1] + 1
            gifts_i = solve(gifts_i)
            if gifts_cluster:
                gifts_i.loc[:, 'TripId'] += gifts_cluster_trip_start
            # south trips
            gifts_south_trip_start = gifts_i['TripId'].iloc[-1] + 1
            gifts_south = solve(gifts_south)
            gifts_south.loc[:, 'TripId'] += gifts_south_trip_start
            # concat
            gift_trips_i_last = pd.concat([gifts_south, gifts_i])
            # score before
            score_i_last = weighted_reindeer_weariness(gift_trips_i_last)
            print 'score without dropping last cluster trip: ', score_i_last

            # moving last trip to south
            gifts_i_trip_last_index = gifts_i['TripId'].iloc[-1]
            gifts_i_trip_last = gifts_i[gifts_i['TripId'] == gifts_i_trip_last_index]
            gifts_south_w_last = pd.concat([gifts_south, gifts_i_trip_last])

            # recalculate south equalibirium
            gifts_i_trip_wo_last = gifts_i[gifts_i['TripId'] != gifts_i_trip_last_index]
            gifts_south_trip_start = gifts_i_trip_wo_last['TripId'].iloc[-1] + 1
            gifts_south_w_last = solve(gifts_south_w_last)
            gifts_south_w_last.loc[:, 'TripId'] += gifts_south_trip_start
            gift_trips_south_last = pd.concat([gifts_south_w_last, gifts_i_trip_wo_last])
            score_south_last = weighted_reindeer_weariness(gift_trips_south_last)
            print 'score after dropping last cluster trip', score_south_last
            if score_i_last < score_south_last:
                print 'last trip stays in cluster'
                gifts_cluster.append(gifts_i)
            else:
                print 'last trip goes to south'
                gifts_cluster.append(gifts_i_trip_wo_last)
                gifts_south = gifts_south_w_last
    gift_trips = pd.concat([gifts_south] + gifts_cluster)

    print '*****************************'
    print '*****************************'
    print '*****************************'
    print(weighted_reindeer_weariness(gift_trips))
    print '*****************************'
    print '*****************************'
    print '*****************************'

# print gift_trips

print 'writing results to file'
gift_trips = np.array(gift_trips)
gift_trips = gift_trips[:, [0, -1]]
gift_trips = pd.DataFrame(gift_trips)
gift_trips.columns = ['GiftId', 'TripId']

gift_trips = gift_trips.astype('int32')
gift_trips.index = gift_trips["GiftId"]
del gift_trips["GiftId"]
# gift_trips.to_csv('cluster_continents_trips_batch_81.csv')

# Basecase: 144525525772.0
# Resolution 10 clustering: 34230724056.0
# Resolution 5 clustering with ordering by latitude: 17723267396.9
# resolution_latitude = 45; resolution_longitude = 1, clustering with ordering by latitude: 13227163205.6
# resolution_longitude = 0.5, clustering with ordering by latitude: 12787535216.9
# resolution_longitude = 0.1, clustering with ordering by latitude: 12674006549.1
# resolution_longitude = 0.1, clustering with ordering by latitude, batch = 3: 12671850614.7
# resolution_longitude = 0.1, clustering with ordering by latitude, batch = 3, 4: 12669475033.6
# resolution_longitude = 0.1, clustering with ordering by latitude, batch = 3, 4, 5: 12667518270.7
# resolution_longitude = 0.1, clustering with ordering by latitude, batch = 3, 4, 5; 5 iterations: 12666394933.7

# V2: longitude ordering
# ordering by latitude, batch = 3, 4, 5; 2 iterations: 12663733540.7
# ordering by latitude, batch = 5; iterative, fililing: 12664637799.4
# ordering by latitude, fililing: 12668167971.9
# ordering by latitude, batch = 5, 6; iterative, fililing: 12663055569.6

# V3 North / South clustering
# cluster north + south before optimization: 12595133701.3
#  cluster north + south, batch = 5 optimization: 12575675485.4
#  cluster north + south, batch = 6 optimization:

# V3.1 continent clustering:
# cluster north + south, removed DBSCAN outliers to south, batch = 5 optimization: 12537678107.5
# cluster north + south, removed DBSCAN outliers to south, batch = 6 optimization: 12536291165.9
# {'min_samples': 1250, 'eps': 16}, batch = 5 optimization: 12516053337.6
# {'min_samples': 1250, 'eps': 16}, batch = 7 optimization: 12513554845.5
# {'min_samples': 1250, 'eps': 16}, batch = 7 optimization, added merkov [batch, 2*batch]:
