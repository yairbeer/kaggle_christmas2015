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
        cur_trip = single_trip_optimize(cur_trip, batch_size, k_changes, changes_iterations)
        opt_trip.append(cur_trip)
    opt_trip = pd.concat(opt_trip)
    return opt_trip


def single_trip_optimize(cur_trip, batch_size, k_changes, changes_iterations):
    if cur_trip.shape[0] > batch_size:
        cur_improve = 1
        trip_i = cur_trip['TripId'].iloc[1]
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
            if k_changes > 0:
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
    return cur_trip


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
    :param trip_gifts: free parameters for optimizing, first & last point is static
    :return: optimized batch without start
    """
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


gifts_trip = pd.read_csv('shootoutyoureye_template.csv')
gifts = pd.read_csv('gifts.csv')
gifts_trip = pd.merge(gifts_trip, gifts, on='GiftId')
print gifts_trip
print(weighted_reindeer_weariness(gifts_trip))
gifts_trip = trips_optimize_v4(gifts_trip, 9, 3, 100)
print gifts_trip
print(weighted_reindeer_weariness(gifts_trip))

print 'writing results to file'
gift_trips = np.array(gifts_trip)
gift_trips = gift_trips[:, [0, 3]]
gift_trips = pd.DataFrame(gift_trips)
gift_trips.columns = ['GiftId', 'TripId']

gift_trips = gift_trips.astype('int32')
gift_trips.index = gift_trips["GiftId"]
del gift_trips["GiftId"]
gift_trips.to_csv('opt_shooteyes_template.csv')
