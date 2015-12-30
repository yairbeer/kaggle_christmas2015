"""move excess gifts into south
"""

from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np
import itertools
import csv

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
        if not trip_i % 20:
            print 'trip %d optimization' % trip_i
        cur_trip = single_trip_optimize(cur_trip, batch_size, k_changes, changes_iterations)
        opt_trip.append(cur_trip)
    opt_trip = pd.concat(opt_trip)
    return opt_trip


def single_trip_optimize(cur_trip, batch_size, k_changes, changes_iterations):
    cur_trip_init_goal = weighted_trip_length(cur_trip[['Latitude', 'Longitude']], list(cur_trip['Weight']))
    if cur_trip.shape[0] < (1.5 * batch_size):
        batch_size = cur_trip.shape[0] / 2
        if not batch_size:
            return cur_trip, cur_trip_init_goal
    # print cur_trip
    # print cur_trip.shape
    # add first and last stop in the north pole
    north_trip_start = pd.DataFrame([[-1, 90, 0, 0, 0]],
                                    columns=["GiftId", "Latitude", "Longitude", "Weight", "TripId"])
    north_trip_end = pd.DataFrame([[-2, 90, 0, 10, 0]],
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
            if cur_trip.iloc[batch_i - 1:].shape[0] > 4:
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
    # print 'iteration improve:', cur_improve
    return cur_trip, cur_trip_final_goal


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


def weighted_sub_trip_length_v2(stops, weights):
    """
    :param stops: list of index places to put presents including end point
    :param weights: weights of all the presents in the batch including end point
    :return: metric score
    """

    tmp_weights = list(weights)
    if tmp_weights > weight_limit:
        raise Exception("One of the sleighs over weight limit!")

    dist = 0
    prev_weight = sum(tmp_weights) - tmp_weights[0]
    for i in range(1, len(stops)):
        dist += haversine(stops[i-1], stops[i]) * prev_weight
        prev_weight = prev_weight - tmp_weights[i]
    return dist


def trips_orginizer(gifts, weight_limit):
    """
    Use sorted in latitude trips in each cell
    """
    cur_trip = 0
    cur_weight = 0
    gifts.loc[:, 'TripId'] = np.ones((gifts.shape[0], 1)) * (-1)

    gifts = gifts.sort_values('Longitude', ascending=True)
    gift_index = list(gifts.index)
    for cur_index in gift_index:
        # add current weight
        if (cur_weight + gifts['Weight'].loc[cur_index]) <= weight_limit:
            gifts['TripId'].at[cur_index] = cur_trip
            cur_weight += gifts['Weight'].loc[cur_index]
        else:
            # add last weight
            # print 'For trip %d, the total weight was %f' % (cur_trip, cur_weight)
            cur_weight = 0
            cur_trip += 1
            gifts['TripId'].at[cur_index] = cur_trip
            cur_weight += gifts['Weight'].loc[cur_index]
    # print 'sorting'
    trips = []
    for trip in gifts['TripId'].unique():
        cur_trip = gifts[gifts['TripId'] == trip]
        cur_trip = cur_trip.sort_values('Latitude', ascending=False)
        trips.append(cur_trip)
    gifts = pd.concat(trips, axis=0)
    # print gifts
    return gifts


def fill_trip(gifts, cur_weight, cur_trip, cur_gift, long_limit, weight_limit):
    """
    Fill trips to the top
    """
    cur_long = cur_gift['Longitude']
    relevant_gifts = gifts[gifts['Longitude'] < (cur_long + long_limit)]
    relevant_gifts = relevant_gifts[gifts['TripId'] < 0]
    relevant_gifts = relevant_gifts.sort_valus('Longitude', ascending=True)
    relevant_gifts_index = list(relevant_gifts.index)
    for cur_index in relevant_gifts_index:
        # add current weight
        if (cur_weight + relevant_gifts['Weight'].loc[cur_index]) <= weight_limit:
            gifts['TripId'].at[cur_index] = cur_trip
            cur_weight += relevant_gifts['Weight'].loc[cur_index]
    return gifts, cur_weight


def gift_switch_optimize_v2(gifts_a, gifts_b, n_tries=100, poisson_items=1.5, trip_max_weight=990):
    """
    Optimizing between 2 trips using poisson probability of adjacted gifts
    """
    best_trip_a = gifts_a
    a_trip_id = gifts_a['TripId'].iloc[0]
    best_trip_b = gifts_b
    b_trip_id = gifts_b['TripId'].iloc[0]

    base_metric_a = weighted_trip_length(gifts_a[['Latitude', 'Longitude']],
                                         list(gifts_a['Weight']))
    base_metric_b = weighted_trip_length(gifts_b[['Latitude', 'Longitude']],
                                         list(gifts_b['Weight']))
    best_metric = base_metric = base_metric_a + base_metric_b

    # greedy
    for change_iter in range(n_tries):
        # print 'iteration %d' % change_iter

        # load current best trips
        cur_trip_a = best_trip_a.copy(deep=True)
        cur_trip_b = best_trip_b.copy(deep=True)
        n_trip_a = cur_trip_a.shape[0]
        n_trip_b = cur_trip_b.shape[0]

        # load change arrays
        items_chosen_a = np.random.poisson(poisson_items)
        items_chosen_b = np.random.poisson(poisson_items)
        if (items_chosen_a or items_chosen_b) and (n_trip_a >= items_chosen_a) and (n_trip_b >= items_chosen_b):
            # remove gifts from current trips
            if items_chosen_a:
                try_a = np.random.choice(np.arange(n_trip_a - items_chosen_a + 1), 1, replace=False)
                try_a = range(try_a, (try_a + items_chosen_a))
                try_a_items = cur_trip_a.iloc[try_a]
                try_a_items['TripId'] = np.repeat(b_trip_id, items_chosen_a)
                try_a_not = range(n_trip_a)
                for moved_vals in try_a:
                    try_a_not.remove(moved_vals)
                cur_trip_a = cur_trip_a.iloc[try_a_not]
                n_trip_a = cur_trip_a.shape[0]

            if items_chosen_b:
                try_b = np.random.choice(np.arange(n_trip_b - items_chosen_b + 1), 1, replace=False)
                try_b = range(try_b, (try_b + items_chosen_b))
                try_b_items = cur_trip_b.iloc[try_b]
                try_b_items['TripId'] = np.repeat(a_trip_id, items_chosen_b)
                try_b_not = range(n_trip_b)
                for moved_vals in try_b:
                    try_b_not.remove(moved_vals)
                cur_trip_b = cur_trip_b.iloc[try_b_not]
                n_trip_b = cur_trip_b.shape[0]

            # move items
            if items_chosen_a:
                if n_trip_b:
                    a_to_index = np.random.randint(n_trip_b)
                    if not a_to_index:
                        cur_trip_b_new = pd.concat([try_a_items, cur_trip_b])
                    elif a_to_index == (n_trip_b - 1):
                        cur_trip_b_new = pd.concat([cur_trip_b, try_a_items])
                    else:
                        cur_trip_tmp_b_p1 = cur_trip_b.iloc[:a_to_index]
                        cur_trip_tmp_b_p2 = cur_trip_b.iloc[a_to_index:]
                        cur_trip_b_new = pd.concat([cur_trip_tmp_b_p1, try_a_items, cur_trip_tmp_b_p2])
                else:
                    cur_trip_b_new = try_a_items
            else:
                cur_trip_b_new = cur_trip_b
            if items_chosen_b:
                if n_trip_a:
                    b_to_index = np.random.randint(n_trip_a)
                    if not b_to_index:
                        cur_trip_a_new = pd.concat([try_b_items, cur_trip_a])
                    elif b_to_index == (n_trip_a - 1):
                        cur_trip_a_new = pd.concat([cur_trip_a, try_b_items])
                    else:
                        cur_trip_tmp_a_p1 = cur_trip_a.iloc[:b_to_index]
                        cur_trip_tmp_a_p2 = cur_trip_a.iloc[b_to_index:]
                        cur_trip_a_new = pd.concat([cur_trip_tmp_a_p1, try_b_items, cur_trip_tmp_a_p2])
                else:
                    cur_trip_a_new = try_b_items
            else:
                cur_trip_a_new = cur_trip_a

            if np.sum(np.array(cur_trip_a_new['Weight'])) < trip_max_weight and \
               np.sum(np.array(cur_trip_b_new['Weight'])) < trip_max_weight:
                cur_metric_a = weighted_trip_length(cur_trip_a_new[['Latitude', 'Longitude']],
                                                    list(cur_trip_a_new['Weight']))
                cur_metric_b = weighted_trip_length(cur_trip_b_new[['Latitude', 'Longitude']],
                                                    list(cur_trip_b_new['Weight']))

                if (cur_metric_a + cur_metric_b) < best_metric:
                    best_metric = cur_metric_a + cur_metric_b
                    best_trip_a = cur_trip_a_new.copy(deep=True)
                    best_trip_b = cur_trip_b_new.copy(deep=True)
                    if not best_trip_a.shape[0] or not best_trip_b.shape[0]:
                        break
    best_trip = pd.concat([best_trip_a, best_trip_b])
    if (best_metric - base_metric) < 0:
        best_trip_a, best_metric_a = single_trip_optimize(best_trip_a, 9, 0, 1)
        best_trip_b, best_metric_b = single_trip_optimize(best_trip_b, 9, 0, 1)
        best_metric = best_metric_a + best_metric_b
        best_trip = pd.concat([best_trip_a, best_trip_b])
        print 'weariness gain: %f' % (best_metric - base_metric)
    return best_trip


def split_trip(gifts, new_trip, trip_list, sort=False):
    """
    Finding Best split point
    """
    if sort:
        best_trip = [gifts.sort_values('Latitude', ascending=False)]
    else:
        best_trip = gifts
    old_trip = int(gifts['TripId'].iloc[0])

    best_metric = base_metric = weighted_trip_length(gifts[['Latitude', 'Longitude']], list(gifts['Weight']))
    n_gifts = gifts.shape[0]
    for new_trip_index in range(2, n_gifts - 2):
        # load current best trips
        cur_trip_a = gifts.iloc[:new_trip_index].copy(deep=True)
        cur_trip_b = gifts.iloc[new_trip_index:].copy(deep=True)
        cur_metric_a = weighted_trip_length(cur_trip_a[['Latitude', 'Longitude']],
                                            list(cur_trip_a['Weight']))
        cur_metric_b = weighted_trip_length(cur_trip_b[['Latitude', 'Longitude']],
                                            list(cur_trip_b['Weight']))
        if (cur_metric_a + cur_metric_b) < best_metric:
            best_metric = cur_metric_a + cur_metric_b
            best_trip = [cur_trip_a, cur_trip_b]

    if (best_metric - base_metric) < 0:
        print 'weariness gain: %f' % (best_metric - base_metric)
        best_trip[1]['TripId'] = np.repeat(new_trip, best_trip[1].shape[0])
        for j, trip_long in enumerate(trip_list):
            if old_trip in trip_long:
                trip_list[j].append(new_trip)
        new_trip += 1
    else:
        best_trip = [best_trip]
    return pd.concat(best_trip), new_trip, trip_list


def gift_switch_optimize_dynamic(gifts_from, gifts_to, n_tries=15, poisson_items=1.5, max_weight=90,
                                 trip_max_weight=990, batch_size=5):
    """
    Optimizing between 2 trips
    """
    gifts_to = gifts_to.sort_values('Latitude', ascending=False)
    to_trip_id = gifts_to['TripId'].iloc[0]
    gifts_from = gifts_from.sort_values('Latitude', ascending=False)

    best_weight_to = np.sum(np.array(gifts_to['Weight']))
    best_trip_to, base_metric_to = single_trip_optimize(gifts_to, batch_size, 0, 5)
    best_trip_from, base_metric_from = single_trip_optimize(gifts_from, batch_size, 0, 5)

    best_metric = base_metric = base_metric_to + base_metric_from

    # greedy
    for change_iter in range(n_tries):
        # print 'iteration %d' % change_iter
        if best_weight_to > (trip_max_weight - int(poisson_items) + 1):
            break

        # load current best trips
        cur_trip_to = best_trip_to.copy(deep=True)
        cur_trip_from = best_trip_from.copy(deep=True)
        n_trip_from = best_trip_from.shape[0]

        # load change arrays
        items_chosen = np.random.poisson(poisson_items)
        if items_chosen and (n_trip_from > items_chosen):
            try_to = np.random.choice(np.arange(n_trip_from - items_chosen + 1), 1, replace=False)
            try_to = range(try_to, (try_to + items_chosen))
            try_from = range(n_trip_from)
            for moved_vals in try_to:
                try_from.remove(moved_vals)

            if np.sum(np.array(cur_trip_from['Weight'].iloc[try_to])) < max_weight:
                moved_gifts = cur_trip_from.iloc[try_to].copy(deep=True)
                moved_gifts['TripId'] = np.repeat(to_trip_id, items_chosen)
                cur_trip_to = pd.concat([cur_trip_to, moved_gifts])
                if np.sum(np.array(cur_trip_to['Weight'])) < trip_max_weight:
                    cur_trip_from = cur_trip_from.iloc[try_from].copy(deep=True)

                    cur_trip_to = cur_trip_to.sort_values('Latitude', ascending=False)
                    cur_trip_from = cur_trip_from.sort_values('Latitude', ascending=False)

                    cur_trip_to, cur_metric_to = single_trip_optimize(cur_trip_to, batch_size, 0, 5)
                    cur_trip_from, cur_metric_from = single_trip_optimize(cur_trip_from, batch_size, 0, 5)

                    if (cur_metric_to + cur_metric_from) < best_metric:
                        best_metric = cur_metric_to + cur_metric_from
                        best_trip_to = cur_trip_to.copy(deep=True)
                        best_trip_from = cur_trip_from.copy(deep=True)
                        if best_trip_from.shape[0] == 0:
                            print 'weariness gain: %f' % (best_metric - base_metric)
                            return pd.concat([best_trip_from, best_trip_to])
                        best_weight_to = np.sum(np.array(cur_trip_to['Weight']))

    if (best_metric - base_metric) < 0:
        print 'weariness gain: %f' % (best_metric - base_metric)
    return pd.concat([best_trip_from, best_trip_to])


def combine_trips(gifts_a, gifts_b, trips):
    """
    Finding if combining is good
    """
    trip_a = int(gifts_a['TripId'].iloc[0])
    trip_b = int(gifts_b['TripId'].iloc[0])
    metric_a = weighted_trip_length(gifts_a[['Latitude', 'Longitude']],
                                    list(gifts_a['Weight']))
    metric_b = weighted_trip_length(gifts_b[['Latitude', 'Longitude']],
                                    list(gifts_b['Weight']))
    total_weight = sum(list(gifts_a['Weight']) + list(gifts_b['Weight']))
    if total_weight < 990:
        print 'work on trips %d, %d' % (trip_a, trip_a)
        combined_gifts = pd.concat([gifts_a, gifts_b])
        combined_gifts = combined_gifts.sort_values('Latitude', ascending=False)
        combined_gifts, combine_metric = single_trip_optimize(combined_gifts, 9,  0, 1)
        if (metric_a + metric_b) > combine_metric:
            print 'weariness gain: %f' % (combine_metric - (metric_a + metric_b))
            combined_gifts['TripId'] = np.repeat(trip_a, combined_gifts.shape[0])
            trips = remove_trip(trip_b, trips)
            return combined_gifts, trips
    return pd.concat([gifts_a, gifts_b]), trips


def remove_trip(removed_trip, trips):
    """
    Remove wasted trip
    """
    for i in range(len(trips)):
        for j in range(len(trips[i])):
            if trips[i][j] == removed_trip:
                if len(trips[i]) == 1:
                    trips.pop(i)
                else:
                    trips[i].pop(j)
                print 'removed trip %d' % removed_trip
                return trips
    return trips


def remove_empty_trips(gifts, trips):
    """
    Remove wasted trip
    """
    for i in range(len(trips)):
        for j in range(len(trips[i])):
            cur_gifts = gifts[gifts['TripId'] == trips[i][j]]
            cur_trip = trips[i][j]
            if cur_gifts.shape[0] == 0:
                if len(trips[i]) == 1:
                    trips.pop(i)
                else:
                    trips[i].pop(j)
                print 'removed trip %d' % cur_trip
                return trips
    return trips

"""
Main program, require sorted trips
"""
# read files
gifts = pd.read_csv('gifts.csv')
print 'orginizing trips with %d weight limit' % 950
gifts = trips_orginizer(gifts, 950)
gifts.index = np.array(gifts.index) + 1
print 'optimizing tracks'
print weighted_reindeer_weariness(gifts)

gifts_save = 'mine_opt_v3.csv'
gifts_out = 'mine_opt_v3_rslts.csv'
trips_out = 'mine_opt_v3_trips.csv'

trips = gifts['TripId'].unique()
trips = list(np.sort(trips))
new_trip = trips[-1] + 1
print 'number of trips is: ', len(trips)
trips = map(lambda x: [x], trips)


trips = remove_empty_trips(gifts, trips)

iterations = 50
for it in range(iterations):
    it_switch = 50
    for i_switch in range(it_switch):
        print 'Iteration %d' % it_switch
        # print gift_trips
        for i in range(0, len(trips)):
            # single iteration per trip
            # Working from the start
            if not (i % 20):
                print 'trip %d optimization' % i
                print weighted_reindeer_weariness(gifts)
                gifts.to_csv(gifts_save)
            for gift_from in trips[i - 1]:
                for gift_to in trips[i]:
                    cur_trip_from = gifts[gifts['TripId'] == gift_from]
                    cur_trip_to = gifts[gifts['TripId'] == gift_to]
                    if cur_trip_from.shape[0] and cur_trip_to.shape[0]:
                        cur_trip_from_to = gift_switch_optimize_v2(cur_trip_from, cur_trip_to, n_tries=100,
                                                                   poisson_items=(((it_switch - i_switch) *
                                                                                  1.0 / 20) + 1))
                        gifts = gifts[gifts.TripId != gift_to]
                        gifts = gifts[gifts.TripId != gift_from]
                        gifts = pd.concat([cur_trip_from_to, gifts])

    trips = remove_empty_trips(gifts, trips)

    n_splits = 0
    max_splits = 10
    # Splitting
    print 'spliting'
    gifts_new = []
    for i in range(0, len(trips)):
        for gift_trip in trips[i]:
            if np.sum(gifts[gifts['TripId'] == gift_trip]['Weight']):
                cur_new_trip = new_trip
                print 'trip %d optimization with weight %f' % \
                      (gift_trip, np.sum(gifts[gifts['TripId'] == gift_trip]['Weight']))
                if n_splits < max_splits:
                    tmp_trip, new_trip, trips = split_trip(gifts[gifts['TripId'] == gift_trip], new_trip, trips)
                else:
                    tmp_trip = gifts[gifts['TripId'] == gift_trip]
                if (new_trip - cur_new_trip) > 0:
                    n_splits += 1
                gifts_new.append(tmp_trip)
    gifts = pd.concat(gifts_new)

    print weighted_reindeer_weariness(gifts)

    with open(trips_out, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in trips:
            csvwriter.writerow(row)

    # Combining
    new_trips = list(trips)
    for i in range(0, len(trips), 2):
        for trip_a in trips[i]:
            for trip_b in trips[i - 1]:
                # single iteration per trip
                # Working from the start
                cur_trip_a = gifts[gifts['TripId'] == trip_a]
                cur_trip_b = gifts[gifts['TripId'] == trip_b]
                # print 'trip_a %d optimization with length %d and weight %f' % (trip_a, cur_trip_a.shape[0],
                #                                                                np.sum(cur_trip_a['Weight']))
                # print 'trip_b %d optimization with length %d and weight %f' % (trip_b, cur_trip_b.shape[0],
                #                                                                np.sum(cur_trip_b['Weight']))
                if cur_trip_a.shape[0] and cur_trip_b.shape[0]:
                    cur_trip, new_trips = combine_trips(cur_trip_a, cur_trip_b, new_trips)
                    gifts = gifts[gifts.TripId != trip_a]
                    gifts = gifts[gifts.TripId != trip_b]
                    gifts = pd.concat([cur_trip, gifts])
    trips = new_trips
    print weighted_reindeer_weariness(gifts)

    with open(trips_out, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in trips:
            csvwriter.writerow(row)

    new_trips = list(trips)
    for i in range(1, len(trips), 2):
        for trip_a in trips[i]:
            for trip_b in trips[i - 1]:
                # single iteration per trip
                # Working from the start
                cur_trip_a = gifts[gifts['TripId'] == trip_a]
                cur_trip_b = gifts[gifts['TripId'] == trip_b]
                # print 'trip_a %d optimization with length %d and weight %f' % (trip_a, cur_trip_a.shape[0],
                #                                                                np.sum(cur_trip_a['Weight']))
                # print 'trip_b %d optimization with length %d and weight %f' % (trip_b, cur_trip_b.shape[0],
                #                                                                np.sum(cur_trip_b['Weight']))
                if cur_trip_a.shape[0] and cur_trip_b.shape[0]:
                    cur_trip, new_trips = combine_trips(cur_trip_a, cur_trip_b, new_trips)
                    gifts = gifts[gifts.TripId != trip_a]
                    gifts = gifts[gifts.TripId != trip_b]
                    gifts = pd.concat([cur_trip, gifts])
    trips = new_trips
    print weighted_reindeer_weariness(gifts)
    gifts.to_csv(gifts_save)
gifts.to_csv(gifts_save)

gifts = pd.DataFrame.from_csv(gifts_save)

print 'writing results to file'
gift_trips = np.array(gifts)
gift_trips = gift_trips[:, [0, 3]]
gift_trips = pd.DataFrame(gift_trips)
gift_trips.columns = ['GiftId', 'TripId']

gift_trips = gift_trips.astype('int32')
gift_trips.index = gift_trips["GiftId"]
del gift_trips["GiftId"]
gift_trips.to_csv(gifts_out)