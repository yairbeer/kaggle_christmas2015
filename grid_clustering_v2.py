"""will remove this part after we get haversine package into scripts
"""

from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np

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


def weighted_reindeer_weariness(all_trips):
    uniq_trips = all_trips.TripId.unique()

    if np.any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")

    dist = 0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId == t]
        dist = dist + weighted_trip_length(this_trip[['Latitude', 'Longitude']], this_trip.Weight.tolist())

    return dist


def grid_cluster(gifts, res_long):
    grid_lon = np.arange(-180, 180, res_long)
    gifts = pd.DataFrame(gifts)
    cluster = pd.DataFrame(np.zeros((gifts.shape[0], 1)))
    cluster.index = gifts.index
    cluster.columns = ['cluster_lon']
    gifts = pd.concat([gifts, cluster], axis=1)
    for lon in grid_lon:
        if not(lon % 10):
            print lon
        lon_eq = ((np.array(gifts['Longitude']) >= lon) * 1 *
                  (np.array(gifts['Longitude']) < lon + res_long) * 1).astype('bool')
        lon_index = gifts[lon_eq].index
        gifts.at[lon_index, 'cluster_lon'] = lon
    return gifts


def trips_in_cluster(gifts, res_long):
    """
    Use close trips in each cell
    """
    cur_trip = 0
    cur_weight = 0
    gift_trips = [[]]
    grid_lon = np.arange(-180, 180, res_long)
    for lon in grid_lon:
        # filter
        gifts_clust = gifts[gifts['cluster_lon'] == lon]
        # print 'For cluster with latitude %d and longitude %d There are %d gifts weighing %f' \
        #       % (lat, lon, gifts_clust.shape[0], np.sum(gifts_clust['Weight']))
        gifts_clust = gifts_clust.sort('Latitude', ascending=False)
        gifts_clust = np.array(gifts_clust)
        for i in range(gifts_clust.shape[0]):
            if (cur_weight + gifts_clust[i, 3]) <= 990:
                gift_trips[cur_trip].append(list(gifts_clust[i, :]) + [cur_trip])
                cur_weight += gifts_clust[i, 3]
            else:
                # print gift_trips[cur_trip]
                gift_trips[cur_trip] = np.array(gift_trips[cur_trip])
                # print gift_trips[cur_trip]
                print 'For trip %d, the total weight was %f' % (cur_trip, cur_weight)
                cur_weight = 0
                cur_trip += 1
                gift_trips.append([])
                gift_trips[cur_trip].append(list(gifts_clust[i, :]) + [cur_trip])
                cur_weight += gifts_clust[i, 3]
    gift_trips = np.vstack(tuple(gift_trips))
    gift_trips = np.array(gift_trips)
    return gift_trips

"""
Start Main program
"""
# GiftId   Latitude   Longitude     Weight  cluster_lat  cluster_lon
gifts = pd.read_csv('gifts.csv')

n_gifts = gifts.shape[0]
resolution_longitude = 0.5

print 'Add cluster index'
gifts = grid_cluster(gifts, resolution_longitude)

print 'There are %d gifts to distribute' % n_gifts
print 'Starting to plan trips by clusters'

gift_trips = trips_in_cluster(gifts, resolution_longitude)
gift_trips = gift_trips[:, [0, -1]]
gift_trips = pd.DataFrame(gift_trips)
gift_trips.columns = ['GiftId', 'TripId']
print gift_trips

all_trips = gift_trips.merge(gifts, on='GiftId')
print(weighted_reindeer_weariness(all_trips))

gift_trips = gift_trips.astype('int32')
gift_trips.index = gift_trips["GiftId"]
del gift_trips["GiftId"]
gift_trips.to_csv('clustering_with_ordering_lat_v2.csv')

# Basecase: 144525525772.0
# Resolution 10 clustering: 34230724056.0
# Resolution 5 clustering with ordering by latitude: 17723267396.9
# resolution_latitude = 45; resolution_longitude = 1, clustering with ordering by latitude: 13227163205.6
# resolution_longitude = 0.5, clustering with ordering by latitude: 12787535216.9
